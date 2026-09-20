"""Offline perception/skill-request prototype; NOT connected to robot commands.

Coordinates: sensor +X forward, +Y left, +Z up; normalized radial ranges in m.
An eventual driver must validate zone ordering and radial/axial range convention.
The uncertainty envelope is a sensitivity scenario, not a sensor guarantee.
"""
from dataclasses import dataclass
import math

import numpy as np


def rays(subsamples=1):
    """Regular tangent-plane approximation of 45 x 45 degree detection FoV.

    Return [row, column, subray, xyz]. Row zero is up; column zero is left.
    This is not a factory-calibrated per-zone optical model.
    """
    edges = np.linspace(-math.tan(math.pi / 8), math.tan(math.pi / 8), 9)
    offsets = (np.arange(subsamples) + .5) / subsamples
    out = np.empty((8, 8, subsamples * subsamples, 3))
    for row in range(8):
        for col in range(8):
            k = 0
            for v in offsets:
                for u in offsets:
                    d = np.array([1., -(edges[col] + u * (edges[col+1]-edges[col])),
                                  -(edges[row] + v * (edges[row+1]-edges[row]))])
                    out[row, col, k] = d / np.linalg.norm(d)
                    k += 1
    return out


def pitch_rotation(down_deg):
    a = math.radians(down_deg)
    return np.array([[math.cos(a), 0., math.sin(a)], [0., 1., 0.],
                     [-math.sin(a), 0., math.cos(a)]])


def zone_half_cones():
    """Maximum angular distance of each pixel corner from its center ray."""
    edges = np.linspace(-math.tan(math.pi/8), math.tan(math.pi/8), 9)
    center = rays()[:, :, 0]
    result = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            for a in (edges[i], edges[i+1]):
                for b in (edges[j], edges[j+1]):
                    corner = np.array([1., -b, -a])
                    corner /= np.linalg.norm(corner)
                    result[i, j] = max(result[i, j], math.acos(np.clip(center[i,j] @ corner, -1, 1)))
    return result


CENTER_RAYS = rays()[:, :, 0]
HALF_CONES = zone_half_cones()


def height_envelope(range_m, rotation, pose_error_deg=1., height_error_m=.003,
                    range_relative_error=.05, include_zone_extent=True):
    """Assumed range + direction + origin error, WITHOUT averaging bias away.

    Near-range 15 mm and far-range 5% refer to specified ST continuous-mode
    full-target conditions. Mixed edges, reflectance, cover glass and multipath
    are uncharacterized; this is NOT a worst-case bound for actual hardware.
    """
    r = np.asarray(range_m)
    d = CENTER_RAYS @ np.asarray(rotation).T
    dr = np.where(r <= .2, .015, range_relative_error*r)
    cone = math.radians(pose_error_deg) + (HALF_CONES if include_zone_extent else 0.)
    chord = 2*np.sin(cone/2)
    return dr*np.abs(d[:, :, 2]) + (r+dr)*chord + height_error_m


@dataclass(frozen=True)
class TofFrame:
    stream_id: str
    sequence: int
    capture_start_s: float  # earliest acquisition time, same monotonic clock
    arrival_s: float
    pose_time_s: float     # pose reconstructed at capture_start_s
    ranges_m: np.ndarray  # 8x8 radial range, no-return represented by valid=False
    valid: np.ndarray     # adapter-normalized status; NOT merely range > 0
    origin_m: np.ndarray  # in a locally level ground reference
    rotation: np.ndarray  # sensor to that reference, calibrated extrinsics
    calibrated: bool = False
    pose_error_deg: float = 1.
    height_error_m: float = .003


@dataclass(frozen=True)
class Decision:
    state: str
    reason: str
    supporting_zones: int = 0
    requested_skill: str | None = None
    execution_authorized: bool = False


def evidence(frame, now_s, max_age_s=.2):
    if not frame.calibrated:
        return Decision('UNKNOWN', 'extrinsics_or_ground_reference_unverified')
    times = [frame.capture_start_s, frame.arrival_s, frame.pose_time_s, now_s]
    if not np.isfinite(times).all() or not frame.capture_start_s <= frame.arrival_s <= now_s:
        return Decision('UNKNOWN', 'invalid_clock_order')
    if now_s-frame.capture_start_s > max_age_s:
        return Decision('UNKNOWN', 'stale_frame')
    if abs(frame.pose_time_s-frame.capture_start_s) > .01:
        return Decision('UNKNOWN', 'pose_not_aligned_to_capture')
    r = np.asarray(frame.ranges_m)
    valid = np.asarray(frame.valid)
    rot = np.asarray(frame.rotation)
    origin = np.asarray(frame.origin_m)
    if r.shape != (8,8) or valid.shape != (8,8) or valid.dtype != np.bool_:
        return Decision('UNKNOWN', 'invalid_frame_shape_or_status_type')
    if (rot.shape != (3,3) or origin.shape != (3,) or not np.isfinite(rot).all()
            or not np.isfinite(origin).all() or not np.allclose(rot.T@rot, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rot), 1., atol=1e-6)):
        return Decision('UNKNOWN', 'invalid_pose')
    if not all(math.isfinite(x) and x >= 0 for x in (frame.pose_error_deg, frame.height_error_m)):
        return Decision('UNKNOWN', 'invalid_uncertainty')
    valid = valid & np.isfinite(r) & (r >= .02) & (r <= 4.)
    roi = np.zeros((8,8), bool)
    roi[:, 2:6] = True
    valid &= roi
    if np.count_nonzero(valid) < 24:
        return Decision('UNKNOWN', 'insufficient_central_coverage')
    safe_r = np.where(valid, r, 0.)
    xyz = origin + safe_r[:, :, None]*(CENTER_RAYS@rot.T)
    radius = height_envelope(safe_r, rot, frame.pose_error_deg, frame.height_error_m)
    lower, upper = xyz[:, :, 2]-radius, xyz[:, :, 2]+radius
    for state, mask in [('OBSTACLE_CANDIDATE', lower > .05),
                        ('DROP_CANDIDATE', upper < -.005),
                        ('TERRAIN_CANDIDATE', lower > .005)]:
        n = int(np.count_nonzero(mask & valid))
        if n >= 4:
            return Decision(state, 'assumed_height_envelope_excludes_current_ground', n)
    return Decision('NO_CONFIRMED_FEATURE', 'does_not_establish_walkable_clearance')


class SkillRequestGate:
    """Three distinct consistent frames request review; never execute a policy.

    The action-history buffer, head controller and body heading reference remain
    owned by the existing control loop. This class has no action output.
    """
    def __init__(self, confirmation_frames=3, max_gap_s=.12):
        if confirmation_frames < 1 or max_gap_s <= 0:
            raise ValueError('invalid confirmation configuration')
        self.confirmation_frames = confirmation_frames
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self):
        self.stream = None
        self.last_sequence = -1
        self.last_capture = None
        self.last_state = None
        self.count = 0

    def update(self, frame, now_s):
        current = evidence(frame, now_s)
        if frame.stream_id != self.stream:
            self.reset()
            self.stream = frame.stream_id
        if current.state == 'UNKNOWN':
            self.count = 0
            self.last_state = None
            return current
        if frame.sequence <= self.last_sequence or (self.last_capture is not None
                and frame.capture_start_s <= self.last_capture):
            return Decision('UNKNOWN', 'duplicate_or_out_of_order_frame')
        gap_ok = self.last_capture is not None and frame.capture_start_s-self.last_capture <= self.max_gap_s
        self.last_sequence = frame.sequence
        self.last_capture = frame.capture_start_s
        if not current.state.endswith('_CANDIDATE'):
            self.count = 0
            self.last_state = None
            return current
        self.count = self.count+1 if gap_ok and self.last_state == current.state else 1
        self.last_state = current.state
        if self.count < self.confirmation_frames:
            return Decision('PENDING', current.state, current.supporting_zones)
        request = 'terrain_review' if current.state == 'TERRAIN_CANDIDATE' else 'hold_or_avoid_review'
        return Decision('REQUEST_ONLY', current.state, current.supporting_zones, request)
