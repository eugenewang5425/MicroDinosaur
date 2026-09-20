"""Sensor contract and geometry checks; no hardware performance assertions."""
from dataclasses import replace
import unittest

import mujoco
import numpy as np

from tof_skill_contract import (CENTER_RAYS, SkillRequestGate, evidence,
                                pitch_rotation, height_envelope)
from probe_tof_feasibility import query_model, source_model, scan, make_frame


def planar_frame(height=0., seq=0, t=0., pitch=35):
    rotation = pitch_rotation(pitch)
    origin = np.array([0.,0.,.263815])
    directions = CENTER_RAYS@rotation.T
    ranges = (height-origin[2])/directions[:,:,2]
    return make_frame(seq,t,ranges,origin,rotation)


class TestTofContract(unittest.TestCase):
    def test_flat_rays_match_analytic_plane(self):
        m,d,_ = query_model('flat')
        frame = planar_frame()
        r,_ = scan(m,d,frame.origin_m,frame.rotation)
        np.testing.assert_allclose(r[:,:,4],frame.ranges_m,atol=1e-12)

    def test_copied_geometry_matches_original_10mm_model(self):
        original,od,_ = source_model('steps_10')
        query,qd,_ = query_model('steps_10')
        mask=np.array([1,0,0,0,0,0],np.uint8)
        geom=np.empty(1,np.int32)
        for x in [-.3,.1,.25]:
            origin=np.array([x,0.,.263815])
            for direction in (CENTER_RAYS@pitch_rotation(35).T).reshape(-1,3):
                a=mujoco.mj_ray(original,od,origin,direction,mask,1,-1,geom)
                b=mujoco.mj_ray(query,qd,origin,direction,None,1,-1,geom)
                self.assertAlmostEqual(a,b,places=11)

    def test_synthetic_step_heights_match_design(self):
        for kind,sign in [('steps',1),('downsteps',-1)]:
            for h in [10,20,30]:
                m,d,_=query_model(f'{kind}_{h}')
                for x,level in [(.1,0),(.4,1),(.55,2),(.8,3)]:
                    distance=mujoco.mj_ray(m,d,np.array([x,0.,1.]),np.array([0.,0.,-1.]),None,1,-1,np.empty(1,np.int32))
                    self.assertAlmostEqual(1-distance,sign*h*.001*level,places=10)

    def test_pitched_flat_ground_never_becomes_a_step(self):
        for pitch in [20,35,45]:
            frame=planar_frame(pitch=pitch)
            self.assertEqual(evidence(frame,.03).state,'NO_CONFIRMED_FEATURE')
            # A one-degree calibration error is within the assumed pose envelope.
            frame=replace(frame,rotation=pitch_rotation(pitch+1))
            self.assertEqual(evidence(frame,.03).state,'NO_CONFIRMED_FEATURE')

    def test_unverified_or_bad_data_is_unknown(self):
        good=planar_frame()
        bad=[replace(good,calibrated=False), replace(good,valid=np.zeros((8,8),bool)),
             replace(good,ranges_m=np.full((8,8),np.nan)),
             replace(good,ranges_m=np.full((8,8),np.inf)),
             replace(good,arrival_s=-1), replace(good,pose_time_s=.1),
             replace(good,rotation=-np.eye(3)), replace(good,pose_error_deg=-1),
             replace(good,ranges_m=np.ones(64)), replace(good,valid=np.ones((8,8),int))]
        for frame in bad:
            self.assertEqual(evidence(frame,.03).state,'UNKNOWN')
        self.assertEqual(evidence(good,.201).reason,'stale_frame')
        self.assertEqual(evidence(good,.01).reason,'invalid_clock_order')

    def test_blind_center_cannot_be_hidden_by_good_edge_zones(self):
        frame=planar_frame()
        valid=np.ones((8,8),bool);valid[:3,2:6]=False
        self.assertEqual(evidence(replace(frame,valid=valid),.03).state,'UNKNOWN')

    def test_duplicates_cannot_accumulate_confirmations(self):
        gate=SkillRequestGate()
        first=planar_frame(.15)
        self.assertEqual(gate.update(first,.03).state,'PENDING')
        for t in [.04,.05,.06]:
            self.assertEqual(gate.update(first,t).reason,'duplicate_or_out_of_order_frame')
        self.assertEqual(gate.count,1)
        second=planar_frame(.15,seq=1,t=1/15)
        self.assertEqual(gate.update(second,1/15+.03).state,'PENDING')
        third=planar_frame(.15,seq=2,t=2/15)
        final=gate.update(third,2/15+.03)
        self.assertEqual(final.state,'REQUEST_ONLY')
        self.assertEqual(final.requested_skill,'hold_or_avoid_review')
        self.assertFalse(final.execution_authorized)

    def test_gaps_sensor_restart_and_stale_frames_break_confirmation(self):
        for disruption in ['gap','restart','stale']:
            gate=SkillRequestGate()
            gate.update(planar_frame(.15),.03)
            gate.update(planar_frame(.15,1,1/15),1/15+.03)
            if disruption=='stale':
                stale=planar_frame(.15,2,2/15)
                self.assertEqual(gate.update(stale,.4).state,'UNKNOWN')
            t=.5 if disruption in ['gap','stale'] else 2/15
            third=planar_frame(.15,2,t)
            if disruption=='restart':
                third=replace(third,stream_id='new_sensor_session',sequence=0)
            self.assertEqual(gate.update(third,t+.03).state,'PENDING')
            self.assertEqual(gate.count,1)

    def test_envelope_includes_zone_and_pose_uncertainty(self):
        frame=planar_frame()
        optimistic=height_envelope(frame.ranges_m,frame.rotation,0,0,include_zone_extent=False)
        full=height_envelope(frame.ranges_m,frame.rotation)
        self.assertTrue(np.all(full>optimistic))
        self.assertTrue(np.all(height_envelope(frame.ranges_m,frame.rotation,2)>full))


if __name__=='__main__':
    unittest.main()
