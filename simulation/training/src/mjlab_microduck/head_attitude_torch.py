"""Batched counterpart of the audited NumPy head loop. Sensor inputs only.

The training action adapter owns packet delay and reset; these kernels know
nothing about MuJoCo state. Joint order is neck, head pitch, yaw, roll.
"""
import math
import torch


def exp_so3(v):
    a = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    k = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + torch.sinc(a / math.pi)[..., None] * k + .5 * torch.sinc(a / (2*math.pi))[..., None]**2 * (k@k)


def log_so3(r):
    skew = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                        r[..., 1, 0]-r[..., 0, 1]), -1)
    sine = .5*torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    theta = torch.atan2(sine, cosine)
    result = skew*(theta/(2*sine.clamp_min(1e-12)))[..., None]
    result = torch.where((theta < 1e-7)[..., None], .5*skew, result)
    near = math.pi-theta < 1e-5
    if torch.any(near):
        _, vectors = torch.linalg.eigh((r[near]+r[near].transpose(-1, -2))/2)
        axis = vectors[..., -1]
        axis *= torch.where((axis*skew[near]).sum(-1, keepdim=True)<0, -1., 1.)
        result[near] = theta[near, None]*axis
    return result


def tilt_from_accel(accel):
    x, y, z = accel.unbind(-1)
    zeros = torch.zeros_like(x)
    roll = torch.atan2(y, z)
    pitch = torch.atan2(-x, torch.hypot(y, z))
    return exp_so3(torch.stack((zeros, pitch, zeros), -1))@exp_so3(torch.stack((roll, zeros, zeros), -1))


class TorchImu:
    """Same complementary tilt update as RelativeImuHeading, fixed 50 Hz.

    bias must be supplied by stationary calibration (or known simulated bias).
    Startup tilt is an estimate from accelerometer data, never world pose.
    """
    def __init__(self, n, device, dtype=torch.float32):
        self.r = torch.eye(3, device=device, dtype=dtype).repeat(n, 1, 1)
        self.accel = torch.zeros(n, 3, device=device, dtype=dtype)
        self.bias = torch.zeros_like(self.accel)

    def initialize(self, ids, accel, bias=None):
        self.accel[ids] = accel
        self.r[ids] = tilt_from_accel(accel)
        self.bias[ids] = 0 if bias is None else bias

    def update(self, gyro, accel, dt):
        self.accel += (1-math.exp(-dt/.1))*(accel-self.accel)
        up = self.r[:, 2, :]
        measured = self.accel/self.accel.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        valid = ((accel.norm(dim=-1)/9.81-1).abs() <= .15) & ((measured*up).sum(-1) >= math.cos(math.radians(20)))
        correction = .3*torch.linalg.cross(measured, up)*valid[:, None]
        self.r = self.r@exp_so3((gyro-self.bias+correction)*dt)
        return self.r


class TorchHeadController:
    def __init__(self, kinematics, n, device, dtype=torch.float32):
        tensor = lambda a: torch.as_tensor(a, device=device, dtype=dtype)
        self.fixed, self.axes = tensor(kinematics.fixed), tensor(kinematics.axes)
        self.reference, self.tip, self.base = map(tensor, (kinematics.reference, kinematics.tip, kinematics.base))
        self.limits = tensor(kinematics.limits)
        self.kp, self.kd = tensor((.5, 1., 2.)), tensor((.01, .03, .04))
        self.bound = tensor((.35, .5, .18))
        self.correction = torch.zeros(n, 3, device=device, dtype=dtype)
        self.last_target = torch.zeros_like(self.correction)
        self.nominal_filtered = torch.zeros_like(self.correction)
        self.error = torch.zeros_like(self.correction)
        self.stale = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids, target):
        self.correction[ids] = 0
        self.last_target[ids] = target
        self.nominal_filtered[ids] = target
        self.error[ids] = 0
        self.stale[ids] = False

    def forward(self, angles):
        r = self.base.T.expand(angles.shape[0], -1, -1)
        axes = []
        for i in range(4):
            r = r@self.fixed[i]
            axes.append(r@self.axes[i])
            r = r@exp_so3(self.axes[i]*(angles[:, i:i+1]-self.reference[i]))
        r = r@self.tip
        return r, r.transpose(-1, -2)@torch.stack(axes[1:], -1)

    def update(self, nominal, angles, orientation, gyro, desired, omega, dt, age):
        _, j = self.forward(angles)
        predicted = orientation@exp_so3(gyro*age.clamp(max=.02)[:, None])
        self.error = log_so3(predicted.transpose(-1, -2)@desired)
        gyro_error = gyro-(predicted.transpose(-1, -2)@omega[..., None]).squeeze(-1)
        control = self.kp*self.error-self.kd*gyro_error  # ki is explicitly zero in the operating recipe.
        eye = torch.eye(3, device=j.device, dtype=j.dtype)
        correction = (j.transpose(-1, -2)@torch.linalg.solve(j@j.transpose(-1, -2)+.05**2*eye,
                                                            control[..., None])).squeeze(-1)
        self.stale = age > .04
        correction = torch.where(self.stale[:, None], 0., correction)
        correction = correction.clamp(-self.bound, self.bound)
        filtered = self.correction+(1-math.exp(-dt/.02))*(correction-self.correction)
        self.correction += (filtered-self.correction).clamp(-2*dt, 2*dt)
        self.nominal_filtered += (1-math.exp(-dt/.10))*(nominal-self.nominal_filtered)
        raw = self.nominal_filtered+self.correction
        clipped = raw.clamp(self.limits[1:, 0]+.035, self.limits[1:, 1]-.035)
        self.last_target += (clipped-self.last_target).clamp(-4*dt, 4*dt)
        return self.last_target.clone()
