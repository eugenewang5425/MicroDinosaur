"""A mirrored, perfectly tracked limb target must remain perfectly tracked."""
import unittest
from types import SimpleNamespace

import mjlab
import torch
from tensordict import TensorDict
from mjlab_microduck.tasks.mdp import limb_pose_tracking
from mjlab_microduck.tasks.symmetry_microdinosaur import microdinosaur_vel_symmetry


class ArmCommandReflectionTest(unittest.TestCase):
    def test_asymmetric_arm_target_preserves_tracking_reward(self):
        actor = torch.zeros(3,81)
        actor[:,76:78] = torch.tensor([[.4,-.2], [0.,.7], [-.6,-.1]])
        actor[:,23:25] = actor[:,76:78]
        obs = TensorDict({'actor':actor,'critic':torch.zeros(3,1)},batch_size=[3])
        aug, _ = microdinosaur_vel_symmetry(None,obs,None)
        reflected = aug['actor'][3:]
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=reflected[:,6:25],
                    default_joint_pos=torch.zeros(3,19)),
                    find_joints_by_actuator_names=lambda patterns:([17,18],['arm_l','arm_r']))
        env = SimpleNamespace(scene={'robot':asset},device='cpu',
                    command_manager=SimpleNamespace(get_command=lambda name:reflected[:,76:78]))
        result = limb_pose_tracking(env,'arm_pose',('.*arm_l.*','.*arm_r.*'))
        torch.testing.assert_close(result,torch.ones(3))
        # Physical reflection exchanges left/right and negates the joint angles.
        torch.testing.assert_close(reflected[:,76:78],-actor[:,[77,76]])


if __name__ == '__main__':
    unittest.main()
