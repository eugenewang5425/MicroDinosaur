import unittest
from types import SimpleNamespace

import mjlab
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_microduck.s288_protocol import KP_STEP,KD_STEP,OUTPUT_ENCODER_STEP,wire_time_seconds
from mjlab_microduck.s288_profile import s288_joint_position,quantize_firmware_gains


class S288ProfileTests(unittest.TestCase):
    def test_position_quantizes_absolute_encoder_before_home_subtraction(self):
        actual = torch.tensor([[.0713,-.2438,.55]])
        home = torch.tensor([[.0873,-.21,.50]])
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=actual,default_joint_pos=home))
        env = SimpleNamespace(scene={'robot':asset})
        selector = SceneEntityCfg('robot',joint_ids=[2,0])
        measured = s288_joint_position(env,asset_cfg=selector)
        error = measured-(actual-home)[:,[2,0]]
        self.assertLessEqual(float(error.abs().max()),OUTPUT_ENCODER_STEP/2+1e-7)
        counts = (measured+home[:,[2,0]])/OUTPUT_ENCODER_STEP
        torch.testing.assert_close(counts,counts.round(),atol=1e-4,rtol=0)

    def test_gain_rounding_is_local_idempotent_and_preserves_pd_signs(self):
        gain = torch.zeros(3,5,10); bias = torch.zeros_like(gain)
        gain[:,:,0]=7; bias[:,:,1]=-7; bias[:,:,2]=-.8
        env = SimpleNamespace(num_envs=3,device='cpu',
            scene={'robot':SimpleNamespace(indexing=SimpleNamespace(ctrl_ids=torch.tensor([1,3])))},
            sim=SimpleNamespace(model=SimpleNamespace(actuator_gainprm=gain,actuator_biasprm=bias)))
        original = gain.clone()
        quantize_firmware_gains(env,torch.tensor([1]))
        torch.testing.assert_close(gain[[0,2]],original[[0,2]])
        torch.testing.assert_close(gain[1,[0,2,4]],original[1,[0,2,4]])
        self.assertLessEqual(abs(float(gain[1,1,0])-7),KP_STEP/2)
        self.assertLessEqual(abs(float(bias[1,1,2])+.8),KD_STEP/2)
        self.assertEqual(float(bias[1,1,1]),-float(gain[1,1,0]))
        saved = gain.clone(),bias.clone()
        quantize_firmware_gains(env,torch.tensor([1]))
        torch.testing.assert_close(gain,saved[0]);torch.testing.assert_close(bias,saved[1])

    def test_dual_bus_wire_floor(self):
        self.assertAlmostEqual(wire_time_seconds(10),.0007666666666666667)
        self.assertAlmostEqual(wire_time_seconds(9),.00069)


if __name__=='__main__':unittest.main()
