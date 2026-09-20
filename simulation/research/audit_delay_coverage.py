"""Audit the controlled training diff and independent command symmetry semantics."""
import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import mjlab
import torch
import yaml
from tensordict import TensorDict
from mjlab.scripts.train import TrainConfig
from mjlab_microduck.tasks import mdp

ROOT = Path(__file__).parent / '20260913_delay_coverage'
RUNS = Path(r'D:\microduck_rl\logs\rsl_rl\microdinosaur_v07_calibration')
CONTROL = RUNS / '20260913_rewardfix_4096x200'
TREATMENT = RUNS / '20260913_delay0_4096x101'
# Reproduce the code actually used by this run even after the independent fix.
spec = importlib.util.spec_from_file_location('frozen_delay_run_symmetry',
        TREATMENT/'source_snapshot/5_symmetry_microdinosaur.py')
sym = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sym
spec.loader.exec_module(sym)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(a, b, path, differences):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            compare(a.get(key), b.get(key), path + '.' + key, differences)
    elif a != b:
        differences.append({'path': path, 'control': a, 'treatment': b})


def main():
    differences = []
    for name in ('agent.yaml', 'env.yaml'):
        compare(yaml.load((CONTROL/'params'/name).read_text(), Loader=yaml.BaseLoader),
                yaml.load((TREATMENT/'params'/name).read_text(), Loader=yaml.BaseLoader),
                name, differences)
    # Actuator tuples are serialized as lists; expand that diff to field level.
    expanded = []
    for row in differences:
        if row['path'] == 'env.yaml.scene.entities.robot.articulation.actuators':
            assert len(row['control']) == len(row['treatment']) == 1
            compare(row['control'][0], row['treatment'][0], row['path']+'.0', expanded)
        else:
            expanded.append(row)
    allowed = {
        'agent.yaml.run_name', 'agent.yaml.max_iterations',
        'env.yaml.scene.entities.robot.articulation.actuators.0.delay_min_lag',
        'env.yaml.observations.actor.terms.joint_pos.delay_min_lag',
        'env.yaml.observations.actor.terms.joint_vel.delay_min_lag',
    }
    assert {d['path'] for d in expanded} == allowed, expanded
    a, b = read(CONTROL/'run_provenance.json'), read(TREATMENT/'run_provenance.json')
    for key in ('source_checkpoint_sha256', 'model_xml_sha256', 'envs'):
        assert a[key] == b[key], key
    baseline_sources = {Path(r['path']).name: r['sha256'] for r in a['source_files']}
    treatment_sources = {Path(r['path']).name: r['sha256'] for r in b['source_files']}
    assert all(treatment_sources[n] == h for n, h in baseline_sources.items() if n != 'train_candidate.py')
    assert b['microdino_probe_environment'] == ''
    report = {'status': 'PASS', 'differences': expanded,
              'same_existing_task_sources': True, 'same_source_and_model': True,
              'control_checkpoint': str(CONTROL/'model_15600.pt'),
              'control_updates': 101, 'treatment_updates': 101,
              'note': 'Control uses checkpoint after 101 updates of a 200-update run; adaptive PPO has no horizon-dependent LR schedule.',
              'delay_contract_before': b['delay_contract_before'],
              'delay_contract_after': b['delay_contract_after']}
    (ROOT/'configuration_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    # A left=.4, right=-.2 arm target is perfectly matched by its joint state.
    # Under reflection, both the state and its target must become [.2, -.4].
    actor = torch.zeros(1, 81)
    actor[0, 6+17:6+19] = torch.tensor([.4, -.2])
    actor[0, 76:78] = torch.tensor([.4, -.2])
    obs = TensorDict({'actor': actor, 'critic': torch.zeros(1, 1)}, batch_size=[1])
    mirrored, _ = sym.microdinosaur_vel_symmetry(None, obs, None)
    ma = mirrored['actor'][1:]

    def reward(joints, command):
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=joints,
                    default_joint_pos=torch.zeros_like(joints)),
                    find_joints_by_actuator_names=lambda patterns: ([17,18], ['arm_l','arm_r']))
        env = SimpleNamespace(scene={'robot': asset}, device='cpu',
                    command_manager=SimpleNamespace(get_command=lambda name: command))
        return float(mdp.limb_pose_tracking(env, 'arm_pose', ('.*arm_l.*', '.*arm_r.*'))[0])

    original_reward = reward(actor[:,6:25], actor[:,76:78])
    mirrored_reward = reward(ma[:,6:25], ma[:,76:78])
    corrected_cmd = -actor[:,[77,76]]
    corrected_reward = reward(ma[:,6:25], corrected_cmd)
    cfg = TrainConfig.from_task('Mjlab-Velocity-Flat-MicroDinosaur')
    angular = cfg.env.rewards['track_angular_velocity']
    # Actual installed reward, with a mock entity exposing its documented state.
    def yaw_reward(actual_yaw, commanded_yaw):
        asset = SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.tensor([[0.,0.,actual_yaw]])))
        env = SimpleNamespace(scene={'robot':asset}, command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.,0.,commanded_yaw]])))
        return float(angular.func(env, **angular.params)[0])
    yaw_checks = {'positive_matched':yaw_reward(.3,.3), 'negative_matched':yaw_reward(-.3,-.3),
                  'opposite_sign':yaw_reward(-.3,.3)}
    assert yaw_checks['positive_matched'] == yaw_checks['negative_matched'] == 1.
    assert yaw_checks['opposite_sign'] < 1
    audit = {'arm_mapping_bug_reproduced': mirrored_reward < original_reward,
             'original_command':actor[0,76:78].tolist(), 'mirrored_command_actual':ma[0,76:78].tolist(),
             'mirrored_command_expected':corrected_cmd[0].tolist(),
             'original_reward':original_reward, 'mirrored_reward':mirrored_reward,
             'corrected_reward':corrected_reward, 'yaw_reward_sign_checks':yaw_checks,
             'angular_reward_weight':angular.weight, 'angular_reward_std':angular.params['std'],
             'symmetry_cfg':sym.SYMMETRY_CFG,
             'scope':'Arm-command inconsistency is real; it does not prove the cause of zero-arm-command walking yaw.'}
    assert audit['arm_mapping_bug_reproduced'] and corrected_reward == original_reward
    (ROOT/'command_symmetry_before.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'configuration':'PASS', **audit}, indent=2))


if __name__ == '__main__':
    main()
