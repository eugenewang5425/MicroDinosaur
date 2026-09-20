"""Acceptance-critical reward gates, without training a policy."""
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
import torch
import mujoco
from mjlab_microduck.tasks import mdp

def test_running_needs_both_whole_feet_and_no_other_ground_contact():
    p=Path('D:/项目/miro_dinosaur/research/20260914_contact_motion/normal_contact_plant/nominal.mjb')
    m=mujoco.MjModel.from_binary_path('x',assets={'x':p.read_bytes()});d=mujoco.MjData(m)
    # Test geometry directly: one foot can have a high center and a low edge.
    mujoco.mj_forward(m,d)
    sim=NS(mj_model=m,data=NS(geom_xmat=torch.tensor(d.geom_xmat[None],dtype=torch.float32),
        geom_xpos=torch.tensor(d.geom_xpos[None],dtype=torch.float32)))
    robot=NS(data=NS(projected_gravity_b=torch.tensor([[0.,0.,-1.]]),root_link_lin_vel_b=torch.tensor([[.6,0.,0.]])))
    support=NS(data=NS(found=torch.tensor([[False]])))
    env=NS(device='cpu',num_envs=1,sim=sim,scene={'robot':robot,'robot_ground':support},
        command_manager=NS(get_command=lambda name:torch.tensor([[.6,0.,0.]])))
    reward=mdp.motion_clear_running(NS(params={}),env)
    env.scene['whole_ground_force']=NS(data=NS(force=torch.zeros(1,1,3)))
    lift=mdp.motion_running_sole_lift(NS(params={}),env)
    for g in reward.gids:sim.data.geom_xpos[:,g,2]=.10
    assert reward(env).item()>.99
    assert lift(env).item()>.99
    env.scene['whole_ground_force'].data.force[0,0,2]=10.
    assert lift(env).item()==0
    env.scene['whole_ground_force'].data.force.zero_()
    support.data.found[:]=True
    assert reward(env).item()==0 # Tail or shell contact invalidates flight.
    support.data.found[:]=False
    g=reward.gids[0];v=reward.vertices[0]
    # Place the mesh's lowest vertex at ground height while the center is up.
    row=sim.data.geom_xmat[:,g].reshape(1,3,3)[:,2,:]
    sim.data.geom_xpos[:,g,2]=-(row@v.T).amin(-1)
    assert reward(env).item()==0
    assert lift(env).item()==0

def test_getup_cannot_count_body_supported_or_inverted_standing():
    data=NS(projected_gravity_b=torch.tensor([[0.,0.,-1.]]),root_link_pos_w=torch.tensor([[0.,0.,.1135]]),
        root_link_lin_vel_b=torch.zeros(1,3),root_link_ang_vel_b=torch.zeros(1,3))
    class Scene(dict):pass
    scene=Scene(robot=NS(data=data),feet_ground_contact=NS(data=NS(found=torch.tensor([[True,True]]))),
        nonfoot_ground=NS(data=NS(found=torch.tensor([[False]]))))
    scene.env_origins=torch.zeros(1,3);env=NS(num_envs=1,scene=scene)
    assert mdp.motion_stable_stand(env).item()>.99
    scene['nonfoot_ground'].data.found[:]=True
    assert mdp.motion_stable_stand(env).item()==0
    scene['nonfoot_ground'].data.found[:]=False
    scene['feet_ground_contact'].data.found[0,1]=False
    assert mdp.motion_stable_stand(env).item()==0
    scene['feet_ground_contact'].data.found[:]=True
    data.projected_gravity_b[0,2]=1
    assert mdp.motion_stable_stand(env).item()==0


def test_recovery_orientation_feedback_exists_past_ninety_degrees():
    class Scene(dict):pass
    data=NS(projected_gravity_b=torch.tensor([[0.,0.,.5]]),root_link_pos_w=torch.tensor([[0.,0.,.065]]))
    scene=Scene(robot=NS(data=data));scene.env_origins=torch.zeros(1,3)
    env=NS(scene=scene);values=[]
    for angle in (120,110,100,90,80):
        data.projected_gravity_b[0,2]=-np.cos(np.deg2rad(angle))
        values.append(mdp.motion_recovery_posture_dense(env).item())
    assert 0<values[0]<values[1]<values[2]<values[3]<values[4]<1
