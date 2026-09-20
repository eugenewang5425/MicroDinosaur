from types import SimpleNamespace as NS
import torch
from mjlab_microduck.tasks import mdp


def test_squat_reward_requires_height_pose_and_bilateral_support():
    names=['left_knee','right_knee']
    q=torch.zeros(5,2)
    q[0]=torch.tensor([.5,-.5])
    q[2:]=q[0]
    robot=NS(find_joints=lambda *a,**k:([0,1],names),data=NS(joint_pos=q,encoder_bias=torch.zeros_like(q),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*3+[[0.,1.,0.],[0.,0.,-1.]])))
    body=NS(command=torch.tensor([[0.,0.,-.025,0.,0.,0.]]*5))
    action=NS(bank={'joint_pos':torch.zeros(1,2),'root_qpos':torch.tensor([[0.,0.,.114,1.,0.,0.,0.]])},
        bank_index=torch.zeros(5,dtype=torch.long))
    scene={'robot':robot,'body_clearance':NS(data=NS(heights=torch.tensor([[.089],[.114],[.089],[.089],[.089]]))),
        'feet_ground_contact':NS(data=NS(found=torch.tensor([[1,1],[1,1],[1,0],[1,1],[1,1]]))),
        'nonfoot_ground':NS(data=NS(found=torch.tensor([[0],[0],[0],[0],[1]])))}
    env=NS(scene=scene,num_envs=5,device='cpu',action_manager=NS(get_term=lambda n:action),
        command_manager=NS(get_command=lambda n:body.command))
    table=[[i*.1,-i*.1] for i in range(6)]
    cfg=NS(params={'reference_names':names,'reference_table':table})
    reward=mdp.independent_squat_tracking(cfg,env)(env,names,table)
    assert reward[0]>.999
    assert reward[1]<.001  # Standing at HOME must not satisfy a squat request.
    assert torch.equal(reward[2:],torch.zeros(3))  # One foot, fallen, body support.


def test_squat_cycle_is_bounded_returns_and_has_no_reset_history():
    command=mdp.IndependentSquatCommand(mdp.IndependentSquatCommandCfg(width=6,resampling_time_range=(100.,100.)),
        NS(step_dt=.02,num_envs=2,device='cpu'))
    command.depth=torch.tensor([.025,0.])
    command.return_at=torch.tensor([7.,7.])
    command._command=torch.zeros(2,6)
    values=[]
    for _ in range(650):
        command._update_command();values.append(command.command.clone())
    a=torch.stack(values)
    assert a[:,1].abs().max()==0  # Explicit idle cannot inherit another row.
    assert a[:,0,2].min()>=-.025001 and a[:,0,2].min()<-.02499
    assert a[-1].abs().max()==0
    assert torch.diff(a[:,0,2]).abs().max()<.00026
    command.elapsed[:]=torch.tensor([5.,6.])
    command._command[:,2]=torch.tensor([-.025,-.015])
    command.reset(torch.tensor([0]));command.compute(0.)
    assert command.elapsed.tolist()==[0.,6.]
    assert command.command[0].abs().max()==0
    assert command.command[1,2]==-.015
