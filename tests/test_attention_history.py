import copy
import numpy as np
import pytest
import torch
from open_graph_epv.attention_model import AttentionEPV
from open_graph_epv.history_data import aligned_history
from open_graph_epv.model import GraphEPV


def sample():
    torch.manual_seed(19)
    x=torch.randn(2,4,23,10)*.1;x[...,4:7]=0
    x[:,:,0,6]=1;x[:,:,1:12,4]=1;x[:,:,12:,5]=1
    mask=torch.ones(2,4,23,dtype=torch.bool);mask[:,:,-2:]=False;mask[:,0]=False
    return x,mask


@pytest.mark.parametrize('mode',['snapshot','history','repeat'])
def test_attention_permutation_mirror_masks_and_gradients(mode):
    x,mask=sample();model=AttentionEPV(mode=mode)
    output=model.symmetric(x,mask)
    order=torch.randperm(23)
    torch.testing.assert_close(model.symmetric(x[:,:,order],mask[:,:,order]),output,atol=1e-6,rtol=1e-5)
    mirror=x.clone();mirror[...,1]*=-1;mirror[...,3]*=-1
    torch.testing.assert_close(model.symmetric(mirror,mask),output)
    changed=x.clone();changed[~mask]=float('nan')
    torch.testing.assert_close(model.symmetric(changed,mask),output)
    output.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_history_ablation_and_parameter_budget():
    x,mask=sample();changed=x.clone();changed[:,:-1,:,:4]+=1
    counts=[]
    for mode in ('snapshot','repeat','history'):
        model=AttentionEPV(mode=mode);before=model(x,mask);after=model(changed,mask)
        if mode=='history':assert not torch.allclose(before,after,atol=1e-7)
        else:torch.testing.assert_close(before,after)
        counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts))==1
    gnn=sum(p.numel() for p in GraphEPV().parameters())
    assert abs(counts[0]/gnn-1)<.02


def scene():
    meta={'home_team':{'id':1},'away_team':{'id':2},'home_team_side':['left_to_right','right_to_left'],
        'pitch_length':100.,'pitch_width':60.,'players':[
            {'id':11,'team_id':1,'player_role':{'acronym':'CF'}},
            {'id':22,'team_id':2,'player_role':{'acronym':'GK'}}]}
    def frame(t,team='home team',reverse=False):
        players=[{'player_id':11,'x':t/10,'y':2.,'is_detected':True},
                 {'player_id':22,'x':40.,'y':0.,'is_detected':True}]
        if reverse:players=players[::-1]
        return {'frame':t,'period':1,'possession':{'group':team},'player_data':players,
            'ball_data':{'x':t/10,'y':0.,'is_detected':True}}
    buffer={t:frame(t,'away team' if t<100 else 'home team',reverse=t<100) for t in range(79,101)}
    return meta,buffer


def test_history_tracks_ids_not_row_order_and_uses_current_team():
    meta,buffer=scene();x,mask=aligned_history(buffer[100],buffer,meta)
    assert mask.all(1).sum()==0  # Padding remains masked.
    np.testing.assert_allclose(x[:,1,0],[.16,.18,.19,.20])
    np.testing.assert_allclose(x[:,2,0],.8)
    np.testing.assert_array_equal(x[:,1,4],1)  # Historical possession is the other team.
    np.testing.assert_array_equal(x[:,2,5],1)
    np.testing.assert_allclose(x[:,1,2],.1,atol=1e-6)  # 1 m/s, backward differences.


def test_history_censors_period_boundary_missing_frames_and_future():
    meta,buffer=scene();buffer[80]['period']=2
    del buffer[90]
    x,mask=aligned_history(buffer[100],buffer,meta)
    assert not mask[:2].any()
    changed=copy.deepcopy(buffer);changed[101]={**buffer[100],'frame':101}
    np.testing.assert_array_equal(aligned_history(buffer[100],changed,meta)[0],x)
    with pytest.raises(ValueError):aligned_history(buffer[100],buffer,meta,offsets=(-1,1,0))
