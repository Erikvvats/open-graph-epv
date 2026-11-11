import numpy as np
import pytest

from open_graph_epv.provider_comparison import provider_values, candidate_events, affine_fit, affine_predict
from open_graph_epv.data import future_target


def event(**changes):
    return dict(event_type='player_possession',event_id='1',period='1',frame_start='100',
                is_player_possession_start_matched='True',possession_epv_for_start='0.2',
                possession_epv_against_start='0.05',**changes)


def test_provider_state_is_not_action_delta_or_endpoint():
    row=event(possession_epv_total='99',possession_epv_for_end='1',possession_epv_against_end='0')
    pair=provider_values(row)
    np.testing.assert_allclose(pair,[.2,.05])
    assert pair[0]-pair[1]==pytest.approx(.15)


@pytest.mark.parametrize('value',['','nan','inf','-0.1','1.1'])
def test_missing_invalid_provider_value_abstains(value):
    row=event();row['possession_epv_for_start']=value
    assert provider_values(row) is None


def test_zero_is_not_missing():
    row=event();row['possession_epv_for_start']='0';row['possession_epv_against_start']='0'
    np.testing.assert_array_equal(provider_values(row),[0,0])


def test_exact_start_selection_and_duplicate_abstention():
    first=event();second={**first,'event_id':'2'}
    third={**first,'event_id':'3','frame_start':'101'}
    unmatched={**first,'event_id':'4','frame_start':'102','is_player_possession_start_matched':'False'}
    other={**first,'event_type':'passing_option','frame_start':'103'}
    selected,counts=candidate_events([first,second,third,unmatched,other])
    assert set(selected)=={(1,101)}
    assert counts['ambiguous_same_frame_events']==2 and counts['unmatched_event_start']==1
    assert counts['possession_events']==4


def test_affine_transfer_uses_both_components_and_fixed_coefficients():
    x=np.array([[0,0],[1,0],[0,1],[.1,.3],[.7,.2]])
    y=.02+.4*x[:,0]-.15*x[:,1]
    fitted=affine_fit(x,y)
    assert fitted['rank']==3
    np.testing.assert_allclose(affine_predict(fitted,x),y)
    test=np.array([[.5,.4],[.8,.2]])
    np.testing.assert_allclose(affine_predict(fitted,test),.02+.4*test[:,0]-.15*test[:,1])
    assert fitted['n_fit']==5


def test_target_excludes_current_shot_and_respects_sign_and_full_horizon():
    shots=[dict(period=1,frame=100,team_id=1,xg=.9),dict(period=1,frame=250,team_id=2,xg=.2)]
    assert future_target(shots,100,1,1,500,150)==-.2
    assert future_target(shots,100,1,2,500,150)==.2
    assert future_target(shots,100,1,1,249,150) is None
    assert future_target(shots,250,1,1,500,150)==0
