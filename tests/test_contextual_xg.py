import copy
import numpy as np
import pytest
from open_graph_epv.xg_context_features import (FEATURES,spatial_features,statsbomb_features,
    skillcorner_features,MissingContext)
from open_graph_epv.contextual_xg import paired_difference


def sb_event():
    return {'location':[100.,40.],'shot':{'freeze_frame':[
        {'player':{'id':1},'position':{'name':'Goalkeeper'},'teammate':False,'location':[118.,40.]},
        {'player':{'id':2},'position':{'name':'Center Back'},'teammate':False,'location':[105.,42.]},
        {'player':{'id':3},'position':{'name':'Center Back'},'teammate':False,'location':[101.,38.]},
        {'player':{'id':4},'position':{'name':'Center Forward'},'teammate':True,'location':[102.,45.]}]}}


def sc_scene():
    event=sb_event()
    def xy(p):return [(p[0]-60)*105/120,(p[1]-40)*68/80]
    s=xy(event['location']);opponents=event['shot']['freeze_frame'][:3]
    shot={'frame_end':'101','period':'1','team_id':'1','player_id':'9','x_end':str(s[0]),'y_end':str(s[1]),'is_player_possession_end_matched':'True'}
    frame={'frame':100,'period':1,'ball_data':dict(x=s[0],y=s[1],is_detected=True),
        'player_data':[dict(player_id=9,x=s[0],y=s[1],is_detected=True)]+[
            dict(player_id=p['player']['id'],x=xy(p['location'])[0],y=xy(p['location'])[1],is_detected=True) for p in opponents]}
    metadata={'home_team':{'id':1},'away_team':{'id':2},'home_team_side':['left_to_right','right_to_left'],
        'pitch_length':105,'pitch_width':68,'players':[{'id':9,'team_id':1,'player_role':{'acronym':'CF'}}]+[
            {'id':p['player']['id'],'team_id':2,'player_role':{'acronym':'GK' if i==0 else 'CB'}} for i,p in enumerate(opponents)]}
    return shot,frame,metadata


def test_identical_physical_scenes_across_providers():
    expected=statsbomb_features(sb_event())
    np.testing.assert_allclose(skillcorner_features(*sc_scene()),expected)
    shot,frame,metadata=sc_scene();metadata['home_team_side'][0]='right_to_left'
    for p in [*frame['player_data'],frame['ball_data']]:p['x']*=-1;p['y']*=-1
    # Event endpoint remains attack-normalized and must NOT be rotated again.
    np.testing.assert_allclose(skillcorner_features(shot,frame,metadata),expected)


def test_symmetry_permutation_and_context_changes():
    event=sb_event();original=statsbomb_features(event)
    mirror=copy.deepcopy(event);mirror['location'][1]=80-mirror['location'][1]
    for p in mirror['shot']['freeze_frame']:p['location'][1]=80-p['location'][1]
    mirror['shot']['freeze_frame'].reverse()
    np.testing.assert_allclose(statsbomb_features(mirror),original)
    event['shot']['freeze_frame'][0]['location'][1]=60
    changed=statsbomb_features(event)
    np.testing.assert_allclose(changed[:8],original[:8])
    assert original[FEATURES.index('keeper_in_goal_triangle')]==1
    assert changed[FEATURES.index('keeper_in_goal_triangle')]==0


def test_outcomes_and_provider_predictions_do_not_enter_features():
    event=sb_event();original=statsbomb_features(event)
    event['shot'].update(outcome={'name':'Goal'},statsbomb_xg=.999,end_location=[120,40,2],body_part={'name':'Head'})
    np.testing.assert_array_equal(statsbomb_features(event),original)
    shot,frame,metadata=sc_scene();original=skillcorner_features(shot,frame,metadata)
    shot.update(lead_to_goal='True',game_interruption_after='goal_for',possession_epv_for_end='999')
    np.testing.assert_array_equal(skillcorner_features(shot,frame,metadata),original)


def test_bad_freeze_frame_abstains():
    event=sb_event();event['shot']['freeze_frame']=[]
    with pytest.raises(MissingContext,match='missing_freeze'):statsbomb_features(event)
    event=sb_event();event['shot']['freeze_frame'][0]['position']['name']='Center Back'
    with pytest.raises(MissingContext,match='keeper'):statsbomb_features(event)
    event=sb_event();event['shot']['freeze_frame'].append(copy.deepcopy(event['shot']['freeze_frame'][0]))
    with pytest.raises(MissingContext,match='duplicate'):statsbomb_features(event)


@pytest.mark.parametrize('failure',['unmatched','postshot','keeper','ball','alignment'])
def test_unreliable_tracking_context_abstains(failure):
    shot,frame,metadata=sc_scene()
    if failure=='unmatched':shot['is_player_possession_end_matched']='False'
    elif failure=='postshot':frame['frame']=101
    elif failure=='keeper':frame['player_data'][1]['is_detected']=False
    elif failure=='ball':frame['ball_data']=None
    else:shot['x_end']='0'
    with pytest.raises(MissingContext):skillcorner_features(shot,frame,metadata)


def test_paired_bootstrap_identity():
    y=np.array([0,1,0,1]);p=np.array([.1,.2,.3,.4]);groups=np.array([1,1,2,2])
    result=paired_difference(y,p,p,groups)
    assert result['log_loss_difference']==0
    assert result['descriptive_match_bootstrap_95_interval']==[0,0]
