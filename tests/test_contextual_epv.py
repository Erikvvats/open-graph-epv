import numpy as np
import pytest
from open_graph_epv.contextual_epv import classify_target,period_for_frame,TARGET,ContextualEnsemble


def shots():
    return [dict(period=1,frame=100,xg=.8,source='context'),
            dict(period=1,frame=250,xg=.2,source='context'),
            dict(period=2,frame=260,xg=.3,source='geometry')]


def test_current_shot_excluded_future_included_and_sign_preserved():
    for value in (.2,-.2):
        assert classify_target(shots(),100,1,500,value,'context','geometry')==(1,250)
    assert classify_target(shots(),250,1,500,0.,'context','geometry')==(0,-1)


def test_missing_context_does_not_become_no_shot():
    s=[dict(period=1,frame=101,xg=.08,source='geometry')]
    assert classify_target(s,100,1,500,-.08,'context','geometry')==(2,101)
    with pytest.raises(ValueError,match='differs'):classify_target(s,100,1,500,0.,'context','geometry')


def test_censoring_and_teacher_identity():
    with pytest.raises(ValueError,match='Incomplete'):classify_target(shots(),100,1,249,.2,'context','geometry')
    with pytest.raises(ValueError,match='Unknown teacher'):classify_target(shots(),100,1,500,.2,'wrong','geometry')
    with pytest.raises(ValueError,match='Nonzero'):classify_target([],100,1,500,.2,'context','geometry')


def test_period_lookup_rejects_gaps_and_overlaps():
    metadata={'match_periods':[dict(period=1,start_frame=0,end_frame=100),dict(period=2,start_frame=110,end_frame=220)]}
    assert period_for_frame(metadata,110)['period']==2
    with pytest.raises(ValueError):period_for_frame(metadata,105)
    metadata['match_periods'][1]['start_frame']=100
    with pytest.raises(ValueError):period_for_frame(metadata,100)


def test_never_skip_a_fallback_to_pick_a_later_context_shot():
    s=[dict(period=1,frame=101,xg=.08,source='geometry'),dict(period=1,frame=102,xg=.4,source='context')]
    assert classify_target(s,100,1,500,.08,'context','geometry')==(2,101)
    with pytest.raises(ValueError):classify_target(s,100,1,500,.4,'context','geometry')
