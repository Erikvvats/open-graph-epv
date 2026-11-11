import copy
import json

import numpy as np
import pytest

from open_graph_epv.xg import sb_shots, geometry, fit_logistic, probability, binary_metrics, label_skillcorner, GEOMETRY
from open_graph_epv.source import digest


def test_statsbomb_excludes_shootouts_and_provider_xg_is_not_a_feature():
    e = {'id': 'one', 'type': {'name': 'Shot'}, 'period': 1, 'location': [108., 40.],
         'shot': {'outcome': {'name': 'Goal'}, 'statsbomb_xg': .9, 'end_location': [120., 40., 1.]}}
    r = sb_shots([e], 1)[0]
    assert r['distance_x'] == pytest.approx(10.5)
    assert r['goal'] == 1
    changed = copy.deepcopy(e)
    changed['shot']['statsbomb_xg'] = 0
    changed['shot']['end_location'] = [120., 90., 8.]
    assert sb_shots([changed], 1)[0] == r
    changed['period'] = 5
    assert sb_shots([changed], 1) == []
    changed['period'] = 1
    changed['shot']['outcome']['name'] = 'Saved'
    before = sb_shots([changed], 1)[0]
    assert geometry([before['distance_x']], [before['lateral_y']]).tolist() == geometry([r['distance_x']], [r['lateral_y']]).tolist()


def test_geometry_reflection_goal_angle_and_rejects_bad_coordinates():
    np.testing.assert_allclose(geometry([10, 25], [3, 10]), geometry([10, 25], [-3, -10]))
    g = geometry([10, 30], [0, 0])
    assert g[0, 1] > g[1, 1] > 0
    with pytest.raises(ValueError):
        geometry([-1], [0])
    with pytest.raises(ValueError):
        geometry([np.nan], [0])


def test_out_of_pitch_shots_are_explicitly_audited():
    event = {'id': 'outside', 'type': {'name': 'Shot'}, 'period': 1,
             'location': [120.2, 1.], 'shot': {'outcome': {'name': 'Off T'}}}
    exclusions = []
    assert sb_shots([event], 1, exclusions) == []
    assert exclusions == [{'match_id': 1, 'event_id': 'outside',
                           'reason': 'shot location outside pitch', 'location': [120.2, 1.]}]


def test_logistic_learns_signal_and_auc_handles_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2))
    p = 1 / (1 + np.exp(-(-2 + 2 * x[:, 0])))
    y = rng.binomial(1, p)
    model = fit_logistic(x[:1500], y[:1500], 1)
    assert binary_metrics(y[1500:], probability(model, x[1500:]))['log_loss'] < binary_metrics(y[1500:], np.full(500, y[:1500].mean()))['log_loss']
    assert binary_metrics(np.array([0, 1, 1, 0]), np.full(4, .5))['auc'] == .5


def test_skillcorner_coordinates_already_attack_normalized(tmp_path):
    d = tmp_path / 'data/matches/1'
    d.mkdir(parents=True)
    (d / '1_match.json').write_text(json.dumps({'pitch_length': 104, 'pitch_width': 68}))
    # Identical normalized geometry for opposite attacking directions.
    (d / '1_dynamic_events.csv').write_text(
        'match_id,event_id,event_type,end_type,period,frame_end,team_id,x_end,y_end,attacking_side\n'
        '1,a,player_possession,shot,1,10,1,40,3,left_to_right\n'
        '1,b,player_possession,shot,1,20,2,40,-3,right_to_left\n')
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    model = {'features': GEOMETRY, 'mean': [0] * 8, 'scale': [1] * 8, 'weights': [-1, -.1] + [0] * 7}
    (model_dir / 'model.json').write_text(json.dumps(model))
    (model_dir / 'report.json').write_text(json.dumps({'model_sha256': digest(model_dir / 'model.json')}))
    result = label_skillcorner(tmp_path, [1], model_dir, tmp_path / 'labels.csv')
    assert result['n_labels'] == 2
    assert result['min_xg'] == result['max_xg']
