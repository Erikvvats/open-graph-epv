import csv
import json

import numpy as np
import pytest
import torch

from open_graph_epv.data import FEATURES, frame_graph, future_target, load_xg
from open_graph_epv.model import GraphEPV, BallEPV
from open_graph_epv.training import validate_split


def example():
    metadata = {"home_team": {"id": 1}, "away_team": {"id": 2},
                "pitch_length": 100, "pitch_width": 60, "home_team_side": ['left_to_right', 'right_to_left'],
                "players": [{"id": i, "team_id": i, "player_role": {"acronym": "GK" if i == 2 else "CF"}}
                            for i in (1, 2)]}
    frame = {"frame": 100, "period": 1, "possession": {"group": "home team"},
             "ball_data": {"x": 3., "y": 2., "is_detected": True},
             "player_data": [{"player_id": i, "x": float(i), "y": 0., "is_detected": True} for i in (1, 2)]}
    return metadata, frame


def test_future_labels_sign_boundary_and_censoring():
    shots = [dict(frame=150, period=1, team_id=1, xg=.3), dict(frame=200, period=2, team_id=2, xg=.8)]
    assert future_target(shots, 100, 1, 1, 400) == .3
    assert future_target(shots, 100, 1, 2, 400) == -.3
    assert future_target(shots, 150, 1, 1, 400) == 0  # excludes current shot and next period
    assert future_target(shots, 0, 1, 1, 400) == .3  # inclusive horizon endpoint
    assert future_target(shots, 100, 1, 1, 200) is None


def test_graph_uses_backward_motion_and_current_metadata_only():
    meta, current = example()
    prior = json.loads(json.dumps(current))
    prior['frame'] = 99
    prior['player_data'][0]['x'] = .9
    x, mask, team = frame_graph(current, prior, meta)
    assert x[1, 2] == pytest.approx(.1)
    assert x[1, 9] == 1
    current['lead_to_goal'] = True
    meta['home_team_score'] = 99
    np.testing.assert_array_equal(frame_graph(current, prior, meta)[0], x)
    prior['period'] = 2
    assert frame_graph(current, prior, meta)[0][1, 9] == 0


def test_extrapolation_mask_and_period_direction():
    meta, frame = example()
    x = frame_graph(frame, None, meta)[0]
    frame['period'] = 2
    np.testing.assert_allclose(frame_graph(frame, None, meta)[0][:, :2], -x[:, :2])
    frame['ball_data']['is_detected'] = False
    assert frame_graph(frame, None, meta) is None
    assert frame_graph(frame, None, meta, include_extrapolated=True) is not None


def test_gnn_is_permutation_invariant_masks_padding_and_enforces_mirror_symmetry():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    meta, frame = example()
    nodes, mask, _ = frame_graph(frame, None, meta)
    x, m = torch.from_numpy(nodes)[None], torch.from_numpy(mask)[None]
    model = GraphEPV(hidden=12).eval()
    perm = torch.randperm(23)
    torch.testing.assert_close(model(x, m), model(x[:, perm], m[:, perm]))
    corrupt = x.clone()
    corrupt[~m] = float('nan')
    torch.testing.assert_close(model(x, m), model(corrupt, m))
    mirror = x.clone()
    mirror[..., [1, 3]] *= -1
    torch.testing.assert_close(model.symmetric(x, m), model.symmetric(mirror, m))
    loss = (model(x, m) - .2).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_missing_xg_and_misaligned_xg_fail(tmp_path):
    folder = tmp_path / 'data/matches/1'
    folder.mkdir(parents=True)
    (folder / '1_dynamic_events.csv').write_text('match_id,event_id,event_type,end_type,period,frame_end,team_id\n1,s1,player_possession,shot,1,100,1\n')
    path = tmp_path / 'xg.csv'
    header = 'match_id,event_id,period,frame,team_id,xg,xg_source\n'
    path.write_text(header + '1,s1,1,100,1,,source\n')
    with pytest.raises(ValueError, match='Missing xG'):
        load_xg(path, tmp_path, [1])
    path.write_text(header + '1,s1,1,101,1,0.2,source\n')
    with pytest.raises(ValueError, match='not aligned'):
        load_xg(path, tmp_path, [1])
    path.write_text(header + '1,s1,1,100,1,0.2,source\n')
    assert load_xg(path, tmp_path, [1])[1][0]['xg'] == .2
    path.write_text(header + '1,s1,1,100,1,0.2,source\n' * 2)
    with pytest.raises(ValueError, match='Duplicate'):
        load_xg(path, tmp_path, [1])


def test_match_split_rejects_leakage_and_missing_matches():
    validate_split({'train': [1], 'val': [2], 'test': [3]}, [1, 2, 3])
    with pytest.raises(ValueError, match='leakage'):
        validate_split({'train': [1], 'val': [1], 'test': [3]}, [1, 3])
    with pytest.raises(ValueError, match='exactly'):
        validate_split({'train': [1], 'val': [2], 'test': [3]}, [1, 2, 3, 4])


def test_ball_baseline_ignores_player_locations():
    meta, frame = example()
    nodes, mask, _ = frame_graph(frame, None, meta)
    x, m = torch.from_numpy(nodes)[None], torch.from_numpy(mask)[None]
    model = BallEPV().eval()
    moved = x.clone()
    moved[:, 1:3, :4] += 5
    torch.testing.assert_close(model.symmetric(x, m), model.symmetric(moved, m))
