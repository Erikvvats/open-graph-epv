import json

import numpy as np
import pytest
import torch

from open_graph_epv.evaluation import audit, paired_match_difference, select
from open_graph_epv.model import GraphEPV, BallEPV
from open_graph_epv.source import digest


def fixture(tmp_path, invalid_test=False):
    data = tmp_path / 'data'
    data.mkdir()
    x = np.zeros((15, 23, 10), dtype=np.float32)
    x[:, 0, 6] = 1
    x[:, 1, 4] = 1
    x[:, 2, 5] = 1
    mask = np.zeros((15, 23), dtype=bool)
    mask[:, :3] = True
    y = np.tile([0., .1, -.2], 5).astype(np.float32)
    if invalid_test:
        y[6:] = np.nan
    np.savez_compressed(data / 'graphs.npz', x=x, mask=mask, y=y, match_id=np.repeat([1, 2, 3, 4, 5], 3), frame=np.arange(15))
    checksum = digest(data / 'graphs.npz')
    split = {'train': [1], 'val': [2], 'test': [3, 4, 5]}
    runs = tmp_path / 'runs'
    for kind in ('gnn', 'ball', 'set'):
        torch.manual_seed(0)
        model = BallEPV() if kind == 'ball' else GraphEPV(edges_enabled=kind == 'gnn')
        path = runs / f'{kind}_seed0'
        path.mkdir(parents=True)
        torch.save({'config': model.config, 'model_kind': kind, 'state_dict': model.state_dict()}, path / 'model.pt')
        (path / 'training.json').write_text(json.dumps({'seed': 0, 'model_kind': kind, 'split': split,
                                                       'training_mean': float(y[:3].mean()), 'dataset_sha256': checksum,
                                                       'model_sha256': digest(path / 'model.pt')}))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'families': ['gnn', 'ball', 'set'], 'seeds': [0],
                                    'selection': 'minimum ensemble validation MSE', 'dataset_sha256': checksum}))
    return data, runs, protocol


def test_selection_never_scores_test_targets_and_rejects_changed_checkpoint(tmp_path):
    data, runs, protocol = fixture(tmp_path, invalid_test=True)
    selection = tmp_path / 'selection.json'
    r = select(data, runs, protocol, selection)
    assert r['test_evaluated'] is False
    assert all(np.isfinite(s['ensemble']['mse']) for s in r['validation'].values())
    with (runs / 'gnn_seed0/model.pt').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Frozen run changed'):
        audit(data, runs, selection, tmp_path / 'audit')
    assert not (tmp_path / 'audit').exists()


def test_complete_frozen_audit_and_no_overwrite(tmp_path):
    data, runs, protocol = fixture(tmp_path)
    selection = tmp_path / 'selection.json'
    select(data, runs, protocol, selection)
    r = audit(data, runs, selection, tmp_path / 'audit')
    assert r['n_matches'] == 3 and r['n_graphs'] == 9
    assert len(r['models']['gnn']['per_match']) == 3
    assert r['comparisons']['gnn_vs_ball']['ordered_resamples'] == 27
    with pytest.raises(FileExistsError):
        audit(data, runs, selection, tmp_path / 'audit')


def test_cluster_bootstrap_retains_match_sizes():
    y = np.zeros(6)
    p = np.array([1., 2., 2., 3., 3., 3.])
    games = np.array([1, 2, 2, 3, 3, 3])
    r = paired_match_difference(y, p, np.zeros(6), games)
    assert r['mse_difference'] == pytest.approx(6.)
    assert r['ci95'] == [1., 9.]
    assert [g['n'] for g in r['per_match']] == [1, 2, 3]
    zero = paired_match_difference(y, p, p, games)
    assert zero['ci95'] == [0., 0.]


def test_no_edges_ablation_never_calls_message_networks():
    model = GraphEPV(hidden=12, edges_enabled=False)
    for message in model.messages:
        message.register_forward_hook(lambda *args: pytest.fail('No-edge model computed messages'))
    x = torch.zeros(1, 23, 10)
    mask = torch.zeros(1, 23, dtype=torch.bool)
    mask[:, :3] = True
    x[:, 0, 6], x[:, 1, 4], x[:, 2, 5] = 1, 1, 1
    p = model(x, mask)
    perm = torch.randperm(23)
    torch.testing.assert_close(p, model(x[:, perm], mask[:, perm]))
    p.sum().backward()
    assert model.encoder[0].weight.grad is not None
    assert all(param.grad is None for message in model.messages for param in message.parameters())


def test_portable_bundle_validates_xg_source_and_scores_real_frame_schema(tmp_path):
    from open_graph_epv.bundle import export_bundle, load_bundle
    from open_graph_epv.data import FEATURES
    data, runs, protocol = fixture(tmp_path)
    selection = tmp_path / 'selection.json'
    select(data, runs, protocol, selection)
    audit(data, runs, selection, tmp_path / 'audit')
    xg = tmp_path / 'xg'
    xg.mkdir()
    (xg / 'model.json').write_text('{}')
    (xg / 'report.json').write_text(json.dumps({'model_sha256': digest(xg / 'model.json')}))
    metadata = {'features': FEATURES, 'horizon_seconds': 15, 'include_extrapolated': False,
                'xg_sources': ['wrong model']}
    (data / 'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='xG source'):
        export_bundle(data, runs, selection, tmp_path / 'audit/report.json', xg, tmp_path / 'bad_bundle')
    metadata['xg_sources'] = [f"statsbomb_geometry_v1:{digest(xg / 'model.json')}"]
    (data / 'manifest.json').write_text(json.dumps(metadata))
    export_bundle(data, runs, selection, tmp_path / 'audit/report.json', xg, tmp_path / 'bundle')
    model = load_bundle(tmp_path / 'bundle', family='gnn')
    meta = {'home_team': {'id': 1}, 'away_team': {'id': 2}, 'pitch_length': 105, 'pitch_width': 68,
            'home_team_side': ['left_to_right', 'right_to_left'],
            'players': [{'id': i, 'team_id': i, 'player_role': {'acronym': 'CF'}} for i in (1, 2)]}
    frame = {'frame': 20, 'period': 1, 'possession': {'group': 'away team'},
             'ball_data': {'x': 0., 'y': 0., 'is_detected': True},
             'player_data': [{'player_id': i, 'x': float(i), 'y': 0., 'is_detected': True} for i in (1, 2)]}
    score = model.score_frame(frame, None, meta)
    assert np.isfinite(score['value']) and score['possession_team_id'] == 2
    assert score['home_signed_value'] == -score['value']
    (tmp_path / 'bundle/xg_model.json').write_text('modified')
    with pytest.raises(ValueError, match='checksum'):
        load_bundle(tmp_path / 'bundle')
