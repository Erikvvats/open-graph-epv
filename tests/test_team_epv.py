import numpy as np
import pytest
import torch

from open_graph_epv.team_epv import relative_targets, to_home_away, team_metrics, TeamEnsemble
from open_graph_epv.team_epv_model import TeamGraphEPV, TeamBallEPV


def test_targets_are_assigned_to_actual_teams_and_recover_signed_target():
    signed = np.array([.4, -.2, .3, -.1, 0.], np.float32)
    home = np.array([True, True, False, False, False])
    values = to_home_away(relative_targets(signed), home)
    np.testing.assert_allclose(values, [[.4, 0], [0, .2], [0, .3], [.1, 0], [0, 0]])
    np.testing.assert_array_equal((values[:, 0]-values[:, 1])*np.where(home, 1, -1), signed)
    np.testing.assert_array_equal(to_home_away(values, home), relative_targets(signed))


def test_predictions_swap_channels_without_negation_or_zeroing():
    # Unlike clipping a signed prediction, both teams can have nonzero value.
    values = np.array([[.03, .02], [.02, .03]])
    np.testing.assert_allclose(to_home_away(values, [True, False]), [[.03, .02], [.03, .02]])
    with pytest.raises(ValueError):
        to_home_away(np.ones((3, 2)), [True])
    with pytest.raises(ValueError):
        relative_targets(np.array([np.nan]))


@pytest.mark.parametrize('model', [TeamGraphEPV(hidden=8, layers=1), TeamGraphEPV(hidden=8, layers=1, edges_enabled=False), TeamBallEPV(hidden=8)])
def test_two_outputs_are_bounded_masked_and_mirror_invariant(model):
    torch.manual_seed(7)
    x = torch.randn(2, 5, 10)
    x[:, :, 4:10] = 0
    x[:, 0, 6] = 1
    x[:, 1:3, 4] = 1
    x[:, 3:, 5] = 1
    mask = torch.ones(2, 5, dtype=torch.bool);mask[:, -1] = False
    model.eval()
    with torch.no_grad():
        result = model.symmetric(x, mask)
        assert result.shape == (2, 2)
        assert (result > 0).all() and (result.sum(-1) <= 1).all()
        reflected = x.clone();reflected[..., 1] *= -1;reflected[..., 3] *= -1
        torch.testing.assert_close(model.symmetric(reflected, mask), result)
        order = torch.tensor([3, 0, 4, 1, 2])
        torch.testing.assert_close(model.symmetric(x[:, order], mask[:, order]), result)
        x[:, -1] = float('nan')
        torch.testing.assert_close(model.symmetric(x, mask), result)


def test_component_loss_detects_equal_net_but_wrong_team_values():
    y = np.array([[.04, .03], [.02, .01]])
    wrong = y + .1
    metrics = team_metrics(y, wrong)
    assert metrics['component_rmse'] == pytest.approx(.1)
    assert metrics['home_net_rmse'] == pytest.approx(0., abs=1e-12)


def test_public_api_preserves_team_identity_across_possession_and_period_changes():
    class ConstantPair:
        def symmetric(self, nodes, mask):
            return torch.tensor([[.03, .02]])
    ensemble = TeamEnsemble([ConstantPair()]*3, 'gnn')
    metadata = {'home_team': {'id': 10}, 'away_team': {'id': 20},
        'pitch_length': 100, 'pitch_width': 68, 'home_team_side': ['left_to_right', 'right_to_left'],
        'players': [{'id': 1, 'team_id': 10, 'player_role': {'acronym': 'CF'}},
                    {'id': 2, 'team_id': 20, 'player_role': {'acronym': 'GK'}}]}
    frame = {'frame': 100, 'period': 1, 'possession': {'group': 'home team'},
        'ball_data': {'x': 1., 'y': 0., 'is_detected': True},
        'player_data': [{'player_id': i, 'x': float(i), 'y': 0., 'is_detected': True} for i in (1, 2)]}
    for period in (1, 2):
        frame['period'] = period
        for group, expected in [('home team', [.03, .02]), ('away team', [.02, .03])]:
            frame['possession']['group'] = group
            result = ensemble.score_frame(frame, None, metadata)
            assert result['home_team_id'] == 10 and result['away_team_id'] == 20
            assert result['home_team_epv'] == pytest.approx(expected[0])
            assert result['away_team_epv'] == pytest.approx(expected[1])
            assert result['net_home_epv'] == pytest.approx(expected[0]-expected[1])
            assert set(result['seed_std']) == {'home_team_epv', 'away_team_epv'}


def test_team_api_rejects_unobserved_ball():
    ensemble = TeamEnsemble([], 'gnn')
    frame = {'period': 1, 'possession': {'group': 'home team'}, 'ball_data': {'x': 0., 'y': 0., 'is_detected': False}}
    metadata = {'home_team': {'id': 10}, 'home_team_side': ['left_to_right', 'right_to_left']}
    with pytest.raises(ValueError, match='Snapshot lacks'):
        ensemble.score_frame(frame, None, metadata)
