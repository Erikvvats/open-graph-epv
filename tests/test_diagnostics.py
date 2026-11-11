import numpy as np
import pytest

from open_graph_epv.diagnostics import position_variants, home_perspective, error_groups, value_bins, shot_error_groups


def test_perturbation_moves_exactly_one_observed_player_one_physical_metre():
    x = np.zeros((4, 10), np.float32)
    x[0, 6] = 1
    x[1, 4] = 1
    x[1, 2:4] = [.3, -.2]
    x[2, 5] = 1
    x[3, 5] = 1
    mask = np.array([True, True, True, False])
    variants, slots = position_variants(x, mask, [50, 35])
    assert len(variants) == 8
    for variant, slot in zip(variants, slots):
        assert slot in (1, 2)
        difference = variant - x
        assert np.count_nonzero(difference) == 1
        assert np.linalg.norm(difference[slot, :2] * [50, 35]) == pytest.approx(1)
        assert np.array_equal(variant[:, 2:], x[:, 2:])
    assert np.array_equal(x[1, :2], [0, 0])


def test_perturbation_rejects_out_of_pitch_moves_without_clipping():
    x = np.zeros((2, 10), np.float32)
    x[0, 6] = 1
    x[1, 0] = 1
    variants, _ = position_variants(x, np.ones(2, bool), [50, 35])
    assert len(variants) == 3
    assert (np.abs(variants[:, :, :2]) <= 1).all()
    x[1, 0] = 1.01
    assert len(position_variants(x, np.ones(2, bool), [50, 35])[0]) == 0
    with pytest.raises(ValueError):
        position_variants(x, np.ones(2, bool), [0, 35])


def test_home_perspective_preserves_roles_and_is_involution():
    x = np.arange(30, dtype=np.float32).reshape(3, 10)
    converted = home_perspective(x, False)
    assert np.array_equal(converted[:, :4], -x[:, :4])
    assert np.array_equal(converted[:, 4:], x[:, 4:])
    assert np.array_equal(home_perspective(converted, False), x)
    assert np.array_equal(home_perspective(x, True), x)


def test_error_shares_reconcile_and_empty_group_is_explicit():
    y = np.array([0., 0., .4]);p = np.array([.1, .2, .1])
    groups = error_groups(y, p)
    assert sum(g['n'] for g in groups.values()) == 3
    assert sum(g['mse_share'] for g in groups.values()) == pytest.approx(1)
    assert groups['opponent_shot']['rmse'] is None
    assert groups['own_shot']['mse_share'] == pytest.approx(.09 / .14)


def test_constant_prediction_bins_do_not_split_ties():
    bins = value_bins(np.array([0., .2, -.1, 0.]), np.zeros(4))
    assert len(bins) == 1 and bins[0]['n'] == 4
    assert bins[0]['target'] == pytest.approx(.025)


def test_shot_error_grouping_counts_one_shot_across_opposite_possessions():
    y = np.array([.4, -.4, 0., .2])
    p = np.zeros(4)
    matches = np.array([1, 1, 1, 2])
    frames = np.array([100, 100, -1, 100])
    kinds = np.array([1, 1, 0, 2])
    r = shot_error_groups(y, p, matches, frames, kinds)
    assert r['n_distinct_future_shots'] == 2
    assert r['ranked_shots'][0]['n_windows'] == 2
    assert r['ranked_shots'][0]['mse_share'] == pytest.approx(.32 / .36)
    kinds[1] = 2
    with pytest.raises(ValueError, match='Inconsistent'):
        shot_error_groups(y, p, matches, frames, kinds)
