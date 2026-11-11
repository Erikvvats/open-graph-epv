"""Validation-only selection, frozen test audit and small-cluster diagnostics."""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .source import digest
from .training import load_epv, metrics, predict, validate_split


def value_metrics(y, p, prior):
    result = metrics(y, p, prior)
    result.update(mae=float(np.abs(y - p).mean()), predicted_mean=float(p.mean()), target_mean=float(y.mean()))
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
    bins = np.searchsorted(edges[1:-1], p, side='right')
    result['value_bins'] = [{'n': int((bins == b).sum()), 'prediction_mean': float(p[bins == b].mean()),
                             'target_mean': float(y[bins == b].mean())} for b in np.unique(bins)]
    return result


def paired_match_difference(y, candidate, reference, match_id):
    """Negative means candidate has lower MSE. Exact enumeration for <= 5 games.

    Samples within a match overlap, so never bootstrap individual snapshots.
    With three games the interval is descriptive, not strong population evidence.
    """
    games = np.unique(match_id)
    if not 2 <= len(games) <= 5:
        raise ValueError('Exact audit supports 2 to 5 independent matches')
    delta = (y - candidate) ** 2 - (y - reference) ** 2
    totals = np.array([[delta[match_id == g].sum(), (match_id == g).sum()] for g in games])
    samples = np.array(list(itertools.product(range(len(games)), repeat=len(games))))
    resampled = totals[samples].sum(1)
    distribution = resampled[:, 0] / resampled[:, 1]
    # Inverse empirical CDF, without interpolating between the 27 outcomes.
    quantile_indices = np.ceil(np.array([.025, .975]) * len(distribution)).astype(int) - 1
    interval = np.sort(distribution)[quantile_indices].tolist()
    loo = [(totals[:, 0].sum() - row[0]) / (totals[:, 1].sum() - row[1]) for row in totals]
    return {'mse_difference': float(delta.mean()), 'ci95': interval,
            'percentile_method': 'inverse empirical CDF of exact ordered resamples',
            'n_matches': len(games), 'ordered_resamples': len(samples),
            'leave_one_match_out_range': [float(min(loo)), float(max(loo))],
            'per_match': [{'match_id': int(g), 'n': int(n), 'mse_difference': float(s / n)}
                          for g, (s, n) in zip(games, totals)],
            'interpretation': 'Descriptive paired whole-match bootstrap; too few matches for strong population claims.'}


def read_run(root, name, expected_dataset, expected_split=None):
    path = Path(root) / name
    report = json.loads((path / 'training.json').read_text())
    if report['dataset_sha256'] != expected_dataset:
        raise ValueError(f'Dataset mismatch in {name}')
    if expected_split is not None and report['split'] != expected_split:
        raise ValueError(f'Split mismatch in {name}')
    if digest(path / 'model.pt') != report['model_sha256']:
        raise ValueError(f'Checkpoint checksum mismatch in {name}')
    return report


def select(data, runs, protocol_path, output):
    """Only validation values are evaluated; all expected families/seeds required."""
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(Path(protocol_path).read_text())
    checksum = digest(data / 'graphs.npz')
    if checksum != protocol['dataset_sha256']:
        raise ValueError('Dataset differs from registered protocol')
    torch.set_num_threads(2)
    split, prior, summaries, records = None, None, {}, []
    with np.load(data / 'graphs.npz', allow_pickle=False) as z:
        for family in protocol['families']:
            values = []
            individual = []
            for seed in protocol['seeds']:
                name = f'{family}_seed{seed}'
                report = read_run(runs, name, checksum, split)
                if report['seed'] != seed or report['model_kind'] != family:
                    raise ValueError(f'Model identity mismatch in {name}')
                if split is None:
                    split, prior = report['split'], report['training_mean']
                    validate_split(split, np.unique(z['match_id']))
                    use = np.isin(z['match_id'], split['val'])
                    x, mask = torch.from_numpy(z['x'][use]), torch.from_numpy(z['mask'][use])
                    y = z['y'][use]
                    if not np.isfinite(y).all():
                        raise ValueError('Invalid validation targets')
                if report['training_mean'] != prior:
                    raise ValueError('Training prior mismatch')
                p = predict(load_epv(Path(runs) / name), x, mask)
                values.append(p)
                individual.append({'seed': seed, **value_metrics(y, p, prior)})
                records.append({'name': name, 'family': family, 'seed': seed,
                                'checkpoint_sha256': report['model_sha256'],
                                'report_sha256': digest(Path(runs) / name / 'training.json')})
            summaries[family] = {'ensemble': value_metrics(y, np.mean(values, axis=0), prior), 'individual': individual}
    winner = min(protocol['families'], key=lambda k: summaries[k]['ensemble']['mse'])
    result = {'selected_family': winner, 'selection_rule': protocol['selection'], 'validation': summaries,
              'dataset_sha256': checksum, 'split': split, 'training_mean': prior, 'models': records,
              'protocol_sha256': digest(protocol_path), 'protocol': protocol,
              'inference_source_sha256': {name: digest(Path(__file__).with_name(name))
                                          for name in ('model.py', 'training.py', 'evaluation.py')},
              'test_evaluated': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    return result


def verify_selection(data, runs, selection):
    if digest(Path(data) / 'graphs.npz') != selection['dataset_sha256']:
        raise ValueError('Prepared dataset changed after selection')
    for name, checksum in selection['inference_source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Inference/evaluation source changed after selection')
    for record in selection['models']:
        path = Path(runs) / record['name']
        if digest(path / 'model.pt') != record['checkpoint_sha256'] or digest(path / 'training.json') != record['report_sha256']:
            raise ValueError(f"Frozen run changed: {record['name']}")


def audit(data, runs, selection_path, output):
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(Path(selection_path).read_text())
    verify_selection(data, runs, selection)  # Must pass before test arrays are used.
    torch.set_num_threads(2)
    with np.load(data / 'graphs.npz', allow_pickle=False) as z:
        use = np.isin(z['match_id'], selection['split']['test'])
        x, mask = torch.from_numpy(z['x'][use]), torch.from_numpy(z['mask'][use])
        y, matches, frames = z['y'][use], z['match_id'][use], z['frame'][use]
    if not np.isfinite(y).all():
        raise ValueError('Invalid test targets')
    prior = selection['training_mean']
    predictions, summaries = {}, {}
    for family in selection['protocol']['families']:
        records = [r for r in selection['models'] if r['family'] == family]
        values = [predict(load_epv(Path(runs) / r['name']), x, mask) for r in records]
        p = np.mean(values, axis=0)
        predictions[family] = p
        summaries[family] = {'ensemble': value_metrics(y, p, prior),
                             'individual': [{'seed': r['seed'], **value_metrics(y, v, prior)} for r, v in zip(records, values)],
                             'per_match': [{'match_id': int(m), **value_metrics(y[matches == m], p[matches == m], prior)}
                                           for m in np.unique(matches)]}
    reference = np.full(len(y), prior)
    comparisons = {'gnn_vs_mean': paired_match_difference(y, predictions['gnn'], reference, matches),
                   'gnn_vs_ball': paired_match_difference(y, predictions['gnn'], predictions['ball'], matches),
                   'gnn_vs_set': paired_match_difference(y, predictions['gnn'], predictions['set'], matches)}
    report = {'selected_family': selection['selected_family'], 'selection_sha256': digest(selection_path),
              'dataset_sha256': selection['dataset_sha256'], 'n_matches': len(np.unique(matches)), 'n_graphs': len(y),
              'models': summaries, 'training_mean_reference': value_metrics(y, reference, prior),
              'comparisons': comparisons, 'test_evaluated': True,
              'limitations': ['Only three independent test matches; overlapping frames are not independent replicates',
                              'Targets are transferred geometry-xG estimates, not observed goals or provider xG',
                              'No causal action or tactical recommendation validation',
                              'Test set is now consumed; further tuning requires new final evaluation data']}
    output.mkdir(parents=True)
    np.savez_compressed(output / 'predictions.npz', target=y, match_id=matches, frame=frames, **predictions)
    report['predictions_sha256'] = digest(output / 'predictions.npz')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
