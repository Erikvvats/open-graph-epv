"""Public StatsBomb goal-labelled xG and explicit transfer to SkillCorner shots.

Only shot geometry is shared across providers. No StatsBomb xG, shot outcome,
freeze-frame, post-shot or SkillCorner provider model columns enter features.
"""
import csv
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from .source import digest, match_path, read_events, shot_events, REVISION

SB_REVISION = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
SB_BASE = f"https://raw.githubusercontent.com/hudl/open-data/{SB_REVISION}/"
SEASONS = ((2, 27), (9, 281), (43, 3), (43, 106), (55, 43), (55, 282))
GEOMETRY = ["distance", "angle", "log_distance", "lateral", "distance_squared",
            "angle_squared", "distance_angle", "lateral_squared"]


def cached_file(root, relative):
    """Per-file receipts permit parallel downloads without a shared write lock."""
    path = Path(root) / relative
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if path.exists():
        if receipt.exists():
            r = json.loads(receipt.read_text())
            if r['revision'] == SB_REVISION and digest(path) == r['sha256']:
                return path
        raise ValueError(f"Unverified StatsBomb cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            with urlopen(SB_BASE + relative, timeout=90) as response, temporary.open('wb') as f:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(block)
            checksum = digest(temporary)
            temporary.replace(path)
            receipt.write_text(json.dumps({'revision': SB_REVISION, 'sha256': checksum,
                                           'url': SB_BASE + relative, 'bytes': path.stat().st_size}, indent=2))
            return path
        except (URLError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def sb_shots(events, match_id, exclusions=None):
    result = []
    for event in events:
        if event['type']['name'] != 'Shot' or event['period'] == 5:
            continue  # Penalty shoot-outs do not belong to match play.
        loc = event.get('location')
        if loc is None or len(loc) < 2 or not np.isfinite(loc[:2]).all():
            raise ValueError(f"Missing shot position: {match_id}/{event['id']}")
        if not (0 <= loc[0] <= 120 and 0 <= loc[1] <= 80):
            if exclusions is None:
                raise ValueError(f"Invalid StatsBomb coordinates: {loc}")
            exclusions.append({'match_id': int(match_id), 'event_id': event['id'],
                               'reason': 'shot location outside pitch', 'location': loc[:2]})
            continue
        outcome = event['shot']['outcome']['name']
        if outcome not in {'Goal', 'Blocked', 'Off T', 'Post', 'Saved', 'Wayward', 'Saved Off Target', 'Saved to Post'}:
            raise ValueError(f"Unknown shot outcome: {outcome}")
        result.append({'match_id': int(match_id), 'event_id': event['id'],
                       'distance_x': (120 - loc[0]) * 105 / 120,
                       'lateral_y': (loc[1] - 40) * 68 / 80,
                       'goal': int(outcome == 'Goal')})
    if len({r['event_id'] for r in result}) != len(result):
        raise ValueError('Duplicate StatsBomb shots')
    return result


def download_shots(root, output, workers=4):
    """Fetch six fixed men's competition-seasons; retain raw inputs + checksums."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for relative in ('README.md', 'LICENSE.pdf', 'data/competitions.json'):
        cached_file(root, relative)
    match_ids, groups = [], []
    for competition, season in SEASONS:
        path = cached_file(root, f'data/matches/{competition}/{season}.json')
        matches = json.loads(path.read_text())
        ids = sorted(int(m['match_id']) for m in matches)
        match_ids.extend(ids)
        groups.append({'competition_id': competition, 'season_id': season, 'matches': ids})
    if len(match_ids) != len(set(match_ids)):
        raise ValueError('Overlapping competition inventories')

    def one(mid):
        path = cached_file(root, f'data/events/{mid}.json')
        excluded = []
        shots = sb_shots(json.loads(path.read_text()), mid, exclusions=excluded)
        return shots, {'match_id': mid, 'sha256': digest(path)}, excluded

    shots, checksums, exclusions = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (rows, checksum, excluded) in enumerate(pool.map(one, sorted(match_ids))):
            shots.extend(rows)
            checksums.append(checksum)
            exclusions.extend(excluded)
            if (index + 1) % 50 == 0 or index + 1 == len(match_ids):
                print(f"StatsBomb: {index + 1}/{len(match_ids)} matches, {len(shots)} shots", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(shots[0]))
        writer.writeheader()
        writer.writerows(shots)
    manifest = {'revision': SB_REVISION, 'seasons': groups, 'inputs': checksums,
                'shots': len(shots), 'shootouts_excluded': True, 'exclusions': exclusions,
                'shots_sha256': digest(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def geometry(distance_x, lateral_y):
    """Physical geometry; reflection-invariant. Goal width assumed 7.32 metres."""
    dx, y = np.asarray(distance_x, dtype=float), np.asarray(lateral_y, dtype=float)
    if not np.isfinite(dx).all() or not np.isfinite(y).all() or (dx < 0).any():
        raise ValueError('Invalid shot geometry (including behind-goal positions)')
    d = np.hypot(dx, y)
    angle = np.arctan2(7.32 * dx, dx * dx + y * y - 3.66 ** 2)
    return np.stack([d, angle, np.log1p(d), np.abs(y), d ** 2,
                     angle ** 2, d * angle, y ** 2], axis=-1)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(x, y, penalty):
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    w = np.zeros(design.shape[1])
    w[0] = np.log(y.mean() / (1 - y.mean()))
    regularizer = np.diag([0.] + [float(penalty)] * x.shape[1])

    def objective(weights):
        logits = design @ weights
        return np.sum(np.logaddexp(0, logits) - y * logits) + .5 * weights @ regularizer @ weights

    for _ in range(100):
        p = sigmoid(design @ w)
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + regularizer + np.eye(len(w)) * 1e-9
        step = np.linalg.solve(hessian, design.T @ (p - y) + regularizer @ w)
        old = objective(w)
        rate = 1.
        while rate > 1e-8 and objective(w - rate * step) > old:
            rate *= .5
        w -= rate * step
        if np.max(np.abs(rate * step)) < 1e-8:
            break
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': w.tolist(), 'penalty': penalty}


def probability(model, x):
    design = np.column_stack([np.ones(len(x)), (x - np.array(model['mean'])) / np.array(model['scale'])])
    return sigmoid(design @ np.array(model['weights']))


def binary_metrics(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    pos, neg = y.sum(), (1 - y).sum()
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    auc = (float((average_ranks[inverse][y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
           if pos and neg else None)
    bins = np.minimum((p * 10).astype(int), 9)
    calibration = [{'n': int((bins == b).sum()), 'predicted': float(p[bins == b].mean()),
                    'observed': float(y[bins == b].mean())} for b in range(10) if (bins == b).any()]
    return {'n': len(y), 'goals': int(pos), 'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            'brier': float(np.mean((p - y) ** 2)), 'auc': auc,
            'ece_10_equal_width': sum(b['n'] * abs(b['predicted'] - b['observed']) for b in calibration) / len(y),
            'mean_prediction': float(p.mean()), 'goal_rate': float(y.mean()), 'calibration_bins': calibration}


def fit_xg(shots_csv, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with Path(shots_csv).open(newline='') as f:
        rows = list(csv.DictReader(f))
    provenance = json.loads(Path(shots_csv).with_suffix('.manifest.json').read_text())
    if provenance['shots_sha256'] != digest(shots_csv) or provenance['revision'] != SB_REVISION:
        raise ValueError('StatsBomb shot table provenance mismatch')
    ids = np.array([int(r['match_id']) for r in rows])
    x = geometry([float(r['distance_x']) for r in rows], [float(r['lateral_y']) for r in rows])
    y = np.array([int(r['goal']) for r in rows])
    games = np.random.default_rng(1701).permutation(np.unique(ids))
    a, b = int(.7 * len(games)), int(.85 * len(games))
    split = {k: sorted(v.tolist()) for k, v in zip(('train', 'val', 'test'), (games[:a], games[a:b], games[b:]))}
    train, val, test = [np.isin(ids, split[k]) for k in ('train', 'val', 'test')]
    if any(y[t].sum() == 0 or y[t].sum() == t.sum() for t in (train, val, test)):
        raise ValueError('Each xG split needs goals and non-goals')
    candidates = []
    for penalty in (.1, 1., 10., 100.):
        model = fit_logistic(x[train], y[train], penalty)
        candidates.append((binary_metrics(y[val], probability(model, x[val]))['log_loss'], model))
    _, model = min(candidates, key=lambda pair: pair[0])
    model.update({'features': GEOMETRY, 'source_revision': SB_REVISION, 'target': 'goal',
                  'shot_table_sha256': digest(shots_csv), 'feature_version': 1})
    prior = float(y[train].mean())
    predictions = probability(model, x)
    metrics = {k: binary_metrics(y[mask], predictions[mask]) for k, mask in [('train', train), ('val', val), ('test', test)]}
    reference = binary_metrics(y[test], np.full(test.sum(), prior))
    # Paired bootstrap over complete test matches, retaining shot weighting.
    p = np.clip(predictions[test], 1e-9, 1 - 1e-9)
    yt, it = y[test], ids[test]
    delta = -(yt * np.log(p) + (1 - yt) * np.log(1 - p)) + yt * np.log(prior) + (1 - yt) * np.log(1 - prior)
    totals = np.array([[delta[it == mid].sum(), (it == mid).sum()] for mid in split['test']])
    indices = np.random.default_rng(123).integers(len(totals), size=(2000, len(totals)))
    sums = totals[indices].sum(1)
    interval = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975]).tolist()
    output.mkdir(parents=True)
    model_path = output / 'model.json'
    model_path.write_text(json.dumps(model, indent=2) + '\n')
    report = {'statsbomb_revision': SB_REVISION, 'split': split, 'split_seed': 1701,
              'n_matches': len(games), 'n_shots': len(rows), 'coordinate_exclusions': provenance['exclusions'],
              'versions': {'python': platform.python_version(), 'numpy': np.__version__},
              'source_sha256': digest(__file__),
              'selection': [{'penalty': m['penalty'], 'validation_log_loss': ll} for ll, m in candidates],
              'selected_penalty': model['penalty'], 'metrics': metrics, 'training_rate_test_baseline': reference,
              'test_log_loss_difference_ci95': interval, 'model_sha256': digest(model_path),
              'skillcorner_matches_used_for_xg_fit_or_selection': 0,
              'limitations': ['Position-only model; no body part, penalty, keeper or pressure features',
                              'StatsBomb coordinates approximated on a 105 by 68 metre pitch',
                              'StatsBomb holdout does not validate A-League transfer calibration']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def label_skillcorner(root, match_ids, model_dir, output):
    output, model_dir = Path(output), Path(model_dir)
    if output.exists():
        raise FileExistsError(output)
    model_path = model_dir / 'model.json'
    model = json.loads(model_path.read_text())
    fit_report = json.loads((model_dir / 'report.json').read_text())
    if digest(model_path) != fit_report['model_sha256'] or model.get('features') != GEOMETRY:
        raise ValueError('Unsupported or modified xG model')
    rows, inputs = [], {}
    source = f"statsbomb_geometry_v1:{digest(model_path)}"
    for mid in match_ids:
        meta = json.loads(match_path(root, mid, 'match.json').read_text())
        for suffix in ('match.json', 'dynamic_events.csv'):
            inputs[f'{mid}_{suffix}'] = digest(match_path(root, mid, suffix))
        for event in shot_events(read_events(root, mid)):
            # Dynamic-event positions are already rotated to attack +x, in
            # actual-pitch metres. Do NOT rotate again using attacking_side.
            dx = meta['pitch_length'] / 2 - float(event['x_end'])
            lateral = float(event['y_end'])
            if dx < 0:
                raise ValueError(f"Shot behind goal requires review: {mid}/{event['event_id']}")
            p = float(probability(model, geometry([dx], [lateral]))[0])
            rows.append({'match_id': int(mid), 'event_id': event['event_id'], 'period': int(event['period']),
                         'frame': int(event['frame_end']), 'team_id': int(event['team_id']),
                         'xg': p, 'xg_source': source})
    if not rows:
        raise ValueError('No SkillCorner shot markers')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'skillcorner_revision': REVISION, 'statsbomb_revision': SB_REVISION,
              'model_sha256': digest(model_path), 'labels_sha256': digest(output), 'inputs': inputs,
              'n_labels': len(rows), 'min_xg': min(r['xg'] for r in rows), 'max_xg': max(r['xg'] for r in rows),
              'mean_xg': float(np.mean([r['xg'] for r in rows])), 'target_kind': 'estimated xG, not provider xG',
              'coordinate_rule': 'goal distance = actual pitch_length/2 - attack-normalized event x_end',
              'model_input': 'shot geometry only; no outcomes or provider xshot/EPV'}
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
