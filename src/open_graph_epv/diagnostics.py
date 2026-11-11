"""Read-only diagnostics of the frozen contextual EPV experiment.

No model selection or fitting. Perturbations describe functional sensitivity,
not causal effects or movement recommendations.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .contextual_epv import TARGET, verify_data, period_for_frame
from .data import frame_graph
from .evaluation import verify_selection
from .source import digest, match_path
from .training import load_epv, predict

FAMILIES = ('gnn', 'set', 'ball')
ROLE_NAMES = ('teammate', 'opponent', 'own_keeper', 'opponent_keeper')


def player_role(node):
    if node[6]:
        return None
    if node[7]:
        return 'own_keeper' if node[4] else 'opponent_keeper'
    return 'teammate' if node[4] else 'opponent'


def position_variants(nodes, mask, half_pitch, metres=1.0):
    """Move one observed player only; do not clip or alter velocities/roles.

    Skip a player already outside the pitch and proposed moves outside it.
    Return variants and originating node slots, retaining exact masks.
    """
    variants, slots = [], []
    if metres <= 0 or not np.isfinite(metres):
        raise ValueError('Positive finite displacement required')
    half_pitch = np.asarray(half_pitch, float)
    if half_pitch.shape != (2,) or not np.isfinite(half_pitch).all() or (half_pitch <= 0).any():
        raise ValueError('Invalid pitch dimensions')
    for slot in np.flatnonzero(mask & (nodes[:, 6] == 0)):
        if (np.abs(nodes[slot, :2]) > 1).any():
            continue
        for axis in (0, 1):
            for sign in (-1, 1):
                value = nodes[slot, axis] + sign * metres / half_pitch[axis]
                if abs(value) > 1:
                    continue
                variant = nodes.copy()
                variant[slot, axis] = value
                variants.append(variant)
                slots.append(int(slot))
    return np.asarray(variants, dtype=nodes.dtype).reshape(-1, *nodes.shape), np.asarray(slots, int)


def home_perspective(nodes, possession_is_home):
    """Convert possession-oriented positions/velocities to home attacking +x."""
    result = nodes.copy()
    if not possession_is_home:
        result[:, :4] *= -1
    return result


def ensemble(models, x, mask):
    x, mask = torch.from_numpy(x), torch.from_numpy(mask)
    return np.mean([predict(m, x, mask, batch_size=64) for m in models], axis=0)


def error_groups(y, prediction):
    total = float(np.square(y - prediction).sum())
    result = {}
    for name, use in (('no_shot', y == 0), ('own_shot', y > 0), ('opponent_shot', y < 0)):
        n = int(use.sum())
        result[name] = {'n': n, 'fraction': n / len(y),
                        'mse_share': float(np.square(y[use] - prediction[use]).sum() / total) if total else 0.,
                        'rmse': float(np.sqrt(np.square(y[use] - prediction[use]).mean())) if n else None,
                        'target_mean': float(y[use].mean()) if n else None,
                        'prediction_mean': float(prediction[use].mean()) if n else None}
    return result


def value_bins(y, prediction):
    # Equal-count bins only when boundaries do not split ties.
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, 6)))
    groups = np.searchsorted(edges[1:-1], prediction, side='right')
    return [{'n': int((groups == k).sum()), 'prediction': float(prediction[groups == k].mean()),
             'target': float(y[groups == k].mean())} for k in np.unique(groups)]


def shot_error_groups(y, prediction, matches, shot_frames, kinds):
    """Attribute overlapping windows' error to their first future shot marker."""
    error = np.square(y.astype(float) - prediction)
    total = float(error.sum())
    keys = sorted(set(zip(matches[shot_frames >= 0], shot_frames[shot_frames >= 0])))
    rows = []
    for mid, frame in keys:
        use = (matches == mid) & (shot_frames == frame)
        sources = np.unique(kinds[use])
        if len(sources) != 1 or sources[0] not in (1, 2) or not np.allclose(np.abs(y[use]), abs(y[use][0])):
            raise ValueError('Inconsistent shot provenance')
        rows.append({'match_id': int(mid), 'shot_frame': int(frame), 'n_windows': int(use.sum()),
                     'xg': float(abs(y[use][0])), 'source': 'context' if sources[0] == 1 else 'geometry',
                     'mse_share': float(error[use].sum() / total) if total else 0.})
    rows.sort(key=lambda r: -r['mse_share'])
    return {'n_distinct_future_shots': len(rows), 'ranked_shots': rows}


def choose_cases(y, prediction, matches, frames):
    """Explicitly hindsight-selected examples, never an unbiased performance set."""
    nonzero = y != 0
    residual = np.abs(y - prediction['gnn'])
    rules = [
        ('Stor framtidig egen sjanse', 'Størst positiv target; valgt med etterpåklokskap.', y > 0, -y),
        ('Stor framtidig motstandersjanse', 'Mest negativ target; valgt med etterpåklokskap.', y < 0, y),
        ('Høy verdi uten kommende skudd', 'Høyest GNN-verdi blant vinduer uten skudd.', y == 0, -prediction['gnn']),
        ('GNN og ballmodell er uenige', 'Størst absolutt forskjell mellom GNN og ballmodell.', np.ones(len(y), bool),
         -np.abs(prediction['gnn'] - prediction['ball'])),
        ('Typisk feil i et skuddvindu', 'Nær median absolutt GNN-feil blant skuddvinduer.', nonzero,
         np.abs(residual - np.median(residual[nonzero]))),
        ('Typisk feil uten skudd', 'Nær median absolutt GNN-feil blant vinduer uten skudd.', ~nonzero,
         np.abs(residual - np.median(residual[~nonzero]))),
    ]
    cases = []
    for title, reason, eligible, score in rules:
        ordered = np.flatnonzero(eligible)[np.argsort(score[eligible], kind='stable')]
        for i in ordered:
            if any(matches[i] == matches[c['index']] and abs(frames[i] - frames[c['index']]) < 500 for c in cases):
                continue
            cases.append({'index': int(i), 'title': title, 'selection': reason})
            break
    return cases


def sensitivity(models, x, masks, mids, metadata, indices, predictions):
    """Finite changes in xG units for +/-1m moves. No label data used."""
    result = {}
    for number, i in enumerate(indices):
        meta = metadata[int(mids[i])]
        half = np.array([meta['pitch_length'], meta['pitch_width']]) / 2
        variants, slots = position_variants(x[i], masks[i], half)
        record = {'roles': [player_role(n) if masks[i, slot] else None for slot, n in enumerate(x[i])],
                  'n_moves': int(len(slots)), 'models': {}}
        for family in FAMILIES:
            delta = np.abs(ensemble(models[family], variants, np.repeat(masks[i][None], len(slots), axis=0))
                           - predictions[family][i]) if len(slots) else np.array([])
            record['models'][family] = [float(delta[slots == slot].mean()) if (slots == slot).any() else None
                                        for slot in range(x.shape[1])]
        result[int(i)] = record
        if (number + 1) % 40 == 0:
            print(f'Local sensitivity: {number + 1}/{len(indices)} snapshots', flush=True)
    return result


def summarize_sensitivity(local, sample):
    result = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for role in ROLE_NAMES:
            means, nodes = [], 0
            for i in sample:
                r = local[int(i)]
                values = [v for kind, v in zip(r['roles'], r['models'][family]) if kind == role and v is not None]
                if values:
                    means.append(float(np.mean(values)))
                    nodes += len(values)
            result[family][role] = {'n_snapshots': len(means), 'n_player_observations': nodes,
                'mean_abs_change': float(np.mean(means)) if means else None,
                'median_abs_change': float(np.median(means)) if means else None,
                'p90_abs_change': float(np.quantile(means, .9)) if means else None}
    return result


def scene_records(root, metadata, x, masks, matches, frames, wanted):
    scenes = {}
    for mid in sorted(set(int(matches[i]) for i in wanted)):
        lookup = {int(frames[i]): int(i) for i in wanted if int(matches[i]) == mid}
        meta = metadata[mid]
        players = {p['id']: p for p in meta['players']}
        previous = None
        with match_path(root, mid, 'tracking_extrapolated.jsonl').open() as handle:
            for line in handle:
                frame = json.loads(line)
                if frame['frame'] in lookup:
                    i = lookup[frame['frame']]
                    graph = frame_graph(frame, previous, meta)
                    if graph is None or not np.array_equal(graph[0], x[i]) or not np.array_equal(graph[1], masks[i]):
                        raise ValueError('Raw scene differs from frozen graph')
                    is_home = graph[2] == meta['home_team']['id']
                    oriented = home_perspective(x[i], is_home)
                    numbers = ['B'] + [str(players[p['player_id']]['number']) for p in frame['player_data']]
                    points = []
                    for slot in np.flatnonzero(masks[i]):
                        node = oriented[slot]
                        points.append({'slot': int(slot), 'x': float(node[0] * meta['pitch_length'] / 2),
                            'y': float(node[1] * meta['pitch_width'] / 2),
                            'vx': float(node[2] * 10), 'vy': float(node[3] * 10),
                            'ball': bool(node[6]), 'keeper': bool(node[7]),
                            'home': bool(node[4]) == is_home if not node[6] else None,
                            'number': numbers[slot]})
                    scenes[i] = {'points': points, 'sign': 1 if is_home else -1,
                                 'period': frame['period'], 'possession': 'home' if is_home else 'away'}
                previous = frame
        print(f'Verified raw scenes: {mid}', flush=True)
    if set(scenes) != set(wanted):
        raise ValueError('Missing exact scene frames')
    return scenes


def make_figures(out, report, x, masks, y, predictions, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle, Circle

    colors = {'gnn': '#167c80', 'set': '#8c59a0', 'ball': '#d47727'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    ax = axes[0, 0]
    all_values = []
    for family in FAMILIES:
        bins = report['value_bins'][family]
        p, t = [b['prediction'] for b in bins], [b['target'] for b in bins]
        all_values.extend(p + t)
        ax.plot(p, t, 'o-', color=colors[family], label=family)
    lo, hi = min(all_values), max(all_values)
    ax.plot([lo, hi], [lo, hi], '--', color='gray')
    ax.set(title='Conditional mean check: 5 prediction bins', xlabel='Mean predicted signed xG',
           ylabel='Mean realized signed teacher target')
    ax.legend()
    ax = axes[0, 1]
    groups = report['error_groups']['gnn']
    labels = ['No shot', 'Own shot', 'Opponent shot']
    for j, (key, label) in enumerate(zip(groups, labels)):
        r = groups[key]
        ax.bar(j - .18, 100 * r['fraction'], width=.36, color='#b4c8cb', label='Share of snapshots' if j == 0 else None)
        ax.bar(j + .18, 100 * r['mse_share'], width=.36, color=colors['gnn'], label='Share of squared error' if j == 0 else None)
        ax.text(j, max(r['fraction'], r['mse_share']) * 100 + 2, f"n={r['n']}", ha='center', fontsize=9)
    ax.set(xticks=range(3), xticklabels=labels, ylabel='Percent', ylim=(0, 110), title='Where GNN error occurs')
    ax.legend()
    ax = axes[1, 0]
    for j, family in enumerate(FAMILIES):
        values = [report['sensitivity'][family][role]['mean_abs_change'] or 0 for role in ROLE_NAMES]
        ax.bar(np.arange(4) + (j - 1) * .24, values, width=.24, color=colors[family], label=family)
    ax.set(xticks=range(4), xticklabels=['Teammates', 'Opponents', 'Own GK', 'Opponent GK'],
           ylabel='Mean absolute EPV change (xG)', title='Move one player 1 m; functional sensitivity')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend()
    ax = axes[1, 1]
    zones = report['ball_zones']
    for family in FAMILIES:
        ax.plot(range(3), [r['prediction_means'][family] for r in zones], 'o-', color=colors[family], label=family)
    ax.plot(range(3), [r['target_mean'] for r in zones], 's--', color='#253342', label='Realized teacher target')
    ax.set(xticks=range(3), xticklabels=['Own third', 'Middle third', 'Attacking third'],
           title='Value by ball location (observational)', ylabel='Mean signed xG')
    ax.legend(fontsize=8)
    fig.suptitle('What does contextual EPV learn? 4,228 snapshots / 3 previously examined matches\n'
                 'Target averages are not goal calibration; local perturbations are not causal effects.', fontsize=13)
    fig.savefig(out / 'diagnostics_overview.png', dpi=170)

    def pitch(ax, length=105, width=68):
        ax.add_patch(Rectangle((-length/2, -width/2), length, width, fill=False, color='#779187'))
        ax.plot([0, 0], [-width/2, width/2], color='#779187', lw=.7)
        ax.add_patch(Circle((0, 0), 9.15, fill=False, color='#779187', lw=.7))
        for side in (-1, 1):
            start = -length/2 if side == -1 else length/2 - 16.5
            ax.add_patch(Rectangle((start, -20.16), 16.5, 40.32, fill=False, color='#779187', lw=.7))
        ax.set(xlim=(-length/2-2, length/2+2), ylim=(-width/2-3, width/2+3), aspect='equal')
        ax.set_xticks([]);ax.set_yticks([])

    heat, axs = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    ball = (x[:, :, :2] * (x[:, :, 6] * masks)[..., None]).sum(1)
    counts, xe, ye = np.histogram2d(ball[:, 0], ball[:, 1], bins=(12, 8), range=((-1, 1), (-1, 1)))
    maps = []
    for values in (predictions['gnn'], y, predictions['gnn'] - predictions['ball']):
        sums, _, _ = np.histogram2d(ball[:, 0], ball[:, 1], bins=(xe, ye), weights=values)
        maps.append(np.where(counts >= 20, sums / np.maximum(counts, 1), np.nan))
    limit = max(float(np.nanmax(np.abs(maps[i]))) for i in (0, 1))
    for j, (ax, values, title) in enumerate(zip(axs, maps, ('Mean GNN value', 'Mean realized target', 'Mean GNN minus ball model'))):
        bound = limit if j < 2 else max(float(np.nanmax(np.abs(values))), 1e-8)
        im = ax.imshow(values.T, origin='lower', extent=(-52.5, 52.5, -34, 34), cmap='RdBu_r', vmin=-bound, vmax=bound)
        pitch(ax);ax.set_title(title);heat.colorbar(im, ax=ax, shrink=.65, label='Signed xG')
    heat.suptitle('Observed ball-location bins: possession team attacks right; minimum 20 snapshots per bin\n'
                  'Normalized pitch display; these are observed means, not hypothetical ball-position values.', fontsize=12)
    heat.savefig(out / 'diagnostics_ball_location.png', dpi=170)
    snapshots, axs = plt.subplots(3, 2, figsize=(12, 13), layout='constrained')
    for ax, case in zip(axs.flat, cases):
        state = case['states'][case['anchor_position']]
        pitch(ax, case['pitch_length'], case['pitch_width'])
        for point in state['points']:
            slot = point['slot'];delta = state['sensitivity']['gnn'][slot]
            size = 65 + min((delta or 0) / .0001, 5) * 45
            color = '#f6bf45' if point['ball'] else ('#178caa' if point['home'] else '#e55f58')
            ax.scatter(point['x'], point['y'], s=size, c=color, marker='s' if point['keeper'] else 'o', edgecolors='white', linewidths=.6)
            if not point['ball']:
                ax.text(point['x'], point['y'], point['number'], ha='center', va='center', fontsize=6, color='white')
        ax.set_title(f"{case['title']}\nGNN {state['values']['gnn']:+.4f} | target {state['target']:+.4f}", fontsize=10)
        ax.set_xlabel(f"Match {case['match_id']}, frame {state['frame']}; home attacks right", fontsize=8)
    snapshots.suptitle('Hindsight-selected case studies; home-team signed values\nMarker size: GNN sensitivity to a 1 m player move, not player importance or recommended movement.', fontsize=12)
    snapshots.savefig(out / 'diagnostics_cases.png', dpi=160)
    with PdfPages(out / 'diagnostics.pdf') as pdf:
        for figure in (fig, heat, snapshots):
            pdf.savefig(figure)
            plt.close(figure)


def run(data, runs, root, output, template, sample_size=96):
    data, runs, root, output, template = map(Path, (data, runs, root, output, template))
    if output.exists():
        raise FileExistsError('Choose a new diagnostics output directory')
    if sample_size < 1:
        raise ValueError('Positive sample size required')
    torch.set_num_threads(2)
    manifest = verify_data(data)
    selection = json.loads((runs / 'selection.json').read_text())
    verify_selection(data, runs, selection)
    audit = json.loads((runs / 'audit/report.json').read_text())
    if audit.get('target') != TARGET or audit['selection_sha256'] != digest(runs / 'selection.json'):
        raise ValueError('Wrong contextual audit')
    if audit['predictions_sha256'] != digest(runs / 'audit/predictions.npz'):
        raise ValueError('Frozen predictions changed')
    with np.load(data / 'graphs.npz', allow_pickle=False) as z:
        use = np.isin(z['match_id'], selection['split']['test'])
        x, masks, y, mids, frames = [z[k][use] for k in ('x', 'mask', 'y', 'match_id', 'frame')]
    with np.load(runs / 'audit/predictions.npz', allow_pickle=False) as z:
        for key, array in (('target', y), ('match_id', mids), ('frame', frames)):
            if not np.array_equal(z[key], array):
                raise ValueError('Prediction row alignment changed')
        predictions = {family: z[family] for family in FAMILIES}
    with np.load(data / 'target_provenance.npz', allow_pickle=False) as z:
        provenance_use = np.isin(z['match_id'], selection['split']['test'])
        if not np.array_equal(z['match_id'][provenance_use], mids) or not np.array_equal(z['frame'][provenance_use], frames):
            raise ValueError('Target provenance row alignment changed')
        shot_frames, kinds = z['future_shot_frame'][provenance_use], z['kind'][provenance_use]
    models = {family: [load_epv(runs / record['name']) for record in selection['models'] if record['family'] == family]
              for family in FAMILIES}
    metadata = {}
    for mid in np.unique(mids):
        for suffix in ('match.json', 'tracking_extrapolated.jsonl'):
            path = match_path(root, int(mid), suffix)
            if digest(path) != manifest['inputs'][str(path.relative_to(root))]:
                raise ValueError('Raw source changed')
        metadata[int(mid)] = json.loads(match_path(root, int(mid), 'match.json').read_text())
    periods = np.array([period_for_frame(metadata[int(mid)], int(frame))['period'] for mid, frame in zip(mids, frames)])
    cases = choose_cases(y, predictions, mids, frames)
    wanted = set()
    for case in cases:
        i = case['index']
        indices = np.flatnonzero((mids == mids[i]) & (periods == periods[i]) & (abs(frames - frames[i]) <= 200))
        case['indices'] = indices.tolist()
        wanted.update(indices.tolist())
    sample = np.unique(np.linspace(0, len(y) - 1, min(sample_size, len(y)), dtype=int))
    # Recompute a deterministic sample as an independent inference check.
    for family in FAMILIES:
        if not np.allclose(ensemble(models[family], x[sample], masks[sample]), predictions[family][sample], atol=1e-7, rtol=1e-5):
            raise ValueError('Current inference differs from the frozen predictions')
    local = sensitivity(models, x, masks, mids, metadata, sorted(wanted | set(sample.tolist())), predictions)
    scenes = scene_records(root, metadata, x, masks, mids, frames, wanted)
    case_records = []
    for case in cases:
        anchor = case['index'];mid = int(mids[anchor]);meta = metadata[mid]
        states = []
        for i in case['indices']:
            scene = scenes[i];sign = scene['sign']
            period = period_for_frame(meta, int(frames[i]))
            states.append({**scene, 'frame': int(frames[i]), 'seconds': float((frames[i] - frames[anchor]) / 10),
                'period_seconds': float((frames[i] - period['start_frame']) / 10),
                'target': float(sign * y[i]), 'values': {f: float(sign * predictions[f][i]) for f in FAMILIES},
                'sensitivity': local[i]['models']})
        case_records.append({'title': case['title'], 'selection': case['selection'], 'match_id': mid,
            'home': meta['home_team']['name'], 'away': meta['away_team']['name'],
            'pitch_length': meta['pitch_length'], 'pitch_width': meta['pitch_width'],
            'anchor_position': case['indices'].index(anchor), 'states': states})
    ball_x = (x[:, :, 0] * x[:, :, 6] * masks).sum(1)
    groups = np.searchsorted([-1/3, 1/3], ball_x, side='right')
    zones = [{'zone': name, 'n': int((groups == k).sum()), 'target_mean': float(y[groups == k].mean()),
              'prediction_means': {f: float(predictions[f][groups == k].mean()) for f in FAMILIES}}
             for k, name in enumerate(('own_third', 'middle_third', 'attacking_third'))]
    motion = {}
    for kind in ('player_motion', 'all_motion'):
        changed = x[sample].copy()
        selected = masks[sample] & ((changed[:, :, 6] == 0) if kind == 'player_motion' else True)
        for column in (2, 3, 9):
            changed[:, :, column][selected] = 0
        motion[kind] = {}
        for f in FAMILIES:
            delta = np.abs(ensemble(models[f], changed, masks[sample]) - predictions[f][sample])
            motion[kind][f] = {'mean_abs_change': float(delta.mean()), 'p90_abs_change': float(np.quantile(delta, .9))}
    report = {'date': '2026-11-10', 'target': TARGET, 'n_snapshots': len(y), 'n_matches': len(metadata),
        'selected_family': selection['selected_family'], 'teacher_coverage': manifest['teacher_coverage'],
        'value_bins': {f: value_bins(y, predictions[f]) for f in FAMILIES},
        'error_groups': {f: error_groups(y, predictions[f]) for f in FAMILIES},
        'shot_errors': shot_error_groups(y, predictions['gnn'], mids, shot_frames, kinds),
        'sensitivity': summarize_sensitivity(local, sample), 'motion_ablation': motion, 'ball_zones': zones,
        'sensitivity_sample': {'rule': 'Evenly spaced row indices in frozen test order; independent of targets and predictions',
                               'n': len(sample), 'indices': sample.tolist(),
                               'n_matches': len(np.unique(mids[sample]))},
        'case_selection': [{'title': c['title'], 'rule': c['selection'], 'match_id': c['match_id'],
                            'anchor_frame': c['states'][c['anchor_position']]['frame']} for c in case_records],
        'limitations': ['Previously examined three test matches; diagnostics do not validate generalization.',
            'Realized first-shot teacher targets are noisy outcomes, not the true conditional expected value.',
            'Hindsight-selected cases are illustrative, not representative evidence.',
            'Fixed-velocity 1m perturbations may be out of distribution; sensitivities are not causal effects or attribution.',
            'Motion ablations set velocity and validity to zero; missing-motion scenes may be out of distribution.',
            'Observed heatmaps mix different situations; they do not describe the effect of moving the ball.',
            'Missing or extrapolated players remain masked; no interpolation or invented positions.'],
        'provenance': {'dataset_sha256': manifest['graphs_sha256'], 'selection_sha256': digest(runs / 'selection.json'),
            'audit_sha256': digest(runs / 'audit/report.json'), 'diagnostics_source_sha256': digest(Path(__file__)),
            'template_sha256': digest(template), 'raw_scenes_match_frozen_graphs': True,
            'sample_predictions_match_frozen_audit': True, 'training_performed': False}}
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    payload = {'cases': case_records, 'report': report}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    html = template.read_text()
    if html.count('__EPV_DATA__') != 1:
        raise ValueError('Template needs one data placeholder')
    (output / 'explorer.html').write_text(html.replace('__EPV_DATA__', encoded))
    make_figures(output, report, x, masks, y, predictions, case_records)
    write_report(output, report)
    print(f'Wrote {output}/explorer.html, diagnostics.pdf, three figures and report.md', flush=True)
    return report


def write_report(out, report):
    g = report['error_groups']['gnn']
    top_shot = report['shot_errors']['ranked_shots'][0]
    shot_rows = '\n'.join(f"| {r['match_id']} | {r['shot_frame']} | {r['n_windows']} | {r['xg']:.4f} | {r['source']} | {100*r['mse_share']:.1f} % |"
                          for r in report['shot_errors']['ranked_shots'][:5])
    sensitivity_rows = '\n'.join('| ' + role + ' | ' + ' | '.join(
        f"{report['sensitivity'][f][role]['mean_abs_change']:.7f}" if report['sensitivity'][f][role]['mean_abs_change'] is not None else '—'
        for f in FAMILIES) + ' |' for role in ROLE_NAMES)
    motion_rows = '\n'.join('| ' + kind + ' | ' + ' | '.join(f"{report['motion_ablation'][kind][f]['mean_abs_change']:.7f}" for f in FAMILIES) + ' |'
                            for kind in report['motion_ablation'])
    text = f'''# Hva lærer den kontekstuelle EPV-modellen?

Diagnostikk 10. november 2026 av ferdigtrente modeller. Ingen ny fitting eller modellseleksjon. Vi bruker alle **{report['n_snapshots']} snapshots fra {report['n_matches']} tidligere undersøkte testkamper** for aggregatene. GNN, spillermodell uten meldinger (`set`) og ballmodell undersøkes på samme situasjoner.

Åpne [den interaktive sekvensviseren](explorer.html) i nettleseren. Den fungerer lokalt uten server, nedlasting eller eksterne biblioteker. Velg sekvens, modell og tidspunkt; spill av de observerte snapshotene. [PDF med tre figursider](diagnostics.pdf) kan brukes uten nettleser.

## Hva figurene måler

![Samlet diagnostikk](diagnostics_overview.png)

**Middelverdier:** Fem grupper etter predikert verdi sammenligner gjennomsnittlig prediksjon med gjennomsnittlig realisert lærer-target. Dette undersøker EPV-ens middelverdier, ikke kalibrering mot faktiske mål. Grupper med få sjeldne skudd kan variere mye med bare tre kamper.

**Feilfordeling:** {100*g['no_shot']['fraction']:.1f} % av vinduene har ikke skudd i horisonten, men disse står for {100*g['no_shot']['mse_share']:.1f} % av GNN-ens samlede kvadrerte feil. Skuddvinduene står for resten. Den realiserte skuddverdien er et mulig framtidig utfall; en korrekt betinget forventning behøver ikke være lik xG-verdien i et vindu der vi etterpå vet at et skudd kom. Denne grupperingen beviser derfor ikke underkalibrering.

**Ett skudd dominerer feilen:** De ikke-null vinduene viser fram mot {report['shot_errors']['n_distinct_future_shots']} forskjellige skuddmarkører. Skuddet i kamp {top_shot['match_id']}, frame {top_shot['shot_frame']}, har estimert xG {top_shot['xg']:.4f}. Dets {top_shot['n_windows']} overlappende vinduer står for **{100*top_shot['mse_share']:.1f} % av all kvadrert GNN-feil**. Dette gjør rådatakontroll og kontroll av lærerens høye skuddverdier særlig relevant. Skuddet er beholdt; feildominans er ikke bevis på en feilaktig label eller grunn til å fjerne det.

| Kamp | Skuddframe | Overlappende vinduer | Estimert xG | Kilde | Andel total kvadrert feil |
|---|---:|---:|---:|---|---:|
{shot_rows}

**Lokal følsomhet:** Vi flytter én observert spiller én meter i hver av fire retninger, uten å endre ballen, andre spillere, lag, keeperrolle eller hastigheter. Flytt utenfor banen utelates. For hver spiller beregnes gjennomsnittlig absolutt verdiendring. Tabellen bruker {report['sensitivity_sample']['n']} jevnt fordelte test-snapshots valgt uten bruk av target eller prediksjon. Først gjennomsnitt innen rolle per snapshot, deretter mellom snapshots der rollen har gyldige flytt. Ingen konfidensintervaller eller bevis på kausal betydning hevdes.

| Rolle | GNN: absolutt endring | Uten meldinger | Bare ball |
|---|---:|---:|---:|
{sensitivity_rows}

En større endring viser at modellens beregning reagerer mer på disse små posisjonsendringene. Det betyr ikke at spilleren er viktigere for utfallet, eller at flyttingen er et godt løp. Den endrede scenen kan ligge utenfor treningsfordelingen, særlig fordi hastigheten holdes fast. Ballmodellen skal være uendret når bare spillere flyttes og er en kontroll for beregningen.

Vi undersøker også å fjerne informasjon om bevegelse. Både hastighet og `velocity_valid` settes til null. Dette er en kunstig informasjonsfjerning, ikke en ny evaluering av modeller trent uten bevegelse.

| Informasjon fjernet | GNN: absolutt endring | Uten meldinger | Bare ball |
|---|---:|---:|---:|
{motion_rows}

## Hvor på banen oppstår høye verdier?

![Observerte verdier etter ballposisjon](diagnostics_ball_location.png)

Besittelseslaget angriper mot høyre i disse kartene. Rutene viser gjennomsnitt over faktisk observerte situasjoner med ballen der, ikke hva som ville skjedd hvis ballen ble flyttet dit. Minst 20 snapshots kreves per rute; blanke ruter mangler nok observasjoner. GNN og realisert target deler fargeskala, mens forskjellen mot ballmodellen har egen skala. Banen er normalisert for framvisning; følsomhetsanalysen bruker faktisk banelengde og bredde.

## Sekvenser og konkrete eksempler

![Utvalgte snapshots](diagnostics_cases.png)

Seks utvalgskriterier dekker stor framtidig egen-/motstandersjanse, høy prediksjon uten skudd, modelluenighet og feil nær medianen med/uten skudd. Klipp fra samme kamp holdes atskilt. Utvalget bruker etterpåinformasjon og er tydelig merket; det er ikke et tilfeldig representativt utvalg.

I sekvensviseren brukes **hjemmelagets perspektiv gjennom hele klippet**. Hjemmelaget angriper mot høyre. Prediksjon og target skifter fortegn når besittelsen er hos bortelaget, slik at de fortsatt er sammenlignbare fra samme lags perspektiv. Banen roterer ikke ved et besittelsesbytte. Kvadrater markerer keepere; ringer viser lokal følsomhet for valgt modell. Spillernummer kommer fra kampmetadata. Bare observerte noder vises.

Tidslinjen viser observerte snapshots, vanligvis to sekunder fra hverandre. Hull bryter kurvene, og spilleren interpolerer ikke manglende tracking. Den stiplede targetkurven er realisert første-skudd-xG i den framtidige horisonten, ikke en observert sann EPV-kurve eller en live tilgjengelig input.

## Hva analysen kan brukes til

Undersøk om modellene gir forståelige tidsforløp og hvor de er uenige. Sammenlign keeper-/utespillerfølsomhet, bevegelsesinformasjon og ballposisjonsmønstre. Identifiser scener for videre rådatakontroll. Test nye hypoteser på nye kamper; disse figurene er utforskende og skal ikke brukes som et nytt urørt testsett.

Læreren bruker kontekstuell xG for 413 av 526 skudd og geometrisk reserveverdi for 113. Skudddekning, overført målkalibrering og kausale handlingseffekter er fortsatt uavklart. [Maskinlesbar rapport med sjekksummer](report.json).

Tracking og hendelser: SkillCorner Open Data. xG-treningsgrunnlag: StatsBomb Open Data. Prosjektets separate datavilkår og attribusjon gjelder også disse avledede visualiseringene. Sekvensviseren inneholder utvalgte spillerposisjoner og holdes i den lokale output-mappen.
'''
    (out / 'report.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='outputs/contextual_epv_graphs_v1')
    parser.add_argument('--runs', default='outputs/contextual_epv_v1')
    parser.add_argument('--root', default='data/skillcorner')
    parser.add_argument('--out', default='outputs/contextual_diagnostics_v1')
    parser.add_argument('--template', default='assets/epv_explorer.html')
    parser.add_argument('--sample-size', type=int, default=96)
    a = parser.parse_args()
    run(a.data, a.runs, a.root, a.out, a.template, a.sample_size)


if __name__ == '__main__':
    main()
