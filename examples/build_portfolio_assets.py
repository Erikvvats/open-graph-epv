"""Export a small, public-source-only portfolio demo and auditable figures.

This reads frozen outputs. It does not fit models or copy checkpoints into the
publication tree. Every included scene comes from pinned SkillCorner files.
"""
import csv
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from open_graph_epv.contextual_epv import verify_data
from open_graph_epv.source import digest, match_path, REVISION
from open_graph_epv.team_epv import load_bundle, source_hashes, FIELDS
from open_graph_epv.xg import SB_REVISION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'outputs/contextual_epv_graphs_v1'
    runs = root/'outputs/team_epv_v1'
    visual = root/'outputs/team_epv_visualizations_v1'
    raw = root/'data/skillcorner'
    demo = root/'demo';demo.mkdir(exist_ok=True)
    figures = root/'reports/figures';figures.mkdir(exist_ok=True)
    original = verify_data(base)
    selection = json.loads((runs/'selection.json').read_text())
    audit = json.loads((runs/'audit/report.json').read_text())
    if (selection['source_sha256'] != source_hashes() or audit['selection_sha256'] != digest(runs/'selection.json')
            or audit['predictions_sha256'] != digest(runs/'audit/predictions.npz')):
        raise ValueError('Frozen experiment changed')
    bundle = load_bundle(runs/'bundle')
    visual_check = json.loads((visual/'verification.json').read_text())
    if visual_check['audit_sha256'] != digest(runs/'audit/report.json'):
        raise ValueError('Viewer uses a different experiment')
    html = (visual/'explorer.html').read_text()
    payload = json.loads(re.search(r'<script id="epv-data" type="application/json">(.*?)</script>', html, re.S)[1])
    titles = ['High upcoming shot value', 'Upcoming opposition chance', 'High value without a shot',
              'Model disagreement', 'Typical shot-window error', 'Typical no-shot error']
    for case, title in zip(payload['cases'], titles):
        case['title'] = title
        case['selection'] = 'Selected using future targets or derived net errors; an illustrative example, not a representative performance sample.'
    (demo/'predictions.js').write_text('window.EPV_DEMO = '+json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')+';\n')

    # One training example only: the notebook's fitting illustration never uses
    # validation or test labels, and saves no fitted weights.
    split = selection['protocol']['split']
    with np.load(base/'graphs.npz', allow_pickle=False) as z:
        with np.load(base/'target_provenance.npz', allow_pickle=False) as provenance:
            eligible = np.isin(z['match_id'], split['train']) & (z['y'] > 0) & (provenance['kind'] == 1)
            i = int(np.flatnonzero(eligible)[0])
        mid, frame_number = int(z['match_id'][i]), int(z['frame'][i])
        expected_x, expected_mask = z['x'][i], z['mask'][i]
    source_files = {}
    # Check all raw files from which demo positions may have been derived.
    ids = sorted({mid} | {int(c['match_id']) for c in payload['cases']})
    for game in ids:
        for suffix in ('match.json', 'tracking_extrapolated.jsonl', 'dynamic_events.csv'):
            path = match_path(raw, game, suffix)
            relative = str(path.relative_to(raw))
            checksum = digest(path)
            if original['inputs'][relative] != checksum:
                raise ValueError('Changed public raw input')
            source_files[relative] = checksum
    meta = json.loads(match_path(raw, mid, 'match.json').read_text())
    previous = None
    with match_path(raw, mid, 'tracking_extrapolated.jsonl').open() as handle:
        for line in handle:
            frame = json.loads(line)
            if frame['frame'] == frame_number:
                break
            previous = frame
        else:
            raise ValueError('Training example frame missing')
    # Retain only fields used by frame_graph; omit personal metadata.
    metadata = {k: meta[k] for k in ('home_team', 'away_team', 'pitch_length', 'pitch_width', 'home_team_side', 'match_periods')}
    metadata['home_team'] = {k: meta['home_team'][k] for k in ('id', 'name')}
    metadata['away_team'] = {k: meta['away_team'][k] for k in ('id', 'name')}
    metadata['players'] = [{'id': p['id'], 'team_id': p['team_id'], 'player_role': {'acronym': p['player_role']['acronym']}} for p in meta['players']]
    def small_frame(value):
        fields = ('x', 'y', 'is_detected')
        return {'frame': value['frame'], 'period': value['period'],
            'possession': {'group': (value.get('possession') or {}).get('group')},
            'ball_data': {k: value['ball_data'].get(k) for k in fields},
            'player_data': [{k: p.get(k) for k in ('player_id', *fields)} for p in value['player_data']]}
    frame, previous = small_frame(frame), small_frame(previous)
    from open_graph_epv.data import frame_graph
    x, mask, _ = frame_graph(frame, previous, metadata)
    if not np.array_equal(x, expected_x) or not np.array_equal(mask, expected_mask):
        raise ValueError('Reduced demo example changes model features')
    labels_path = root/'outputs/skillcorner_contextual_xg_v1/shot_xg.csv'
    if digest(labels_path) != original['xg_sha256']:
        raise ValueError('Changed teacher labels')
    with labels_path.open() as handle:
        shots = [{'period': int(r['period']), 'frame': int(r['frame']), 'team_id': int(r['team_id']),
                  'xg': float(r['xg']), 'source': r['xg_source']} for r in csv.DictReader(handle) if int(r['match_id']) == mid]
    # Only the local horizon is necessary to demonstrate first-shot labeling.
    shots = [s for s in shots if s['period'] == frame['period'] and frame_number < s['frame'] <= frame_number+150]
    example = {'match_id': mid, 'split': 'train', 'metadata': metadata, 'current': frame, 'previous': previous,
        'future_shots': shots, 'recorded_prediction': bundle.score_frame(frame, previous, metadata),
        'note': 'Small extract of pinned SkillCorner Open Data. xG and recorded predictions are derived from StatsBomb-trained models; no model weights included.'}
    write_json(demo/'example.json', example)

    with np.load(runs/'audit/predictions.npz', allow_pickle=False) as z:
        y = np.stack([z[f'{field}_target'] for field in FIELDS], -1)
        pred = np.stack([z[f'gnn_{field}'] for field in FIELDS], -1)
        mid_test = z['match_id']
        with np.load(base/'target_provenance.npz', allow_pickle=False) as prov:
            use = np.isin(prov['match_id'], split['test'])
            if not np.array_equal(prov['match_id'][use], mid_test) or not np.array_equal(prov['frame'][use], z['frame']):
                raise ValueError('Shot error grouping alignment changed')
            shot_frame = prov['future_shot_frame'][use]
    error = np.square(y.astype(float)-pred).sum(1)
    shot_rows = []
    for game, shot in sorted(set(zip(mid_test[shot_frame >= 0], shot_frame[shot_frame >= 0]))):
        use = (mid_test == game) & (shot_frame == shot)
        shot_rows.append({'match_id': int(game), 'shot_frame': int(shot), 'n_windows': int(use.sum()),
            'xg': float(y[use].sum(1)[0]), 'squared_error_share': float(error[use].sum()/error.sum())})
    shot_rows.sort(key=lambda r: -r['squared_error_share'])
    transfer = json.loads((root/'outputs/skillcorner_contextual_xg_v1/shot_audit.json').read_text())
    largest = shot_rows[0]
    label = next(r for r in transfer if r['match_id'] == largest['match_id'] and r['frame'] == largest['shot_frame'])
    result = {'target': audit['target'], 'selected_family': selection['selected_family'],
        'validation': selection['validation'], 'test': audit['metrics'], 'prior': audit['training_prior'],
        'n_test_snapshots': len(y), 'n_test_matches': 3, 'n_distinct_future_shots': len(shot_rows),
        'ranked_shot_errors': shot_rows, 'no_shot_error_share': float(error[shot_frame < 0].sum()/error.sum()),
        'largest_shot_teacher_audit': label,
        'status': 'Exploratory test on previously examined matches; no causal or native SkillCorner superiority claim.'}
    write_json(root/'reports/portfolio_results.json', result)

    families = ['prior', 'ball', 'set', 'gnn']
    values = [audit['training_prior']['component_rmse']] + [audit['metrics'][f]['component_rmse'] for f in families[1:]]
    fig, ax = plt.subplots(figsize=(8, 4.4), layout='constrained')
    ax.bar(range(4), values, color=['#93a6ac', '#d47727', '#8c59a0', '#167c80'])
    for j, v in enumerate(values):ax.text(j, v+.0005, f'{v:.5f}', ha='center')
    ax.set(xticks=range(4), xticklabels=['Training prior', 'Ball only', 'Players, no messages*', 'GNN'],
           ylim=(0, .04), ylabel='Two-team test RMSE (lower is better)',
           title='Player context adds a small improvement on this dataset')
    fig.text(.5, .005, '*Selected on validation. Three seeds per family; 3 previously examined test matches.', ha='center', fontsize=9)
    fig.savefig(figures/'portfolio_results.png', dpi=170);plt.close(fig)

    case = payload['cases'][0];state = case['states'][case['anchor_position']]
    fig, (ax, trace) = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    length, width = case['pitch_length'], case['pitch_width']
    ax.set_facecolor('#edf5ef');ax.add_patch(Rectangle((-length/2, -width/2), length, width, fill=False, color='#779187'))
    ax.plot([0, 0], [-width/2, width/2], color='#779187');ax.add_patch(Circle((0, 0), 9.15, fill=False, color='#779187'))
    for side in (-1, 1):ax.add_patch(Rectangle((-length/2 if side == -1 else length/2-16.5, -20.16), 16.5, 40.32, fill=False, color='#779187'))
    for point in sorted(state['points'], key=lambda p: p['ball']):
        ax.scatter(point['x'], point['y'], s=45 if point['ball'] else 65,
            color='#e7b12d' if point['ball'] else '#178caa' if point['home'] else '#e55f58',
            marker='s' if point['keeper'] else 'o', edgecolor='white', linewidth=.7)
    ax.set(xlim=(-length/2-2, length/2+2), ylim=(-width/2-2, width/2+2), aspect='equal', xticks=[], yticks=[],
        title=f"{case['home']} – {case['away']}\nHome attacks right; observed players only")
    family = selection['selected_family']
    for key, color, name in zip(FIELDS, ('#178caa', '#e55f58'), ('Home EPV', 'Away EPV')):
        times, predicted = [], []
        for j, s in enumerate(case['states']):
            if j and s['frame']-case['states'][j-1]['frame'] > 20:
                times.append(np.nan);predicted.append(np.nan)
            times.append(s['seconds']);predicted.append(s['values'][family][key])
        trace.plot(times, predicted, 'o-', color=color, label=name, ms=4)
    trace.axvline(0, color='#789', ls=':')
    trace.set(xlabel='Seconds relative to illustrated snapshot', ylabel='Expected first-shot xG', ylim=(0, None),
        title=f"Selected no-message model at anchor\nHome {state['values'][family][FIELDS[0]]:.4f} · Away {state['values'][family][FIELDS[1]]:.4f}")
    trace.legend();fig.suptitle('Two team values, one situation — illustrative hindsight-selected example')
    fig.savefig(figures/'portfolio_scene.png', dpi=170);plt.close(fig)

    shares = [largest['squared_error_share'], sum(r['squared_error_share'] for r in shot_rows[1:5]),
              sum(r['squared_error_share'] for r in shot_rows[5:]), result['no_shot_error_share']]
    fig, ax = plt.subplots(figsize=(8, 4.1), layout='constrained')
    ax.barh(range(4), np.array(shares)*100, color=['#b54b4b', '#d68c62', '#6c9f9a', '#93a6ac'])
    for j, v in enumerate(shares):ax.text(v*100+.7, j, f'{100*v:.1f}%', va='center')
    ax.set(yticks=range(4), yticklabels=[f"One shot ({largest['n_windows']} windows)", 'Next four shots', 'Remaining shots', 'No shot in horizon'],
        xlabel='Share of total two-component squared error (%)', xlim=(0, max(shares)*100+10),
        title='Error concentration limits confidence in the headline score')
    ax.invert_yaxis();fig.savefig(figures/'portfolio_error_concentration.png', dpi=170);plt.close(fig)

    third_party = root/'third_party';third_party.mkdir(exist_ok=True)
    shutil.copyfile(raw/'LICENSE', third_party/'SkillCorner-LICENSE.txt')
    shutil.copyfile(root/'data/statsbomb/LICENSE.pdf', third_party/'StatsBomb-LICENSE.pdf')
    provenance = {'date': '2026-11-10', 'scope': 'Only this standalone open-data project; no parent-project imports, data or checkpoints.',
        'sources': {'skillcorner': {'repository': 'https://github.com/SkillCorner/opendata', 'revision': REVISION},
                    'statsbomb': {'repository': 'https://github.com/hudl/open-data', 'revision': SB_REVISION}},
        'skillcorner_raw_file_sha256': source_files,
        'team_selection_sha256': digest(runs/'selection.json'), 'team_audit_sha256': digest(runs/'audit/report.json'),
        'xg_labels_sha256': digest(labels_path), 'builder_sha256': digest(Path(__file__)),
        'assets': {str(p.relative_to(root)): digest(p) for p in [demo/'example.json', demo/'predictions.js',
            root/'reports/portfolio_results.json', *sorted(figures.glob('portfolio_*.png'))]},
        'model_weights_distributed': False, 'example_training_match_id': mid,
        'checks': {'example_features_match_frozen_training_graph': True, 'raw_sources_match_pinned_dataset': True,
                   'frozen_bundle_verified': True}}
    write_json(root/'reports/public_provenance.json', provenance)
    print('Built public demo data, three figures, current-model error audit and provenance.')


if __name__ == '__main__':
    main()
