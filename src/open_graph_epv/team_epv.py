"""Train, evaluate and deploy home_team_epv and away_team_epv separately."""
import argparse
import copy
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

from .contextual_epv import verify_data as verify_base
from .data import FEATURES, frame_graph
from .evaluation import paired_match_difference
from .source import digest, match_path
from .training import predict, validate_split
from .team_epv_model import TeamGraphEPV, TeamBallEPV

FIELDS = ['home_team_epv', 'away_team_epv']
TARGET = 'Two team-specific expected contextual xG values for the first shot in (t,t+15s], with explicit geometry fallback'
SOURCES = ('team_epv.py', 'team_epv_model.py', 'model.py', 'data.py', 'training.py', 'evaluation.py')


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCES}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write('\n')


def relative_targets(signed):
    signed = np.asarray(signed)
    if not np.isfinite(signed).all() or (abs(signed) > 1).any():
        raise ValueError('Invalid signed shot label')
    return np.stack([np.maximum(signed, 0), np.maximum(-signed, 0)], axis=-1)


def to_home_away(values, possession_is_home):
    """Swap team channels, never negate values. This operation is its own inverse."""
    values = np.asarray(values)
    home = np.asarray(possession_is_home, bool)
    if values.shape[-1] != 2 or values.shape[:-1] != home.shape:
        raise ValueError('Expected [...,2] values and matching possession flags')
    return np.where(home[..., None], values, values[..., ::-1])


def prepare(base, root, split_path, output):
    base, root, output = map(Path, (base, root, output))
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_base(base)
    split = json.loads(Path(split_path).read_text())
    with np.load(base / 'graphs.npz', allow_pickle=False) as z:
        mids, frames, signed = z['match_id'], z['frame'], z['y']
    validate_split(split, np.unique(mids))
    home = np.zeros(len(mids), bool)
    found = np.zeros(len(mids), bool)
    teams = {}
    for mid in np.unique(mids):
        mid = int(mid)
        for suffix in ('match.json', 'tracking_extrapolated.jsonl'):
            path = match_path(root, mid, suffix)
            if digest(path) != manifest['inputs'][str(path.relative_to(root))]:
                raise ValueError('Changed source for team identity')
        meta = json.loads(match_path(root, mid, 'match.json').read_text())
        teams[str(mid)] = {'home_team_id': meta['home_team']['id'], 'away_team_id': meta['away_team']['id']}
        indices = np.flatnonzero(mids == mid)
        lookup = {int(frames[i]): int(i) for i in indices}
        with match_path(root, mid, 'tracking_extrapolated.jsonl').open() as handle:
            for line in handle:
                frame = json.loads(line)
                i = lookup.get(frame['frame'])
                if i is None:
                    continue
                group = (frame.get('possession') or {}).get('group')
                if group not in ('home team', 'away team') or found[i]:
                    raise ValueError('Missing or duplicate possession identity')
                home[i] = group == 'home team'
                found[i] = True
        print(f'{mid}: mapped {int(found[indices].sum())} team labels', flush=True)
    if not found.all():
        raise ValueError('Prepared frames lack raw team identity')
    labels = to_home_away(relative_targets(signed), home)
    if not np.array_equal((labels[:, 0] - labels[:, 1]) * np.where(home, 1, -1), signed):
        raise ValueError('Team targets do not reconstruct the old signed target')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'team_targets.npz', home_team_epv=labels[:, 0], away_team_epv=labels[:, 1],
                        possession_is_home=home, match_id=mids, frame=frames)
    result = {'target': TARGET, 'fields': FIELDS, 'base_graphs_sha256': manifest['graphs_sha256'],
        'base_manifest_sha256': digest(base / 'manifest.json'), 'targets_sha256': digest(output / 'team_targets.npz'),
        'source_sha256': source_hashes(), 'split': split, 'n_graphs': len(mids), 'teams': teams,
        'teacher_coverage': manifest['teacher_coverage'], 'horizon_seconds': 15,
        'interpretation': 'First shot overall, not first shot per team. Both observed labels are zero if no shot occurs. Predicted expectations may both be positive.'}
    save(output / 'manifest.json', result)
    protocol = {'date': '2026-11-10', 'target': TARGET, 'fields': FIELDS, 'families': ['gnn', 'ball', 'set'],
        'seeds': [0, 1, 2], 'max_epochs': 30, 'patience': 8, 'batch_size': 64, 'learning_rate': .001,
        'weight_decay': .001, 'loss': 'Mean of squared errors across both team components',
        'selection': 'Lowest validation two-component MSE of equal-weight three-seed ensemble',
        'split': split, 'manifest_sha256': digest(output / 'manifest.json'), 'source_sha256': source_hashes(),
        'test_status': 'Three previously examined test matches; exploratory extension, not fresh final evaluation',
        'output_constraint': 'Nonnegative components, sum <= 1; softmax third mass has no no-shot interpretation'}
    save(output / 'protocol.json', protocol)
    return result


def load_data(base, data):
    base, data = Path(base), Path(data)
    original = verify_base(base)
    manifest = json.loads((data / 'manifest.json').read_text())
    protocol = json.loads((data / 'protocol.json').read_text())
    if (manifest['target'] != TARGET or manifest['source_sha256'] != source_hashes()
            or manifest['base_graphs_sha256'] != original['graphs_sha256']
            or manifest['base_manifest_sha256'] != digest(base / 'manifest.json')
            or manifest['targets_sha256'] != digest(data / 'team_targets.npz')
            or protocol['manifest_sha256'] != digest(data / 'manifest.json')
            or protocol['source_sha256'] != source_hashes()):
        raise ValueError('Team data/protocol/source changed')
    with np.load(base / 'graphs.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(data / 'team_targets.npz', allow_pickle=False) as z:
        for key in ('match_id', 'frame'):
            if not np.array_equal(arrays[key], z[key]):
                raise ValueError('Team label row alignment changed')
        arrays['home_away'] = np.stack([z[k] for k in FIELDS], -1)
        arrays['possession_is_home'] = z['possession_is_home']
    arrays['relative'] = to_home_away(arrays['home_away'], arrays['possession_is_home'])
    return arrays, manifest, protocol


def make_model(family, config=None):
    if family == 'ball':
        return TeamBallEPV(**(config or {}))
    if family not in ('gnn', 'set'):
        raise ValueError('Unknown model family')
    return TeamGraphEPV(**(config or {'edges_enabled': family == 'gnn'}))


def train(base, data, family, seed, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    arrays, manifest, p = load_data(base, data)
    if family not in p['families'] or seed not in p['seeds']:
        raise ValueError('Unregistered family/seed')
    tr, va = [np.isin(arrays['match_id'], p['split'][k]) for k in ('train', 'val')]
    x, mask, y = [torch.from_numpy(arrays[k][tr]) for k in ('x', 'mask', 'relative')]
    xv, mv = [torch.from_numpy(arrays[k][va]) for k in ('x', 'mask')]
    yv = arrays['relative'][va]
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    model = make_model(family)
    prior = y.mean(0)
    if (prior <= 0).any() or prior.sum() >= 1:
        raise ValueError('Need positive training support for both components')
    with torch.no_grad():
        model.head[-1].weight.mul_(.01)
        model.head[-1].bias.copy_(torch.cat([prior, 1-prior.sum().reshape(1)]).log())
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(p['max_epochs']):
        model.train()
        for indices in torch.randperm(len(x)).split(p['batch_size']):
            batch = x[indices].clone()
            sign = torch.where(torch.rand(len(batch)) < .5, -1., 1.)[:, None]
            batch[..., 1] *= sign;batch[..., 3] *= sign
            loss = (model(batch, mask[indices]) - y[indices]).square().mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        values = predict(model, xv, mv)
        mse = float(np.square(yv-values).mean())
        history.append({'epoch': epoch+1, 'component_mse': mse, 'component_rmse': mse**.5})
        print(f'epoch={epoch+1} val_component_rmse={mse**.5:.6f}', flush=True)
        if mse < best:
            best, best_state, stale = mse, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= p['patience']:
            break
    output.mkdir(parents=True)
    torch.save({'state_dict': best_state, 'config': model.config, 'family': family, 'schema': 'team-epv-v1'}, output/'model.pt')
    result = {'target': TARGET, 'family': family, 'seed': seed, 'config': model.config,
        'n_parameters': sum(t.numel() for t in model.parameters()), 'history': history,
        'best_validation': min(history, key=lambda r: r['component_mse']), 'training_prior_for_against': prior.tolist(),
        'protocol_sha256': digest(Path(data)/'protocol.json'), 'source_sha256': source_hashes(),
        'model_sha256': digest(output/'model.pt'), 'test_evaluated': False,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__}}
    save(output/'training.json', result)
    return result


def load_model(directory):
    directory = Path(directory)
    report = json.loads((directory/'training.json').read_text())
    if report['target'] != TARGET or report['source_sha256'] != source_hashes() or report['model_sha256'] != digest(directory/'model.pt'):
        raise ValueError('Team model provenance changed')
    checkpoint = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
    if checkpoint['schema'] != 'team-epv-v1' or checkpoint['family'] != report['family']:
        raise ValueError('Wrong model identity')
    model = make_model(checkpoint['family'], checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def ensemble(models, x, mask):
    return np.mean([predict(m, torch.from_numpy(x), torch.from_numpy(mask)) for m in models], axis=0)


def team_metrics(y, values):
    delta = np.square(np.asarray(y, float) - values)
    return {'n': len(y), 'component_mse': float(delta.mean()), 'component_rmse': float(np.sqrt(delta.mean())),
        'home_team_rmse': float(np.sqrt(delta[:, 0].mean())), 'away_team_rmse': float(np.sqrt(delta[:, 1].mean())),
        'home_net_rmse': float(np.sqrt(np.square((y[:, 0]-y[:, 1])-(values[:, 0]-values[:, 1])).mean())),
        'predicted_home_mean': float(values[:, 0].mean()), 'predicted_away_mean': float(values[:, 1].mean()),
        'target_home_mean': float(y[:, 0].mean()), 'target_away_mean': float(y[:, 1].mean())}


def select_and_evaluate(base, data, runs, legacy=None):
    runs, data = Path(runs), Path(data)
    if (runs/'selection.json').exists() or (runs/'audit').exists():
        raise FileExistsError('Selection/audit already exists')
    a, manifest, p = load_data(base, data)
    torch.set_num_threads(2)
    va = np.isin(a['match_id'], p['split']['val'])
    records, models, validation = [], {}, {}
    prior = None
    for family in p['families']:
        models[family] = []
        for seed in p['seeds']:
            name = f'{family}_seed{seed}';directory = runs/name
            r = json.loads((directory/'training.json').read_text())
            if r['protocol_sha256'] != digest(data/'protocol.json') or r['seed'] != seed or r['family'] != family:
                raise ValueError('Run differs from protocol')
            if prior is not None and prior != r['training_prior_for_against']:
                raise ValueError('Training prior mismatch')
            prior = r['training_prior_for_against']
            models[family].append(load_model(directory))
            records.append({'name': name, 'family': family, 'seed': seed, 'model_sha256': digest(directory/'model.pt'),
                            'report_sha256': digest(directory/'training.json')})
        relative = ensemble(models[family], a['x'][va], a['mask'][va])
        validation[family] = team_metrics(a['home_away'][va], to_home_away(relative, a['possession_is_home'][va]))
    winner = min(p['families'], key=lambda f: validation[f]['component_mse'])
    selection = {'selected_family': winner, 'validation': validation, 'models': records, 'protocol': p,
        'protocol_sha256': digest(data/'protocol.json'), 'source_sha256': source_hashes(), 'fields': FIELDS,
        'target': TARGET, 'training_prior_for_against': prior, 'test_evaluated': False}
    save(runs/'selection.json', selection)  # Freeze before any test predictions.
    te = np.isin(a['match_id'], p['split']['test'])
    y, home = a['home_away'][te], a['possession_is_home'][te]
    predicted = {f: to_home_away(ensemble(models[f], a['x'][te], a['mask'][te]), home) for f in p['families']}
    baseline = to_home_away(np.tile(prior, (len(y), 1)), home)
    metrics = {f: team_metrics(y, v) for f, v in predicted.items()}
    report = {'target': TARGET, 'fields': FIELDS, 'selected_family': winner, 'n_test_matches': len(p['split']['test']),
        'metrics': metrics, 'training_prior': team_metrics(y, baseline), 'teacher_coverage': manifest['teacher_coverage'],
        'selection_sha256': digest(runs/'selection.json'), 'source_sha256': source_hashes(),
        'legacy_net_comparison': None,
        'limitations': ['Previously examined three test matches; exploratory evaluation.',
            'First shot overall within 15 seconds; not each team first shot or at-least-one-goal probability.',
            'Both predictions may be positive; their sum is expected first-shot xG, not shot probability.',
            'Teacher transfer calibration and causal action effects remain unverified.']}
    if legacy:
        legacy = Path(legacy)
        legacy_report = json.loads((legacy/'report.json').read_text())
        if digest(legacy/'predictions.npz') != legacy_report['predictions_sha256']:
            raise ValueError('Changed legacy predictions')
        with np.load(legacy/'predictions.npz', allow_pickle=False) as z:
            if not np.array_equal(z['match_id'], a['match_id'][te]) or not np.array_equal(z['frame'], a['frame'][te]) or not np.array_equal(z['target'], a['y'][te]):
                raise ValueError('Legacy rows/targets mismatch')
            net = (predicted['gnn'][:, 0]-predicted['gnn'][:, 1])*np.where(home, 1, -1)
            report['legacy_net_comparison'] = paired_match_difference(a['y'][te], net, z['gnn'], a['match_id'][te])
            report['legacy_gnn_net_rmse'] = float(np.sqrt(np.square(z['target']-z['gnn']).mean()))
    audit = runs/'audit';audit.mkdir()
    np.savez_compressed(audit/'predictions.npz', match_id=a['match_id'][te], frame=a['frame'][te],
        possession_is_home=home, home_team_epv_target=y[:, 0], away_team_epv_target=y[:, 1],
        **{f'{f}_{field}': v[:, j] for f, v in predicted.items() for j, field in enumerate(FIELDS)})
    report['predictions_sha256'] = digest(audit/'predictions.npz')
    save(audit/'report.json', report)
    return report


def export_bundle(data, runs, output):
    data, runs, output = map(Path, (data, runs, output))
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads((runs/'selection.json').read_text())
    audit = json.loads((runs/'audit/report.json').read_text())
    manifest = json.loads((data/'manifest.json').read_text())
    if selection['source_sha256'] != source_hashes() or audit['selection_sha256'] != digest(runs/'selection.json'):
        raise ValueError('Selection/audit changed')
    if selection['protocol']['manifest_sha256'] != digest(data/'manifest.json'):
        raise ValueError('Wrong team data')
    families = sorted({'gnn', selection['selected_family']})
    output.mkdir(parents=True)
    for record in selection['models']:
        directory = runs/record['name']
        for name, key in (('model.pt', 'model_sha256'), ('training.json', 'report_sha256')):
            if digest(directory/name) != record[key]:
                raise ValueError('Frozen run changed')
        if record['family'] in families:
            shutil.copytree(directory, output/'models'/record['name'])
    for path, name in ((runs/'selection.json', 'selection.json'), (runs/'audit/report.json', 'audit.json'),
                       (data/'manifest.json', 'dataset_manifest.json'), (data/'protocol.json', 'protocol.json')):
        shutil.copyfile(path, output/name)
    sources = output/'source/open_graph_epv';sources.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, sources/path.name)
    (output/'NOTICE.txt').write_text('Two nonnegative expected first-shot xG values within 15 seconds: home_team_epv and away_team_epv.\n'
        'StatsBomb-trained contextual xG on SkillCorner Open Data; 413 contextual shots and 113 geometry fallbacks.\n'
        'Code MIT; separate data/derived-analysis terms and attribution still apply.\n'
        'https://github.com/hudl/open-data/blob/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/LICENSE.pdf\n'
        'https://github.com/SkillCorner/opendata\n')
    result = {'schema': 'team-epv-v1', 'target': TARGET, 'fields': FIELDS, 'features': FEATURES,
        'selected_family': selection['selected_family'], 'included_families': families,
        'teacher_coverage': manifest['teacher_coverage'], 'source_sha256': source_hashes(),
        'models': [r for r in selection['models'] if r['family'] in families],
        'files': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    save(output/'manifest.json', result)
    return result


class TeamEnsemble:
    def __init__(self, models, family):
        self.models, self.family = models, family

    @torch.no_grad()
    def score_frame(self, frame, previous, metadata):
        graph = frame_graph(frame, previous, metadata, include_extrapolated=False)
        if graph is None:
            raise ValueError('Snapshot lacks known possession or observed tracking')
        x, mask, team = graph
        relative = np.array([m.symmetric(torch.from_numpy(x)[None], torch.from_numpy(mask)[None])[0].numpy() for m in self.models])
        home = np.full(len(relative), team == metadata['home_team']['id'])
        values = to_home_away(relative, home)
        means = values.mean(0)
        return {'home_team_epv': float(means[0]), 'away_team_epv': float(means[1]),
            'home_team_id': metadata['home_team']['id'], 'away_team_id': metadata['away_team']['id'],
            'net_home_epv': float(means[0]-means[1]), 'family': self.family, 'frame': frame['frame'],
            'possession_team_id': team, 'horizon_seconds': 15, 'target': TARGET,
            'seed_std': dict(zip(FIELDS, values.std(0).tolist()))}


def load_bundle(directory, family=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema') != 'team-epv-v1' or manifest['fields'] != FIELDS or manifest['source_sha256'] != source_hashes():
        raise ValueError('Wrong team bundle or changed inference source')
    for path, checksum in manifest['files'].items():
        if digest(directory/path) != checksum:
            raise ValueError(f'Bundle changed: {path}')
    family = family or manifest['selected_family']
    if family not in manifest['included_families']:
        raise ValueError('Family not included in bundle')
    models = [load_model(directory/'models'/r['name']) for r in manifest['models'] if r['family'] == family]
    if len(models) != 3:
        raise ValueError('Expected three models')
    return TeamEnsemble(models, family)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'evaluate', 'export'):
        s = sub.add_parser(name)
        s.add_argument('--base', default='outputs/contextual_epv_graphs_v1')
        s.add_argument('--data', default='outputs/team_epv_data_v1')
        s.add_argument('--runs', default='outputs/team_epv_v1')
        if name == 'prepare':
            s.add_argument('--root', default='data/skillcorner');s.add_argument('--split', default='configs/split.json')
        elif name == 'train':
            s.add_argument('--family', choices=['gnn', 'ball', 'set'], required=True);s.add_argument('--seed', type=int, required=True)
        elif name == 'evaluate':
            s.add_argument('--legacy', default='outputs/contextual_epv_v1/audit')
        if name != 'evaluate':
            s.add_argument('--out', required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        r = prepare(a.base, a.root, a.split, a.out);print('Prepared two-team labels:', r['n_graphs'])
    elif a.command == 'train':
        train(a.base, a.data, a.family, a.seed, a.out)
    elif a.command == 'evaluate':
        r = select_and_evaluate(a.base, a.data, a.runs, a.legacy or None);print(json.dumps(r['metrics'], indent=2))
    else:
        r = export_bundle(a.data, a.runs, a.out);print('Exported', r['included_families'])


if __name__ == '__main__':
    main()
