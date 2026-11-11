"""Portable inference bundle with checksums and explicit value semantics."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .data import FEATURES, frame_graph
from .evaluation import verify_selection
from .source import digest
from .training import load_epv


def export_bundle(data, runs, selection_path, audit_path, xg_directory, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(Path(selection_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit['selection_sha256'] != digest(selection_path) or audit['dataset_sha256'] != selection['dataset_sha256']:
        raise ValueError('Audit does not correspond to frozen selection')
    verify_selection(data, runs, selection)
    xg_directory = Path(xg_directory)
    xg_report = json.loads((xg_directory / 'report.json').read_text())
    if digest(xg_directory / 'model.json') != xg_report['model_sha256']:
        raise ValueError('xG model checksum mismatch')
    dataset_meta = json.loads((Path(data) / 'manifest.json').read_text())
    if (dataset_meta['features'] != FEATURES or dataset_meta['horizon_seconds'] != 15
            or dataset_meta['include_extrapolated']
            or dataset_meta['xg_sources'] != [f"statsbomb_geometry_v1:{xg_report['model_sha256']}"]):
        raise ValueError('Bundle input/target schema or xG source differs from training data')
    output.mkdir(parents=True)
    families = sorted({'gnn', selection['selected_family']})
    included = [r for r in selection['models'] if r['family'] in families]
    for record in included:
        target = output / 'models' / record['name']
        target.mkdir(parents=True)
        for name in ('model.pt', 'training.json'):
            shutil.copyfile(Path(runs) / record['name'] / name, target / name)
    shutil.copyfile(selection_path, output / 'selection.json')
    shutil.copyfile(audit_path, output / 'audit.json')
    for name in ('model.json', 'report.json'):
        shutil.copyfile(xg_directory / name, output / f'xg_{name}')
    shutil.copyfile(Path(data) / 'manifest.json', output / 'dataset_manifest.json')
    sources = output / 'source' / 'open_graph_epv'
    sources.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, sources / path.name)
    notice = ('Research artifact. Tracking: SkillCorner Open Data. xG training: StatsBomb Open Data.\n'
              'Code MIT license does not override data/derived-analysis terms. StatsBomb terms:\n'
              'https://github.com/hudl/open-data/blob/master/LICENSE.pdf\n'
              'Value: expected signed first-shot geometry-xG over 15 seconds, relative to current possession.\n'
              'Seed dispersion is NOT a calibrated confidence interval. No causal tactical validation.\n')
    (output / 'NOTICE.txt').write_text(notice)
    manifest = {'schema_version': 1, 'selected_family': selection['selected_family'], 'included_families': families,
                'models': included, 'features': FEATURES, 'horizon_seconds': 15, 'include_extrapolated': False,
                'runtime_source_sha256': {name: digest(Path(__file__).with_name(name)) for name in ('model.py', 'data.py')},
                'files': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


class EPVEnsemble:
    def __init__(self, models, family):
        self.models = models
        self.family = family

    @torch.no_grad()
    def score_frame(self, frame, previous, metadata):
        graph = frame_graph(frame, previous, metadata, include_extrapolated=False)
        if graph is None:
            raise ValueError('Snapshot lacks known possession, detected ball, or enough observed players')
        nodes, mask, team = graph
        x, m = torch.from_numpy(nodes)[None], torch.from_numpy(mask)[None]
        values = np.array([float(model.symmetric(x, m)[0]) for model in self.models])
        value = float(values.mean())
        return {'family': self.family, 'frame': frame['frame'], 'possession_team_id': team,
                'value': value, 'home_signed_value': value if team == metadata['home_team']['id'] else -value,
                'seed_std': float(values.std()), 'visible_players': int(mask.sum()) - 1,
                'horizon_seconds': 15, 'target': 'estimated signed future first-shot xG'}


def load_bundle(directory, family=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema_version'] != 1 or manifest['features'] != FEATURES:
        raise ValueError('Unsupported bundle feature schema')
    for name, checksum in manifest['runtime_source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Runtime feature/model implementation differs from bundle')
    for relative, checksum in manifest['files'].items():
        if digest(directory / relative) != checksum:
            raise ValueError(f'Bundle checksum mismatch: {relative}')
    family = family or manifest['selected_family']
    if family not in manifest['included_families']:
        raise ValueError('Requested family not included in bundle')
    models = [load_epv(directory / 'models' / r['name']) for r in manifest['models'] if r['family'] == family]
    if not models:
        raise ValueError('Empty ensemble')
    return EPVEnsemble(models, family)
