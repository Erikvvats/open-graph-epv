"""Package this standalone open-data project using an explicit source allowlist.

No parent workspace files, full datasets or model weights are included. The
small public demo extracts are deliberate and listed in public_provenance.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT_FILES = ('README.md', 'LICENSE', 'pyproject.toml', 'requirements.txt', '.gitignore')
DIRECTORIES = {
    'src': {'.py'}, 'tests': {'.py'}, 'configs': {'.json'},
    'examples': {'.py'}, 'reports': {'.md', '.json', '.csv', '.png', '.pdf'},
    'assets': {'.md', '.html', '.png'}, '.github': {'.yml'},
    'docs': {'.md'}, 'notebooks': {'.ipynb'},
    'demo': {'.html', '.js', '.json'}, 'third_party': {'.txt', '.pdf'},
}
SKIP = {'__pycache__', '.pytest_cache', 'build', 'dist', '.ipynb_checkpoints'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files(root):
    root = root.resolve()
    files = []
    for directory, suffixes in DIRECTORIES.items():
        base = root/directory
        if base.is_symlink():
            raise ValueError(f'Symlink is not allowed: {directory}')
        for path in base.rglob('*'):
            relative = path.relative_to(root)
            if any(p in SKIP or p.endswith('.egg-info') for p in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink is not allowed: {relative}')
            if path.is_file() and path.suffix in suffixes:
                files.append(path)
    files.extend(root/name for name in ROOT_FILES)
    for path in files:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError(f'File escapes standalone root: {path.name}')
        if path.stat().st_size > 5_000_000:
            raise ValueError(f'Unexpectedly large publication file: {path.name}')
    return sorted(files)


def verify_assets(root):
    provenance = json.loads((root/'reports/public_provenance.json').read_text())
    for relative, expected in provenance['assets'].items():
        path = root/relative
        if root.resolve() not in path.resolve().parents or sha256(path) != expected:
            raise ValueError(f'Public asset differs from verified export: {relative}')
    if sha256(root/'examples/build_portfolio_assets.py') != provenance['builder_sha256']:
        raise ValueError('Asset builder changed since provenance was recorded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.out)
    receipt = output.with_suffix('.manifest.json')
    if output.exists() or receipt.exists():
        raise FileExistsError('Choose a new archive path')
    verify_assets(root)
    files = collect_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = {str(p.relative_to(root)): sha256(p) for p in files}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path('open-graph-epv')/path.relative_to(root))
    manifest = {'archive': output.name, 'sha256': sha256(output),
        'bytes': output.stat().st_size, 'file_count': len(files), 'files': checksums,
        'scope': 'Standalone open-data source, reports and declared small public demo extracts',
        'contains_full_datasets': False, 'contains_model_weights': False,
        'contains_parent_workspace_files': False}
    receipt.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
