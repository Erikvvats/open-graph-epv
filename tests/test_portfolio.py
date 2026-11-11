"""Publication boundaries and provenance: no network or raw training data needed."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('package_portfolio', ROOT/'examples/package_portfolio.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_public_demo_matches_verified_export():
    package.verify_assets(ROOT)
    example = json.loads((ROOT/'demo/example.json').read_text())
    assert example['split'] == 'train'
    assert example['match_id'] not in {2015213, 2016236, 2017461}


def test_publication_ignores_parent_data_weights_and_secrets(tmp_path):
    root = tmp_path/'standalone'
    root.mkdir()
    for name in package.ROOT_FILES:
        (root/name).write_text('example')
    for name in ('data', 'outputs', 'src', 'demo'):
        (root/name).mkdir()
    for relative in ('data/private.json', 'outputs/weights.pt', 'src/weights.pt', 'demo/.env'):
        (root/relative).write_text('must not ship')
    (tmp_path/'parent_private.csv').write_text('must not ship')
    (root/'src/public.py').write_text('pass')
    (root/'demo/example.json').write_text('{}')
    paths = {str(p.relative_to(root)) for p in package.collect_files(root)}
    assert paths == set(package.ROOT_FILES) | {'src/public.py', 'demo/example.json'}


def test_publication_rejects_symlink_to_parent(tmp_path):
    root = tmp_path/'standalone'
    (root/'demo').mkdir(parents=True)
    private = tmp_path/'private.json'
    private.write_text('{}')
    (root/'demo/leak.json').symlink_to(private)
    with pytest.raises(ValueError, match='Symlink'):
        package.collect_files(root)
