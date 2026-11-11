import hashlib
import io
import json

import numpy as np
import pytest

from open_graph_epv import source
from open_graph_epv.data import FEATURES
from open_graph_epv.training import train, load_epv


def test_lfs_download_verification_and_modified_cache(tmp_path, monkeypatch):
    body = b'{"frame": 1}\n'
    checksum = hashlib.sha256(body).hexdigest()
    pointer = f'version https://git-lfs.github.com/spec/v1\noid sha256:{checksum}\nsize {len(body)}\n'.encode()
    monkeypatch.setattr(source, 'urlopen', lambda url, **kw: io.BytesIO(body if url.startswith(source.MEDIA) else pointer))
    path = source.fetch(tmp_path, 'data/test.jsonl')
    assert path.read_bytes() == body
    manifest = json.loads((tmp_path / 'download_manifest.json').read_text())
    assert manifest['files']['data/test.jsonl']['sha256'] == checksum
    path.write_text('modified')
    with pytest.raises(ValueError, match='Unverified'):
        source.fetch(tmp_path, 'data/test.jsonl')


def test_lfs_rejects_bad_hash(tmp_path, monkeypatch):
    pointer = b'version https://git-lfs.github.com/spec/v1\noid sha256:invalid\nsize 3\n'
    monkeypatch.setattr(source, 'urlopen', lambda url, **kw: io.BytesIO(b'abc' if url.startswith(source.MEDIA) else pointer))
    with pytest.raises(ValueError, match='verification'):
        source.fetch(tmp_path, 'data/test.jsonl')
    assert not (tmp_path / 'data/test.jsonl').exists()


def test_training_smoke_does_not_evaluate_test_targets(tmp_path):
    # Synthetic fixture only. NaN test targets would break evaluation if read.
    data = tmp_path / 'prepared'
    data.mkdir()
    x = np.zeros((12, 23, len(FEATURES)), dtype=np.float32)
    mask = np.zeros((12, 23), dtype=bool)
    mask[:, :3] = True
    x[:, 0, 6] = 1
    x[:, 1, 4] = 1
    x[:, 2, 5] = 1
    x[:, 1, 0] = .2
    x[:, 2, 0] = -.2
    y = np.array([.1, 0, -.2, 0] * 2 + [float('nan')] * 4, dtype=np.float32)
    np.savez_compressed(data / 'graphs.npz', x=x, mask=mask, y=y, match_id=np.repeat([1, 2, 3], 4))
    (data / 'manifest.json').write_text(json.dumps({'graphs_sha256': source.digest(data / 'graphs.npz')}))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': [1], 'val': [2], 'test': [3]}))
    result = train(data, split, tmp_path / 'run', epochs=1)
    assert result['test_evaluated'] is False
    assert np.isfinite(result['history'][0]['rmse'])
    assert (tmp_path / 'run/model.pt').exists()
    model = load_epv(tmp_path / 'run')
    import torch
    with torch.no_grad():
        assert torch.isfinite(model.symmetric(torch.from_numpy(x[:2]), torch.from_numpy(mask[:2]))).all()
    with (tmp_path / 'run/model.pt').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_epv(tmp_path / 'run')
