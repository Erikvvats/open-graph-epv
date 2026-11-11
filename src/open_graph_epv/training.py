"""Training with explicit match splits and an untouched test partition."""
import copy
import json
import platform
from pathlib import Path

import numpy as np
import torch

from .model import GraphEPV, BallEPV
from .source import digest


def validate_split(split, available):
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split requires exactly train, val and test lists")
    groups = {k: set(v) for k, v in split.items()}
    if any(not v or len(v) != len(split[k]) for k, v in groups.items()):
        raise ValueError("Every split must contain distinct matches")
    if groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test']:
        raise ValueError("Match leakage between splits")
    if set.union(*groups.values()) != set(available):
        raise ValueError("Split must partition exactly the prepared matches")


@torch.no_grad()
def predict(model, x, mask, batch_size=128):
    model.eval()
    return torch.cat([model.symmetric(x[i:i + batch_size], mask[i:i + batch_size])
                      for i in range(0, len(x), batch_size)]).numpy()


def metrics(y, prediction, mean):
    mse = float(np.mean((y - prediction) ** 2))
    baseline = float(np.mean((y - mean) ** 2))
    return {"n": len(y), "mse": mse, "rmse": mse ** .5, "baseline_rmse": baseline ** .5,
            "rmse_skill": 1 - (mse / baseline) ** .5 if baseline > 0 else None}


def load_epv(directory):
    """Load a local trained run and verify it against its recorded checksum."""
    directory = Path(directory)
    report = json.loads((directory / 'training.json').read_text())
    if digest(directory / 'model.pt') != report['model_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    kind = checkpoint['model_kind']
    if kind != report['model_kind'] or kind not in ('gnn', 'ball', 'set'):
        raise ValueError('Unknown or mismatched model kind')
    model = (BallEPV if kind == 'ball' else GraphEPV)(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def train(data, split_path, output, epochs=30, seed=0, batch_size=64, model_kind='gnn'):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(Path(split_path).read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    if digest(data / "graphs.npz") != manifest['graphs_sha256']:
        raise ValueError("Prepared dataset checksum changed")
    with np.load(data / "graphs.npz", allow_pickle=False) as z:
        validate_split(split, np.unique(z['match_id']))
        # Test features and targets are never evaluated during model selection.
        train_rows = np.isin(z['match_id'], split['train'])
        val_rows = np.isin(z['match_id'], split['val'])
        x, mask, y = [torch.from_numpy(z[k][train_rows].copy()) for k in ('x', 'mask', 'y')]
        xv, mv, yv = [z[k][val_rows].copy() for k in ('x', 'mask', 'y')]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not np.isfinite(xv).all() or not np.isfinite(yv).all():
        raise ValueError("Nonfinite model data")
    if not (y.abs() > 0).any():
        raise ValueError("No nonzero xG targets in training partition")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    if model_kind not in ('gnn', 'ball', 'set'):
        raise ValueError('Unknown model kind')
    model = BallEPV() if model_kind == 'ball' else GraphEPV(edges_enabled=model_kind == 'gnn')
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in ('model.py', 'training.py', 'data.py')}
    # Start near the training-only prior instead of arbitrary large EPV values.
    with torch.no_grad():
        model.head[-1].weight.mul_(.01)
        model.head[-1].bias.fill_(float(torch.atanh(y.mean().clamp(-.99, .99))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    xv, mv = torch.from_numpy(xv), torch.from_numpy(mv)
    mean = float(y.mean())
    history, best, stale, best_state = [], float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(x)).split(batch_size):
            batch = x[indices].clone()
            sign = torch.where(torch.rand(len(batch)) < .5, -1., 1.)[:, None]
            batch[..., 1] *= sign
            batch[..., 3] *= sign
            loss = (model(batch, mask[indices]) - y[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        score = metrics(yv, predict(model, xv, mv), mean)
        history.append({"epoch": epoch + 1, **score})
        print(f"epoch={epoch + 1} val_rmse={score['rmse']:.6f}", flush=True)
        if score['mse'] < best:
            best, best_state, stale = score['mse'], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 8:
            break
    output.mkdir(parents=True)
    torch.save({"state_dict": best_state, "config": model.config, "model_kind": model_kind}, output / "model.pt")
    report = {"seed": seed, "model_kind": model_kind, "split": split, "dataset_sha256": manifest['graphs_sha256'],
              "training_mean": mean, "loss": "MSE", "history": history,
              "best_validation": min(history, key=lambda r: r['mse']),
              "n_training_graphs": len(x), "n_validation_graphs": len(xv),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "source_sha256": source_hashes,
              "test_evaluated": False, "model_sha256": digest(output / 'model.pt')}
    (output / "training.json").write_text(json.dumps(report, indent=2) + '\n')
    return report
