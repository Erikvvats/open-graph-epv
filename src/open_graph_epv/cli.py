import argparse
import json
from pathlib import Path

from . import source
from .data import iter_graphs, prepare, shot_template


def main():
    parser = argparse.ArgumentParser(description="SkillCorner open-data GNN EPV")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit", "shot-template", "inspect", "prepare"):
        p = commands.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("data/skillcorner"))
        p.add_argument("--matches", nargs="+", required=True)
        if name == "download":
            p.add_argument("--tracking", action="store_true")
        if name in ("audit", "shot-template", "prepare"):
            p.add_argument("--out", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--xg-csv", type=Path, required=True)
            p.add_argument("--stride", type=int, default=20)
            p.add_argument("--horizon-seconds", type=float, default=15)
            p.add_argument("--include-extrapolated", action="store_true")
    p = commands.add_parser("train")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-kind", choices=['gnn', 'ball', 'set'], default='gnn')
    p = commands.add_parser("xg-download")
    p.add_argument("--root", type=Path, default=Path("data/statsbomb"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p = commands.add_parser("xg-train")
    p.add_argument("--shots", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("xg-label")
    p.add_argument("--root", type=Path, default=Path("data/skillcorner"))
    p.add_argument("--matches", nargs="+", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for command in ('select', 'evaluate'):
        p = commands.add_parser(command)
        p.add_argument('--data', type=Path, required=True)
        p.add_argument('--runs', type=Path, default=Path('outputs'))
        p.add_argument('--out', type=Path, required=True)
        if command == 'select':
            p.add_argument('--protocol', type=Path, default=Path('configs/epv_evaluation_v1.json'))
        else:
            p.add_argument('--selection', type=Path, required=True)
    p = commands.add_parser('export')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=Path('outputs'))
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--xg-model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('plot-report')
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, 'matches') and args.command != 'download' and args.matches == ['all']:
        catalogue = json.loads((args.root / 'data/matches.json').read_text())
        args.matches = sorted(int(m['id']) for m in catalogue)
    if args.command == "download":
        source.download(args.root, args.matches, args.tracking)
    elif args.command == "audit":
        result = source.audit(args.root, args.matches)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    elif args.command == "shot-template":
        shot_template(args.root, args.matches, args.out)
        print(f"Wrote {args.out}; xG is intentionally blank, not zero")
    elif args.command == "inspect":
        import numpy as np
        import torch
        from .model import GraphEPV
        torch.manual_seed(0)
        torch.set_num_threads(2)
        model = GraphEPV().eval()
        for match in args.matches:
            count, visible, sample = 0, [], []
            for (x, mask, team), frame, period, end in iter_graphs(args.root, match):
                count += 1
                visible.append(int(mask.sum()))
                if len(sample) < 8:
                    sample.append((x, mask))
            if not sample:
                raise ValueError(f"No valid graphs in match {match}")
            x = torch.from_numpy(np.stack([s[0] for s in sample]))
            mask = torch.from_numpy(np.stack([s[1] for s in sample]))
            with torch.no_grad():
                values = model.symmetric(x, mask)
            print(json.dumps({"match_id": int(match), "eligible_snapshots": count,
                              "mean_visible_nodes": float(np.mean(visible)),
                              "forward_finite": bool(torch.isfinite(values).all()),
                              "trained": False, "note": "Shape/finite-value check only; no predictive score"}))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.root, args.matches, args.xg_csv, args.out, args.stride,
                                 args.horizon_seconds, args.include_extrapolated), indent=2))
    elif args.command == "train":
        from .training import train
        train(args.data, args.split, args.out, args.epochs, args.seed, model_kind=args.model_kind)
    elif args.command == "xg-download":
        from .xg import download_shots
        result = download_shots(args.root, args.out, args.workers)
        print(f"Saved {result['shots']} StatsBomb shots")
    elif args.command == "xg-train":
        from .xg import fit_xg
        result = fit_xg(args.shots, args.out)
        print(json.dumps(result['metrics'], indent=2))
    elif args.command == "xg-label":
        from .xg import label_skillcorner
        print(json.dumps(label_skillcorner(args.root, args.matches, args.model, args.out), indent=2))
    elif args.command == 'select':
        from .evaluation import select
        result = select(args.data, args.runs, args.protocol, args.out)
        print('Selected on validation:', result['selected_family'])
    elif args.command == 'evaluate':
        from .evaluation import audit
        result = audit(args.data, args.runs, args.selection, args.out)
        print(json.dumps({k: v['ensemble']['rmse'] for k, v in result['models'].items()}, indent=2))
    elif args.command == 'export':
        from .bundle import export_bundle
        result = export_bundle(args.data, args.runs, args.selection, args.audit, args.xg_model, args.out)
        print('Exported families:', result['included_families'])
    elif args.command == 'plot-report':
        from .figures import plot_report
        plot_report(args.selection, args.audit, args.out)


if __name__ == "__main__":
    main()
