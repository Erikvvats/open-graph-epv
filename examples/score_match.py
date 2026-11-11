"""Score one observed SkillCorner frame with the frozen GNN ensemble."""
import argparse
import json
from pathlib import Path

from open_graph_epv.bundle import load_bundle
from open_graph_epv.source import match_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/skillcorner'))
    parser.add_argument('--bundle', type=Path, default=Path('outputs/bundle_v1'))
    parser.add_argument('--match', type=int, default=2010085)
    parser.add_argument('--frame', type=int, default=100)
    args = parser.parse_args()
    model = load_bundle(args.bundle)
    metadata = json.loads(match_path(args.root, args.match, 'match.json').read_text())
    previous = None
    with match_path(args.root, args.match, 'tracking_extrapolated.jsonl').open() as f:
        for line in f:
            current = json.loads(line)
            if current['frame'] == args.frame:
                result = model.score_frame(current, previous, metadata)
                print(json.dumps({'match_id': args.match, **result}, indent=2))
                return
            previous = current
    raise ValueError('Requested frame is not in the tracking file')


if __name__ == '__main__':
    main()
