"""Read a raw tracking frame and return home_team_epv and away_team_epv."""
import argparse
import json
import torch

from open_graph_epv.team_epv import load_bundle
from open_graph_epv.source import match_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', default='outputs/team_epv_v1/bundle')
    p.add_argument('--root', default='data/skillcorner')
    p.add_argument('--match', required=True, type=int)
    p.add_argument('--frame', required=True, type=int)
    p.add_argument('--family', choices=['gnn', 'set', 'ball'])
    a = p.parse_args();torch.set_num_threads(2)
    model = load_bundle(a.bundle, a.family)
    meta = json.loads(match_path(a.root, a.match, 'match.json').read_text())
    previous = None
    with match_path(a.root, a.match, 'tracking_extrapolated.jsonl').open() as handle:
        for line in handle:
            frame = json.loads(line)
            if frame['frame'] == a.frame:
                print(json.dumps(model.score_frame(frame, previous, meta), indent=2));return
            if frame['frame'] > a.frame:
                break
            previous = frame
    raise ValueError('Requested exact frame is unavailable')


if __name__ == '__main__':
    main()
