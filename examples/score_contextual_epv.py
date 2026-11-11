"""Score a raw SkillCorner snapshot with a contextual-target EPV bundle."""
import argparse
import json
from pathlib import Path
import torch
from open_graph_epv.contextual_epv import load_bundle
from open_graph_epv.source import match_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',default='outputs/contextual_epv_v1/bundle')
    p.add_argument('--root',default='data/skillcorner');p.add_argument('--match',type=int,required=True)
    p.add_argument('--frame',type=int,required=True);p.add_argument('--family',choices=['gnn','ball','set'])
    a=p.parse_args();torch.set_num_threads(2);model=load_bundle(a.bundle,a.family)
    metadata=json.loads(match_path(a.root,a.match,'match.json').read_text());previous=None
    with match_path(a.root,a.match,'tracking_extrapolated.jsonl').open() as stream:
        for line in stream:
            current=json.loads(line)
            if current['frame']==a.frame:
                result=model.score_frame(current,previous,metadata)
                print(json.dumps({'match_id':a.match,**result},indent=2));return
            if current['frame']>a.frame:break
            previous=current
    raise ValueError('Exact tracking frame unavailable')


if __name__=='__main__':main()
