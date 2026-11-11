"""Score one raw tracking frame with a three-seed transformer ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from open_graph_epv.attention_experiment import load
from open_graph_epv.history_data import aligned_history,OFFSETS
from open_graph_epv.source import match_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/skillcorner')
    p.add_argument('--runs',default='outputs/attention_history_v1')
    p.add_argument('--mode',choices=['snapshot','history','repeat'],required=True)
    p.add_argument('--match',type=int,required=True);p.add_argument('--frame',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    meta=json.loads(match_path(a.root,a.match,'match.json').read_text());buffer={};found=None
    with match_path(a.root,a.match,'tracking_extrapolated.jsonl').open() as stream:
        for line in stream:
            current=json.loads(line);f=current['frame']
            if f>a.frame:break
            buffer[f]=current
            for old in list(buffer):
                if old<f+min(OFFSETS)-1:del buffer[old]
            if f==a.frame:found=current;break
    if found is None:raise ValueError('Exact frame not found')
    x,mask=aligned_history(found,buffer,meta)
    if mask[-1].sum()<3 or not mask[-1,0]:raise ValueError('Current frame is not an eligible observed graph')
    models=[load(Path(a.runs)/f'{a.mode}_seed{seed}') for seed in (0,1,2)]
    with torch.no_grad():
        values=[float(m.symmetric(torch.from_numpy(x)[None],torch.from_numpy(mask)[None])[0]) for m in models]
    print(json.dumps({'match_id':a.match,'frame':a.frame,'mode':a.mode,'value':float(np.mean(values)),
        'seed_values':values,'seed_std_not_confidence_interval':float(np.std(values)),
        'target':'signed first-shot estimated xG over next 15s relative to current possession',
        'current_possession':found['possession']['group'],'observed_nodes_by_time':mask.sum(1).tolist(),
        'offset_seconds':[v/10 for v in OFFSETS]},indent=2))


if __name__=='__main__':main()
