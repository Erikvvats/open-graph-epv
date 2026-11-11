"""Train contextual-target GNN and controls, freeze selection, audit and export."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--data',default='outputs/contextual_epv_graphs_v1');p.add_argument('--runs',default='outputs/contextual_epv_v1')
    p.add_argument('--protocol',default='configs/contextual_epv_v1.json');p.add_argument('--split',default='configs/split.json')
    p.add_argument('--teacher',default='outputs/contextual_xg_v1');p.add_argument('--labels',default='outputs/skillcorner_contextual_xg_v1')
    a=p.parse_args()
    if not 1<=a.workers<=3:raise ValueError('Use one to three workers')
    runs=Path(a.runs);runs.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'}
    prefix=[sys.executable,'-m','open_graph_epv.contextual_epv']
    def execute(family,seed):
        name=f'{family}_seed{seed}';directory=runs/name
        if (directory/'training.json').exists():return name
        with (runs/f'{name}.log').open('x') as log:
            result=subprocess.run(prefix+['train','--data',a.data,'--protocol',a.protocol,'--split',a.split,
                '--family',family,'--seed',str(seed),'--out',str(directory)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {runs/name}.log')
        print('Completed:',name,flush=True);return name
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(execute,family,seed) for family in ('gnn','ball','set') for seed in (0,1,2)]
        for f in as_completed(futures):f.result()
    common=['--data',a.data,'--runs',a.runs]
    selection=runs/'selection.json';audit=runs/'audit'
    subprocess.run(prefix+['select',*common,'--protocol',a.protocol,'--out',str(selection)],env=env,check=True)
    subprocess.run(prefix+['evaluate',*common,'--selection',str(selection),'--out',str(audit)],env=env,check=True)
    subprocess.run(prefix+['export',*common,'--selection',str(selection),'--audit',str(audit/'report.json'),
        '--teacher',a.teacher,'--labels',a.labels,'--out',str(runs/'bundle')],env=env,check=True)


if __name__=='__main__':main()
