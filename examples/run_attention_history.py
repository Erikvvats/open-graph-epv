"""Run the nine registered experiments, then freeze selection and audit once.

Run from the project directory with PYTHONPATH=src or an installed package.
Each subprocess uses two CPU threads; two workers is the default.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=3:raise ValueError('Use one to three workers')
    runs=Path('outputs/attention_history_v1');runs.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'}
    prefix=[sys.executable,'-m','open_graph_epv.attention_experiment']
    def execute(mode,seed):
        name=f'{mode}_seed{seed}';directory=runs/name
        if (directory/'training.json').exists():
            print('Already complete:',name,flush=True);return name
        with (runs/f'{name}.log').open('x') as log:
            result=subprocess.run(prefix+['train','--mode',mode,'--seed',str(seed),'--out',str(directory)],stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect {runs/name}.log')
        print('Completed:',name,flush=True);return name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,mode,seed) for mode in ('snapshot','history','repeat') for seed in (0,1,2)]
        for future in as_completed(futures):future.result()
    selection=runs/'selection.json'
    subprocess.run(prefix+['select','--out',str(selection)],check=True,env=env)
    subprocess.run(prefix+['evaluate','--selection',str(selection),'--out',str(runs/'audit')],check=True,env=env)


if __name__=='__main__':main()
