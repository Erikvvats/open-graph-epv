"""Train all two-team EPV models, freeze selection, evaluate and export."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='outputs/contextual_epv_graphs_v1')
    p.add_argument('--data', default='outputs/team_epv_data_v1')
    p.add_argument('--runs', default='outputs/team_epv_v1')
    p.add_argument('--legacy', default='outputs/contextual_epv_v1/audit')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.workers <= 3:
        raise ValueError('Use one to three workers')
    runs = Path(a.runs);runs.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, '-m', 'open_graph_epv.team_epv']
    common = ['--base', a.base, '--data', a.data]
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'VECLIB_MAXIMUM_THREADS': '2'}
    def execute(family, seed):
        name = f'{family}_seed{seed}'
        if (runs/name/'training.json').exists():
            return name
        with (runs/f'{name}.log').open('x') as log:
            result = subprocess.run(prefix+['train', *common, '--family', family, '--seed', str(seed), '--out', str(runs/name)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')
        print('Completed:', name, flush=True)
        return name
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(execute, f, s) for f in ('gnn', 'ball', 'set') for s in (0, 1, 2)]
        for future in as_completed(futures):
            future.result()
    subprocess.run(prefix+['evaluate', *common, '--runs', a.runs, '--legacy', a.legacy], env=env, check=True)
    subprocess.run(prefix+['export', '--data', a.data, '--runs', a.runs, '--out', str(runs/'bundle')], env=env, check=True)


if __name__ == '__main__':
    main()
