"""Repeat the same-state SkillCorner benchmark with the new contextual target."""
import argparse
import json
from pathlib import Path
from open_graph_epv.provider_comparison import prepare,calibrate,evaluate
from open_graph_epv.contextual_epv import TARGET,verify_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/skillcorner')
    p.add_argument('--data',default='outputs/contextual_epv_graphs_v1');p.add_argument('--runs',default='outputs/contextual_epv_v1')
    p.add_argument('--xg',default='outputs/skillcorner_contextual_xg_v1/shot_xg.csv')
    p.add_argument('--protocol',default='configs/contextual_provider_comparison_v1.json')
    p.add_argument('--out',default='outputs/contextual_provider_v1');a=p.parse_args()
    verify_data(a.data);out=Path(a.out);selection=Path(a.runs)/'selection.json'
    if out.exists():raise FileExistsError(out)
    prepare(a.root,a.data,a.xg,selection,out/'pairs')
    calibrate(out/'pairs',a.data,a.runs,selection,a.protocol,out/'calibration.json')
    report=evaluate(out/'pairs',a.data,a.runs,selection,out/'calibration.json',out/'audit')
    report.update(target=TARGET,limitations=[
        'Transferred contextual xG with explicit geometric fallback, not verified goals or official provider xG.',
        'Provider horizon and goal target differ; validation-only adaptation tests our proxy task.',
        'Previously examined three test matches; exploratory extension, not a fresh holdout.',
        'SkillCorner model version and training overlap are unknown.',
        'Partial observed tracking, uncertain shot coverage and unverified teacher calibration remain limitations.'])
    (out/'audit/report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:m['rmse'] for name,m in report['metrics'].items()},indent=2))


if __name__=='__main__':main()
