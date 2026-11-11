"""Publish compact numerical results and a figure from the frozen paired audit."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',default='outputs/provider_audit_v1')
    parser.add_argument('--out',default='reports');args=parser.parse_args()
    source,out=Path(args.audit),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report=json.loads((source/'report.json').read_text())
    (out/'provider_comparison_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'provider_comparison_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','n','rmse','mse','mae','predicted_mean','target_mean'])
        for name,m in report['metrics'].items():writer.writerow([name,*[m[k] for k in ('n','rmse','mse','mae','predicted_mean','target_mean')]])
    names=['training_mean','ball','set','gnn','skillcorner_adapted','gnn_adapted']
    labels=['Training\nmean','Ball-only','No\nmessages','GNN','SkillCorner\nadapted','GNN\nadapted']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    values=[report['metrics'][name]['rmse'] for name in names]
    axes[0].bar(labels,values,color=['#999999','#85c1e9','#a9cce3','#117a65','#d68910','#48c9b0'])
    for i,v in enumerate(values):axes[0].text(i,v+.0004,f'{v:.5f}',ha='center',fontsize=8)
    axes[0].set(ylim=(0,.033),ylabel='RMSE on our signed 15-second xG target',title=f"{report['n_test']:,} exact event starts; three test matches")
    keys=['gnn_minus_skillcorner_adapted','gnn_adapted_minus_skillcorner_adapted']
    for i,key in enumerate(keys):
        comp=report['comparisons'][key];d=comp['mse_difference'];lo,hi=comp['ci95']
        axes[1].errorbar(d*1e6,i,xerr=np.array([[d-lo],[hi-d]])*1e6,fmt='o',capsize=5,color=['#117a65','#48c9b0'][i])
    axes[1].axvline(0,color='gray',ls='--',lw=1)
    axes[1].set(yticks=[0,1],yticklabels=['GNN vs\nSkillCorner adapted','GNN adapted vs\nSkillCorner adapted'],
        ylim=(-.6,1.6),xlabel='MSE difference × 10⁶ (negative favors GNN)',title='Descriptive whole-match bootstrap intervals')
    fig.suptitle('Task-transfer comparison — not a test of native SkillCorner goal probability',fontsize=12)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fig.savefig(figures/'provider_comparison.png',dpi=160);fig.savefig(figures/'provider_comparison.pdf');plt.close(fig)
    print('Wrote numerical report, metrics CSV and PNG/PDF figure.')


if __name__=='__main__':main()
