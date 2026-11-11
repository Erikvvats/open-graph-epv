"""Primary EPV workflow: train and deploy against audited contextual-xG targets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from . import training
from .bundle import EPVEnsemble
from .contextual_xg import load_models as load_teacher
from .data import FEATURES,prepare as prepare_graphs,load_xg
from .evaluation import select as select_models,audit as audit_models,verify_selection,value_metrics,paired_match_difference
from .source import digest,match_path

TARGET='signed first future contextual xG within 15 seconds, with explicit geometric fallback'
SOURCES=('contextual_epv.py','model.py','data.py','training.py','evaluation.py','bundle.py')


def source_hashes():return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def period_for_frame(metadata,frame):
    matches=[p for p in metadata['match_periods'] if p['period'] in (1,2) and p['start_frame']<=frame<=p['end_frame']]
    if len(matches)!=1:raise ValueError('Ambiguous or missing snapshot period')
    return matches[0]


def classify_target(shots,frame,period,period_end,target,context_source,geometry_source):
    if frame+150>period_end:raise ValueError('Incomplete target horizon')
    candidates=[s for s in shots if s['period']==period and frame<s['frame']<=frame+150]
    if not candidates:
        if target!=0:raise ValueError('Nonzero label without a future shot')
        return 0,-1
    first=min(candidates,key=lambda s:s['frame'])
    if not np.isclose(abs(target),first['xg'],atol=1e-8,rtol=1e-6):raise ValueError('Target differs from first-shot xG')
    if first['source']==context_source:kind=1
    elif first['source']==geometry_source:kind=2
    else:raise ValueError('Unknown teacher source')
    return kind,first['frame']


def prepare(root,labels,teacher,output,reference=None):
    root,labels,teacher,output=map(Path,(root,labels,teacher,output))
    if output.exists():raise FileExistsError(output)
    models,selection=load_teacher(teacher)
    transfer=json.loads((labels/'report.json').read_text())
    if models['selected_family']!='context':raise ValueError('This workflow requires the selected spatial context teacher')
    if transfer['model_sha256']!=selection['models_sha256'] or digest(labels/'shot_xg.csv')!=transfer['labels_sha256']:
        raise ValueError('Teacher or transferred labels changed')
    if digest(labels/'shot_audit.json')!=transfer['audit_sha256']:raise ValueError('Transfer audit changed')
    context_source=f'statsbomb_spatial_context_v1:{selection["models_sha256"]}'
    geometry_source=f'statsbomb_geometry_v1:{models["fallback_model_sha256"]}'
    mids=sorted(int(r['id']) for r in json.loads((root/'data/matches.json').read_text()))
    manifest=prepare_graphs(root,mids,labels/'shot_xg.csv',output,stride=20,horizon_seconds=15,include_extrapolated=False)
    # Both the teacher and EPV preparation must use the same verified raw inputs.
    if manifest['inputs']!=transfer['inputs']:raise ValueError('Tracking/event inputs differ from teacher transfer')
    if manifest['xg_sources']!=sorted([context_source,geometry_source]):raise ValueError('Unexpected shot teacher sources')
    with np.load(output/'graphs.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    shot_map=load_xg(labels/'shot_xg.csv',root,mids)
    metadata={mid:json.loads(match_path(root,mid,'match.json').read_text()) for mid in mids}
    kind=[];future_frames=[]
    for mid,frame,target in zip(arrays['match_id'],arrays['frame'],arrays['y']):
        period=period_for_frame(metadata[int(mid)],int(frame))
        k,f=classify_target(shot_map[int(mid)],int(frame),period['period'],period['end_frame'],float(target),context_source,geometry_source)
        kind.append(k);future_frames.append(f)
    np.savez_compressed(output/'target_provenance.npz',kind=np.array(kind,np.int8),future_shot_frame=np.array(future_frames),
                        match_id=arrays['match_id'],frame=arrays['frame'])
    if reference:
        with np.load(Path(reference)/'graphs.npz',allow_pickle=False) as old:
            for key in ('x','mask','match_id','frame'):
                if not np.array_equal(arrays[key],old[key]):raise ValueError(f'Snapshot comparison changed: {key}')
            if not np.array_equal(arrays['y']!=0,old['y']!=0):raise ValueError('Shot presence changed when replacing teacher')
    manifest.update(target=TARGET,teacher_models_sha256=selection['models_sha256'],
        teacher_selection_sha256=digest(teacher/'selection.json'),transfer_report_sha256=digest(labels/'report.json'),
        teacher_coverage={'context_shots':transfer['n_new_model_applied'],'fallback_shots':transfer['n_shots']-transfer['n_new_model_applied'],
                          'total_shots':transfer['n_shots']},
        target_kind_counts={str(k):kind.count(k) for k in (0,1,2)},
        target_kind_meaning={'0':'no future shot','1':'contextual xG','2':'geometric fallback xG'},
        target_provenance_sha256=digest(output/'target_provenance.npz'),
        preparation_source_sha256=source_hashes(),reference_snapshots_verified=reference is not None)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_data(data):
    data=Path(data);m=json.loads((data/'manifest.json').read_text())
    if m.get('target')!=TARGET or m.get('preparation_source_sha256')!=source_hashes():raise ValueError('Wrong target or changed pipeline source')
    for file,key in [('graphs.npz','graphs_sha256'),('target_provenance.npz','target_provenance_sha256')]:
        if digest(data/file)!=m[key]:raise ValueError(f'Changed dataset {file}')
    return m


def register(data,split_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    m=verify_data(data);split=json.loads(Path(split_path).read_text());training.validate_split(split,m['matches'])
    result={'date':'2026-11-06','target':TARGET,'families':['gnn','ball','set'],'seeds':[0,1,2],
        'dataset_sha256':m['graphs_sha256'],'dataset_manifest_sha256':digest(Path(data)/'manifest.json'),
        'teacher_models_sha256':m['teacher_models_sha256'],'teacher_coverage':m['teacher_coverage'],
        'split':split,'split_sha256':digest(split_path),'max_epochs':30,'batch_size':64,'early_stopping_patience':8,
        'optimizer':'AdamW, lr=0.001, weight_decay=0.001; MSE; clip norm 1',
        'selection':'Choose family by validation MSE of equal-weight three-seed mirror-averaged ensemble; report all families.',
        'freeze':'Dataset, source, training reports and checkpoints frozen before test evaluation.',
        'test_status':'Previously examined three test matches; exploratory evaluation on a new teacher target, not a fresh holdout.',
        'source_sha256':source_hashes(),'comparisons':['GNN versus training mean','GNN versus ball','GNN versus no messages'],
        'uncertainty':'Exact paired whole-match bootstrap, 27 ordered resamples; descriptive only.',
        'scope':'Predict contextual-xG teacher values; no verified goal calibration or causal action valuation claim.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify_protocol(data,protocol_path):
    m=verify_data(data);p=json.loads(Path(protocol_path).read_text())
    if (p['source_sha256']!=source_hashes() or p['dataset_sha256']!=m['graphs_sha256']
            or p['dataset_manifest_sha256']!=digest(Path(data)/'manifest.json')):raise ValueError('Registered protocol/data changed')
    return m,p


def train(data,protocol_path,split_path,family,seed,output):
    m,p=verify_protocol(data,protocol_path)
    if family not in p['families'] or seed not in p['seeds'] or digest(split_path)!=p['split_sha256']:
        raise ValueError('Unregistered training settings')
    r=training.train(data,split_path,output,epochs=p['max_epochs'],seed=seed,batch_size=p['batch_size'],model_kind=family)
    r.update(target=TARGET,teacher_models_sha256=m['teacher_models_sha256'],contextual_protocol_sha256=digest(protocol_path))
    (Path(output)/'training.json').write_text(json.dumps(r,indent=2)+'\n')
    return r


def select(data,runs,protocol_path,output):
    m,p=verify_protocol(data,protocol_path)
    for family in p['families']:
        for seed in p['seeds']:
            r=json.loads((Path(runs)/f'{family}_seed{seed}'/'training.json').read_text())
            if r.get('target')!=TARGET or r.get('contextual_protocol_sha256')!=digest(protocol_path):raise ValueError('Run is not contextual-target training')
    return select_models(data,runs,protocol_path,output)


def evaluate(data,runs,selection_path,output):
    m=verify_data(data);selection=json.loads(Path(selection_path).read_text())
    if selection['protocol']['source_sha256']!=source_hashes():raise ValueError('Changed contextual experiment source')
    report=audit_models(data,runs,selection_path,output)
    with np.load(Path(data)/'target_provenance.npz',allow_pickle=False) as z:
        test=np.isin(z['match_id'],selection['split']['test']);kind=z['kind'][test]
    with np.load(Path(output)/'predictions.npz',allow_pickle=False) as z:
        subset_metrics={}
        for k in (0,1,2):
            use=kind==k
            subset_metrics[m['target_kind_meaning'][str(k)]]={'n':int(use.sum()),
                'models':{family:value_metrics(z['target'][use],z[family][use],selection['training_mean']) for family in ('gnn','ball','set')} if use.any() else {}}
    report.update(target=TARGET,teacher_coverage=m['teacher_coverage'],target_kind_counts=m['target_kind_counts'],
        test_target_subsets=subset_metrics,dataset_manifest_sha256=digest(Path(data)/'manifest.json'),
        pipeline_source_sha256=source_hashes(),
        limitations=['Contextual teacher used for 413/526 shot markers; 113 use explicit geometric fallback.',
            'Only three test matches, already examined in earlier experiments; exploratory extension.',
            'Target is an estimated signed future xG value, not observed goals or official provider xG.',
            'SkillCorner transfer calibration, shot coverage and penalty identification remain unverified.',
            'RMSE under a changed teacher cannot be directly compared with old-target RMSE.',
            'Overlapping snapshots are not independent; bootstrap omits retraining and teacher uncertainty.',
            'No validated causal movement recommendation or reinforcement-learning policy.'])
    (Path(output)/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def export_bundle(data,runs,selection_path,audit_path,teacher,labels,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    m=verify_data(data);selection=json.loads(Path(selection_path).read_text());verify_selection(data,runs,selection)
    audit=json.loads(Path(audit_path).read_text());_,teacher_selection=load_teacher(teacher)
    if audit['selection_sha256']!=digest(selection_path) or audit.get('target')!=TARGET:raise ValueError('Wrong target audit')
    if teacher_selection['models_sha256']!=m['teacher_models_sha256']:raise ValueError('Wrong xG teacher')
    if digest(Path(labels)/'report.json')!=m['transfer_report_sha256']:raise ValueError('Wrong transfer provenance')
    output.mkdir(parents=True)
    families=sorted({'gnn',selection['selected_family']})
    records=[r for r in selection['models'] if r['family'] in families]
    for r in records:
        destination=output/'models'/r['name'];destination.mkdir(parents=True)
        for file in ('model.pt','training.json'):shutil.copyfile(Path(runs)/r['name']/file,destination/file)
    for source,name in [(selection_path,'selection.json'),(audit_path,'audit.json'),
        (Path(data)/'manifest.json','dataset_manifest.json'),(Path(teacher)/'models.json','xg_models.json'),
        (Path(teacher)/'selection.json','xg_selection.json'),(Path(labels)/'report.json','xg_transfer_report.json')]:
        shutil.copyfile(source,output/name)
    sources=output/'source'/'open_graph_epv';sources.mkdir(parents=True)
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,sources/source.name)
    (output/'NOTICE.txt').write_text('Research model: contextual StatsBomb-trained xG targets on SkillCorner tracking.\n'
        '113/526 shot markers use an explicit geometry fallback. SkillCorner probability calibration is unverified.\n'
        'Value is signed first-shot estimated xG within 15 seconds, relative to current possession.\n'
        'Code MIT; data/derived artifacts retain separate provider terms and attribution.\n'
        'https://github.com/hudl/open-data/blob/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/LICENSE.pdf\n'
        'https://github.com/SkillCorner/opendata\n')
    manifest={'schema_version':'contextual-epv-v1','target':TARGET,'features':FEATURES,'horizon_seconds':15,
        'selected_family':selection['selected_family'],'included_families':families,'models':records,
        'teacher_coverage':m['teacher_coverage'],'runtime_source_sha256':source_hashes(),
        'files':{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class ContextualEnsemble(EPVEnsemble):
    def score_frame(self,frame,previous,metadata):
        result=super().score_frame(frame,previous,metadata)
        result.update(target=TARGET,teacher_coverage=self.teacher_coverage)
        return result


def load_bundle(directory,family=None):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    if m.get('schema_version')!='contextual-epv-v1' or m.get('target')!=TARGET or m['features']!=FEATURES:
        raise ValueError('Not a contextual EPV inference bundle')
    if m['runtime_source_sha256']!=source_hashes():raise ValueError('Bundle runtime code changed')
    for path,checksum in m['files'].items():
        if digest(directory/path)!=checksum:raise ValueError(f'Bundle file changed: {path}')
    family=family or m['selected_family']
    if family not in m['included_families']:raise ValueError('Requested family not exported')
    models=[training.load_epv(directory/'models'/r['name']) for r in m['models'] if r['family']==family]
    if len(models)!=3:raise ValueError('Expected a three-seed ensemble')
    ensemble=ContextualEnsemble(models,family);ensemble.teacher_coverage=m['teacher_coverage']
    return ensemble


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',default='data/skillcorner');p.add_argument('--labels',default='outputs/skillcorner_contextual_xg_v1')
    p.add_argument('--teacher',default='outputs/contextual_xg_v1');p.add_argument('--reference');p.add_argument('--out',required=True)
    p=sub.add_parser('register');p.add_argument('--data',default='outputs/contextual_epv_graphs_v1');p.add_argument('--split',default='configs/split.json');p.add_argument('--out',required=True)
    p=sub.add_parser('train');p.add_argument('--data',default='outputs/contextual_epv_graphs_v1');p.add_argument('--protocol',default='configs/contextual_epv_v1.json')
    p.add_argument('--split',default='configs/split.json');p.add_argument('--family',choices=['gnn','ball','set'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
    for name in ('select','evaluate','export'):
        p=sub.add_parser(name);p.add_argument('--data',default='outputs/contextual_epv_graphs_v1');p.add_argument('--runs',default='outputs/contextual_epv_v1');p.add_argument('--out',required=True)
        if name=='select':p.add_argument('--protocol',default='configs/contextual_epv_v1.json')
        else:p.add_argument('--selection',default='outputs/contextual_epv_v1/selection.json')
        if name=='export':
            p.add_argument('--audit',default='outputs/contextual_epv_v1/audit/report.json');p.add_argument('--teacher',default='outputs/contextual_xg_v1');p.add_argument('--labels',default='outputs/skillcorner_contextual_xg_v1')
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.root,a.labels,a.teacher,a.out,a.reference);print(json.dumps(r['target_kind_counts'],indent=2))
    elif a.command=='register':r=register(a.data,a.split,a.out);print('Registered contextual-target protocol')
    elif a.command=='train':train(a.data,a.protocol,a.split,a.family,a.seed,a.out)
    elif a.command=='select':r=select(a.data,a.runs,a.protocol,a.out);print('Selected on validation:',r['selected_family'])
    elif a.command=='evaluate':
        r=evaluate(a.data,a.runs,a.selection,a.out);print(json.dumps({k:v['ensemble']['rmse'] for k,v in r['models'].items()},indent=2))
    else:r=export_bundle(a.data,a.runs,a.selection,a.audit,a.teacher,a.labels,a.out);print('Exported contextual EPV bundle:',r['included_families'])


if __name__=='__main__':main()
