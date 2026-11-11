"""Compare frozen EPV models and SkillCorner values at exactly matched event starts.

This is a task-transfer benchmark against estimated future xG, not verified
goal probability or an independent test of SkillCorner's native target.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import frame_graph, future_target, load_xg
from .evaluation import verify_selection, value_metrics, paired_match_difference
from .source import digest, match_path, read_events
from .training import load_epv, predict


def provider_values(event):
    """Missing values abstain; zero is a valid probability."""
    try:
        pair=np.array([float(event[k]) for k in ('possession_epv_for_start','possession_epv_against_start')])
    except (KeyError,TypeError,ValueError):
        return None
    return pair if np.isfinite(pair).all() and ((0<=pair)&(pair<=1)).all() else None


def candidate_events(events):
    counts=Counter();grouped=defaultdict(list)
    for event in events:
        if event['event_type']!='player_possession':continue
        counts['possession_events']+=1
        if provider_values(event) is None:
            counts['missing_or_invalid_provider_value']+=1;continue
        if event.get('is_player_possession_start_matched')!='True':
            counts['unmatched_event_start']+=1;continue
        grouped[(int(event['period']),int(event['frame_start']))].append(event)
    eligible={}
    for key,events in grouped.items():
        if len(events)!=1:counts['ambiguous_same_frame_events']+=len(events)
        else:eligible[key]=events[0]
    return eligible,counts


def affine_fit(x,y):
    x=np.asarray(x,float).reshape(len(y),-1)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite calibration data')
    design=np.column_stack([np.ones(len(y)),x])
    coef,_,rank,_=np.linalg.lstsq(design,y,rcond=None)
    return {'intercept':float(coef[0]),'coefficients':coef[1:].tolist(),'rank':int(rank),'n_fit':len(y)}


def affine_predict(model,x):
    x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    return model['intercept']+x@np.asarray(model['coefficients'])


def prepare(root,data,xg,selection_path,output):
    root,data,output=Path(root),Path(data),Path(output)
    if output.exists():raise FileExistsError(output)
    selection=json.loads(Path(selection_path).read_text())
    original=json.loads((data/'manifest.json').read_text())
    if original['horizon_seconds']!=15 or original['include_extrapolated']:raise ValueError('Unsupported frozen target')
    if digest(xg)!=original['xg_sha256']:raise ValueError('Shot xG source changed')
    matches=selection['split']['val']+selection['split']['test']
    shots=load_xg(xg,root,matches)
    vectors=[];masks=[];targets=[];records=[];audits=[];inputs={}
    for mid in matches:
        for suffix in ('match.json','dynamic_events.csv','tracking_extrapolated.jsonl'):
            path=match_path(root,mid,suffix);relative=str(path.relative_to(root));checksum=digest(path)
            if original['inputs'].get(relative)!=checksum:raise ValueError(f'Changed source {relative}')
            inputs[relative]=checksum
        meta=json.loads(match_path(root,mid,'match.json').read_text())
        ends={p['period']:p['end_frame'] for p in meta['match_periods']}
        events,counts=candidate_events(read_events(root,mid))
        found=set();previous=None
        with match_path(root,mid,'tracking_extrapolated.jsonl').open() as stream:
            for line in stream:
                frame=json.loads(line)
                if previous is not None and frame['frame']<=previous['frame']:raise ValueError('Unordered tracking')
                key=(frame.get('period'),frame['frame'])
                if key in events:
                    found.add(key);event=events[key]
                    graph=frame_graph(frame,previous,meta,include_extrapolated=False)
                    if graph is None:counts['unobservable_graph']+=1
                    elif graph[2]!=int(event['team_id']):counts['possession_team_mismatch']+=1
                    else:
                        y=future_target(shots[mid],frame['frame'],frame['period'],graph[2],ends[frame['period']],150)
                        if y is None:counts['incomplete_horizon']+=1
                        else:
                            pair=provider_values(event)
                            vectors.append(graph[0]);masks.append(graph[1]);targets.append(y)
                            records.append({'match_id':mid,'event_id':event['event_id'],'frame':frame['frame'],
                                'period':frame['period'],'team_id':graph[2],'provider_for':float(pair[0]),
                                'provider_against':float(pair[1]),'split':'val' if mid in selection['split']['val'] else 'test'})
                            counts['included']+=1
                previous=frame
        counts['missing_exact_frame']=len(set(events)-found)
        if sum(v for k,v in counts.items() if k!='possession_events')!=counts['possession_events']:
            raise ValueError('Audit counts do not reconcile')
        audits.append({'match_id':mid,**dict(counts)})
        print(f'{mid}: {counts["included"]}/{counts["possession_events"]} exact-start pairs',flush=True)
    if not records or {r['match_id'] for r in records}!=set(matches):raise ValueError('Missing comparison matches')
    output.mkdir(parents=True)
    np.savez_compressed(output/'pairs.npz',x=np.asarray(vectors),mask=np.asarray(masks),y=np.asarray(targets),
        match_id=np.array([r['match_id'] for r in records]),frame=np.array([r['frame'] for r in records]),
        provider=np.array([[r['provider_for'],r['provider_against']] for r in records]))
    (output/'rows.json').write_text(json.dumps(records,indent=2)+'\n')
    manifest={'target':original['target'],'horizon_seconds':15,'xg_sources':original['xg_sources'],
        'n_pairs':len(records),'audit':audits,'inputs':inputs,'split':selection['split'],
        'original_selection_sha256':digest(selection_path),'original_dataset_sha256':selection['dataset_sha256'],
        'xg_sha256':digest(xg),'pairs_sha256':digest(output/'pairs.npz'),'rows_sha256':digest(output/'rows.json'),
        'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in ('provider_comparison.py','data.py')}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_pairs(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    for name,key in [('pairs.npz','pairs_sha256'),('rows.json','rows_sha256')]:
        if digest(directory/name)!=manifest[key]:raise ValueError(f'Comparison data changed: {name}')
    for name,checksum in manifest['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=checksum:raise ValueError(f'Comparison source changed: {name}')
    return manifest


def family_predictions(selection,runs,x,mask):
    x,mask=torch.from_numpy(x),torch.from_numpy(mask)
    result={}
    for family in ('gnn','ball','set'):
        records=[r for r in selection['models'] if r['family']==family]
        if sorted(r['seed'] for r in records)!=[0,1,2]:raise ValueError('Expected all three frozen seeds')
        result[family]=np.mean([predict(load_epv(Path(runs)/r['name']),x,mask) for r in records],axis=0)
    return result


def calibrate(pairs,data,runs,selection_path,protocol_path,output):
    pairs,output=Path(pairs),Path(output)
    if output.exists():raise FileExistsError(output)
    selection=json.loads(Path(selection_path).read_text());verify_selection(data,runs,selection)
    manifest=verify_pairs(pairs)
    if manifest['original_selection_sha256']!=digest(selection_path):raise ValueError('Selection mismatch')
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol['families']!=['gnn','ball','set'] or protocol['seeds']!=[0,1,2]:raise ValueError('Unexpected protocol')
    torch.set_num_threads(2)
    with np.load(pairs/'pairs.npz',allow_pickle=False) as z:
        use=np.isin(z['match_id'],selection['split']['val'])
        y=z['y'][use];provider=z['provider'][use]
        predictions=family_predictions(selection,runs,z['x'][use],z['mask'][use])
    net=provider[:,0]-provider[:,1]
    transforms={'skillcorner_adapted':affine_fit(provider,y),'skillcorner_net_adapted':affine_fit(net,y),
                'gnn_adapted':affine_fit(predictions['gnn'],y)}
    predictions.update(skillcorner_raw=net,skillcorner_adapted=affine_predict(transforms['skillcorner_adapted'],provider),
        skillcorner_net_adapted=affine_predict(transforms['skillcorner_net_adapted'],net),
        gnn_adapted=affine_predict(transforms['gnn_adapted'],predictions['gnn']))
    prior=selection['training_mean'];predictions['training_mean']=np.full(len(y),prior)
    predictions['validation_mean']=np.full(len(y),y.mean())
    result={'transforms':transforms,'validation_mean':float(y.mean()),'training_mean':prior,
        'fit_matches':selection['split']['val'],'test_matches':selection['split']['test'],
        'n_validation':len(y),'validation':{k:value_metrics(y,p,prior) for k,p in predictions.items()},
        'protocol':protocol,'protocol_sha256':digest(protocol_path),'pairs_manifest_sha256':digest(pairs/'manifest.json'),
        'original_selection_sha256':digest(selection_path),'frozen_models':selection['models'],
        'test_evaluated':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def evaluate(pairs,data,runs,selection_path,calibration_path,output):
    pairs,output=Path(pairs),Path(output)
    if output.exists():raise FileExistsError(output)
    selection=json.loads(Path(selection_path).read_text());verify_selection(data,runs,selection)
    manifest=verify_pairs(pairs);calibration=json.loads(Path(calibration_path).read_text())
    if calibration['pairs_manifest_sha256']!=digest(pairs/'manifest.json'):raise ValueError('Prepared pairs changed')
    if calibration['original_selection_sha256']!=digest(selection_path):raise ValueError('Frozen model selection changed')
    if calibration['fit_matches']!=selection['split']['val'] or calibration['test_matches']!=selection['split']['test']:
        raise ValueError('Calibration split mismatch')
    torch.set_num_threads(2)
    with np.load(pairs/'pairs.npz',allow_pickle=False) as z:
        use=np.isin(z['match_id'],selection['split']['test'])
        y=z['y'][use];provider=z['provider'][use];groups=z['match_id'][use]
        predictions=family_predictions(selection,runs,z['x'][use],z['mask'][use])
    transforms=calibration['transforms'];net=provider[:,0]-provider[:,1]
    predictions.update(skillcorner_raw=net,skillcorner_adapted=affine_predict(transforms['skillcorner_adapted'],provider),
        skillcorner_net_adapted=affine_predict(transforms['skillcorner_net_adapted'],net),
        gnn_adapted=affine_predict(transforms['gnn_adapted'],predictions['gnn']),
        training_mean=np.full(len(y),calibration['training_mean']),validation_mean=np.full(len(y),calibration['validation_mean']))
    prior=calibration['training_mean']
    metrics={name:value_metrics(y,p,prior) for name,p in predictions.items()}
    contrast_pairs=[('gnn','skillcorner_adapted'),('gnn_adapted','skillcorner_adapted'),
                    ('gnn','skillcorner_raw'),('gnn','ball'),('gnn','set'),('gnn','training_mean')]
    contrasts={f'{a}_minus_{b}':paired_match_difference(y,predictions[a],predictions[b],groups) for a,b in contrast_pairs}
    result={'purpose':calibration['protocol']['purpose'],'n_test':len(y),'n_test_matches':len(np.unique(groups)),
        'n_nonzero_targets':int((y!=0).sum()),'metrics':metrics,'comparisons':contrasts,
        'per_match':[{'match_id':int(mid),'n':int((groups==mid).sum()),
            'metrics':{name:value_metrics(y[groups==mid],p[groups==mid],prior) for name,p in predictions.items()}}
            for mid in np.unique(groups)],
        'gnn_provider_raw_pearson':float(np.corrcoef(predictions['gnn'],net)[0,1]),
        'calibration':calibration,'calibration_sha256':digest(calibration_path),
        'pairing_audit':manifest['audit'],'source_sha256':manifest['source_sha256'],
        'limitations':['Transferred geometric xG is an estimated target, not verified goals or official StatsBomb xG.',
            'Provider horizon, interruption handling and goal target differ; raw MSE is not a like-for-like native evaluation.',
            'Only three test matches, already used in earlier evaluation; exploratory extension.',
            'SkillCorner training-data overlap and open-data model version are unknown.',
            'Event-start selection and detection filtering restrict the evaluated population.',
            'Bootstrap conditions on fitted models; no xG-transfer, refit or label uncertainty.']}
    output.mkdir(parents=True)
    rows=json.loads((pairs/'rows.json').read_text());rows=[r for r in rows if r['split']=='test']
    with (output/'predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['match_id','event_id','frame','period','team_id','target',
                                                   'provider_for','provider_against',*predictions])
        writer.writerows([r['match_id'],r['event_id'],r['frame'],r['period'],r['team_id'],float(y[i]),
            r['provider_for'],r['provider_against'],*[float(p[i]) for p in predictions.values()]] for i,r in enumerate(rows))
    result['predictions_sha256']=digest(output/'predictions.csv')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',default='data/skillcorner');p.add_argument('--xg',default='outputs/skillcorner_xg_v1.csv')
    p.add_argument('--data',default='outputs/graphs_v1');p.add_argument('--selection',default='outputs/selection_v1.json');p.add_argument('--out',required=True)
    for name in ('calibrate','evaluate'):
        p=sub.add_parser(name);p.add_argument('--pairs',required=True);p.add_argument('--data',default='outputs/graphs_v1')
        p.add_argument('--runs',default='outputs');p.add_argument('--selection',default='outputs/selection_v1.json');p.add_argument('--out',required=True)
        if name=='calibrate':p.add_argument('--protocol',default='configs/provider_comparison_v1.json')
        else:p.add_argument('--calibration',required=True)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.root,a.data,a.xg,a.selection,a.out);print('Prepared',r['n_pairs'],'pairs')
    elif a.command=='calibrate':r=calibrate(a.pairs,a.data,a.runs,a.selection,a.protocol,a.out);print(json.dumps(r['transforms'],indent=2))
    else:
        r=evaluate(a.pairs,a.data,a.runs,a.selection,a.calibration,a.out)
        print(json.dumps({n:round(m['rmse'],6) for n,m in r['metrics'].items()},indent=2))


if __name__=='__main__':main()
