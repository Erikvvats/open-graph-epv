"""Goal-trained StatsBomb contextual xG, with an audited SkillCorner transfer."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .source import digest,match_path,read_events,shot_events,REVISION
from .xg import SB_REVISION,geometry,fit_logistic,probability,binary_metrics
from .xg_context_features import FEATURES,FAMILIES,MissingContext,statsbomb_features,skillcorner_features

SOURCES=('contextual_xg.py','xg_context_features.py','xg.py')


def source_hashes():return {n:digest(Path(__file__).with_name(n)) for n in SOURCES}


def frozen_geometry(directory):
    directory=Path(directory);report=json.loads((directory/'report.json').read_text())
    if digest(directory/'model.json')!=report['model_sha256']:raise ValueError('Geometry model changed')
    return json.loads((directory/'model.json').read_text()),report


def prepare(root,old_model,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    parent=json.loads((root/'shots.manifest.json').read_text())
    if parent['revision']!=SB_REVISION or parent['shots_sha256']!=digest(root/'shots.csv'):raise ValueError('Changed shot corpus')
    _,old=frozen_geometry(old_model)
    with (root/'shots.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    x=np.zeros((len(rows),len(FEATURES)));eligible=np.zeros(len(rows),bool)
    x[:,:8]=geometry([float(r['distance_x']) for r in rows],[float(r['lateral_y']) for r in rows])
    lookup={(int(r['match_id']),r['event_id']):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('Duplicate shots')
    seen=set();audit=[];inputs={}
    for entry in parent['inputs']:
        mid=entry['match_id'];path=root/f'data/events/{mid}.json'
        blob=path.read_bytes();checksum=hashlib.sha256(blob).hexdigest()
        receipt=json.loads(path.with_suffix('.json.receipt.json').read_text())
        if checksum!=entry['sha256'] or checksum!=receipt['sha256'] or receipt['revision']!=SB_REVISION:
            raise ValueError(f'Changed raw events: {mid}')
        inputs[str(path.relative_to(root))]=checksum
        for event in json.loads(blob):
            key=(mid,event['id'])
            if key not in lookup:continue
            if key in seen:raise ValueError('Duplicate raw event')
            seen.add(key);i=lookup[key]
            if event['type']['name']!='Shot' or int(event['shot']['outcome']['name']=='Goal')!=int(rows[i]['goal']):
                raise ValueError('Outcome mismatch')
            reason=None
            try:
                values=statsbomb_features(event)
                if not np.allclose(values[:8],x[i,:8],atol=1e-9,rtol=1e-12):raise ValueError('Geometry mismatch')
                x[i,8:]=values[8:];eligible[i]=True
            except MissingContext as error:reason=str(error)
            audit.append({'match_id':mid,'event_id':event['id'],'context_eligible':bool(eligible[i]),'reason':reason})
    if seen!=set(lookup):raise ValueError('Missing raw events')
    groups=np.array([int(r['match_id']) for r in rows]);y=np.array([int(r['goal']) for r in rows])
    output.mkdir(parents=True)
    np.savez_compressed(output/'shots.npz',x=x,y=y,context_eligible=eligible,match_id=groups,event_id=np.array([r['event_id'] for r in rows]))
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    result={'revision':SB_REVISION,'features':FEATURES,'n_shots':len(y),'n_eligible':int(eligible.sum()),
        'n_goals':int(y.sum()),'n_eligible_goals':int(y[eligible].sum()),'split':old['split'],
        'by_split':{name:{'shots':int(np.isin(groups,mids).sum()),'eligible':int((np.isin(groups,mids)&eligible).sum()),
                'eligible_goals':int(y[np.isin(groups,mids)&eligible].sum()),'matches':len(mids)} for name,mids in old['split'].items()},
        'exclusions':dict(Counter(r['reason'] for r in audit if not r['context_eligible'])),
        'inputs':inputs,'parent_shots_sha256':parent['shots_sha256'],'shots_sha256':digest(output/'shots.npz'),
        'audit_sha256':digest(output/'audit.json'),'source_sha256':source_hashes(),
        'old_model_sha256':old['model_sha256'],'old_report_sha256':digest(Path(old_model)/'report.json')}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify_data(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    if digest(directory/'shots.npz')!=m['shots_sha256'] or m['source_sha256']!=source_hashes():raise ValueError('Data or source changed')
    if digest(directory/'audit.json')!=m['audit_sha256']:raise ValueError('Audit changed')
    return m


def select(data,old_model,protocol_path,output):
    data,output=Path(data),Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=verify_data(data);original,old=frozen_geometry(old_model)
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol['source_sha256']!=source_hashes() or protocol['shots_sha256']!=manifest['shots_sha256']:
        raise ValueError('Unregistered data or source')
    if protocol['split']!=old['split'] or manifest['old_model_sha256']!=old['model_sha256']:raise ValueError('Changed split/reference')
    with np.load(data/'shots.npz',allow_pickle=False) as z:
        train=np.isin(z['match_id'],old['split']['train'])&z['context_eligible']
        val=np.isin(z['match_id'],old['split']['val'])&z['context_eligible']
        x,y,xv,yv=z['x'][train],z['y'][train],z['x'][val],z['y'][val]
    models={};summaries={}
    for family,columns in FAMILIES.items():
        candidates=[]
        for penalty in protocol['penalties']:
            model=fit_logistic(x[:,columns],y,penalty)
            metrics=binary_metrics(yv,probability(model,xv[:,columns]))
            candidates.append((metrics,model))
        metrics,model=min(candidates,key=lambda item:item[0]['log_loss'])
        models[family]={'fit':model,'columns':columns,'features':[FEATURES[i] for i in columns]}
        summaries[family]={'selected_penalty':model['penalty'],'metrics':metrics,
                          'candidates':[{'penalty':m['penalty'],'validation_log_loss':v['log_loss']} for v,m in candidates]}
        print(f'{family}: penalty={model["penalty"]} validation log loss={metrics["log_loss"]:.6f}',flush=True)
    winner=min(FAMILIES,key=lambda k:summaries[k]['metrics']['log_loss'])
    summaries['frozen_geometry']={'metrics':binary_metrics(yv,probability(original,xv[:,:8]))}
    output.mkdir(parents=True)
    bundle={'features':FEATURES,'models':models,'selected_family':winner,'target':'goal',
        'training_matches':old['split']['train'],'source_revision':SB_REVISION,'fallback_geometry':original,
        'fallback_model_sha256':old['model_sha256'],'training_feature_min':x.min(0).tolist(),'training_feature_max':x.max(0).tolist(),
        'training_feature_q01':np.quantile(x,.01,axis=0).tolist(),'training_feature_q99':np.quantile(x,.99,axis=0).tolist(),
        'training_feature_mean':x.mean(0).tolist(),'training_feature_std':x.std(0).tolist(),
        'n_training_shots':len(y),'n_training_goals':int(y.sum())}
    (output/'models.json').write_text(json.dumps(bundle,indent=2)+'\n')
    result={'selected_family':winner,'validation':summaries,'n_train':len(y),'n_val':len(yv),'split':old['split'],
        'models_sha256':digest(output/'models.json'),'protocol':protocol,'protocol_sha256':digest(protocol_path),
        'data_manifest_sha256':digest(data/'manifest.json'),'source_sha256':source_hashes(),'test_evaluated':False}
    (output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def paired_difference(y,p,q,groups,seed=123,repetitions=2000):
    def loss(v):
        v=np.clip(v,1e-9,1-1e-9);return -(y*np.log(v)+(1-y)*np.log(1-v))
    delta=loss(p)-loss(q);mids=np.unique(groups)
    sums=np.array([delta[groups==m].sum() for m in mids]);counts=np.array([(groups==m).sum() for m in mids])
    samples=np.random.default_rng(seed).integers(len(mids),size=(repetitions,len(mids)))
    draws=sums[samples].sum(1)/counts[samples].sum(1)
    return {'log_loss_difference':float(delta.mean()),'descriptive_match_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),
        'n_matches':len(mids),'repetitions':repetitions,'interpretation':'Conditional on fitted models; reused holdout, no refitting or domain-transfer uncertainty.'}


def load_models(directory):
    directory=Path(directory);s=json.loads((directory/'selection.json').read_text())
    if s['source_sha256']!=source_hashes() or digest(directory/'models.json')!=s['models_sha256']:raise ValueError('Frozen model/source changed')
    return json.loads((directory/'models.json').read_text()),s


def evaluate(data,models,output):
    data,output=Path(data),Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=verify_data(data);bundle,selection=load_models(models)
    if digest(data/'manifest.json')!=selection['data_manifest_sha256']:raise ValueError('Data manifest changed after selection')
    with np.load(data/'shots.npz',allow_pickle=False) as z:
        use=np.isin(z['match_id'],selection['split']['test'])
        x,y,eligible,groups,ids=[z[k][use] for k in ('x','y','context_eligible','match_id','event_id')]
    frozen=probability(bundle['fallback_geometry'],x[:,:8])
    predictions={name:probability(model['fit'],x[eligible][:,model['columns']]) for name,model in bundle['models'].items()}
    predictions['frozen_geometry']=frozen[eligible]
    metrics={name:binary_metrics(y[eligible],p) for name,p in predictions.items()}
    contrasts={f'{a}_minus_{b}':paired_difference(y[eligible],predictions[a],predictions[b],groups[eligible])
        for a,b in [('keeper','geometry'),('context','geometry'),('context','keeper'),(bundle['selected_family'],'frozen_geometry')]}
    hybrid=frozen.copy();hybrid[eligible]=predictions[bundle['selected_family']]
    result={'selected_family':bundle['selected_family'],'n_test_all':len(y),'n_test_context':int(eligible.sum()),
        'n_test_goals':int(y.sum()),'n_context_goals':int(y[eligible].sum()),'metrics_on_same_context_shots':metrics,
        'comparisons':contrasts,'all_shots_hybrid':binary_metrics(y,hybrid),'all_shots_frozen_geometry':binary_metrics(y,frozen),
        'hybrid_minus_frozen_geometry':paired_difference(y,hybrid,frozen,groups),
        'selection':selection,'selection_sha256':digest(Path(models)/'selection.json'),
        'data_audit':{k:manifest[k] for k in ('n_shots','n_eligible','n_goals','n_eligible_goals','by_split','exclusions')},
        'limitations':['Test matches held out from fitting but previously examined for geometry research.',
            'Freeze-frame coverage and shot timing differ from SkillCorner tracking.',
            'No body part, verified penalty type, measured impact height or post-shot information.',
            'StatsBomb coordinates mapped to a nominal 105 x 68 m pitch.',
            'This test does not establish SkillCorner calibration or tactical intervention effects.']}
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',y=y,match_id=groups,event_id=ids,context_eligible=eligible,
                        hybrid=hybrid,frozen_geometry=frozen,**{f'eligible_{k}':v for k,v in predictions.items()})
    result['predictions_sha256']=digest(output/'predictions.npz')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def transfer(root,models,output):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    bundle,selection=load_models(models);family=bundle['selected_family'];selected=bundle['models'][family]
    receipt=json.loads((root/'download_manifest.json').read_text())
    if receipt['revision']!=REVISION:raise ValueError('Wrong SkillCorner revision')
    mids=sorted(int(r['id']) for r in json.loads((root/'data/matches.json').read_text()))
    inputs={};rows=[];audit=[];context_values=[];applied_values=[]
    lower=np.array(bundle['training_feature_min']);upper=np.array(bundle['training_feature_max'])
    columns=selected['columns']
    for mid in mids:
        for suffix in ('match.json','dynamic_events.csv','tracking_extrapolated.jsonl'):
            path=match_path(root,mid,suffix);relative=str(path.relative_to(root));checksum=digest(path)
            if receipt['files'].get(relative,{}).get('sha256')!=checksum:raise ValueError(f'Unverified input {relative}')
            inputs[relative]=checksum
        meta=json.loads(match_path(root,mid,'match.json').read_text());shots=shot_events(read_events(root,mid))
        wanted={(int(s['period']),int(s['frame_end'])-1) for s in shots};frames={}
        with match_path(root,mid,'tracking_extrapolated.jsonl').open() as f:
            for line in f:
                frame=json.loads(line);key=(frame.get('period'),frame['frame'])
                if key in wanted:frames[key]=frame
        for shot in shots:
            base=geometry(float(meta['pitch_length'])/2-float(shot['x_end']),float(shot['y_end']))
            original=float(probability(bundle['fallback_geometry'],base[None])[0]);value=original
            reason=None;outside=[];context=False;applied=False
            try:
                vector=skillcorner_features(shot,frames.get((int(shot['period']),int(shot['frame_end'])-1)),meta)
                if not np.allclose(vector[:8],base):raise ValueError('Transfer geometry differs from fallback')
                context=True;context_values.append(vector)
                outside=[FEATURES[i] for i in columns if vector[i]<lower[i]-1e-7 or vector[i]>upper[i]+1e-7]
                if outside:reason='outside_selected_training_feature_range'
                else:
                    value=float(probability(selected['fit'],vector[None,columns])[0]);applied=True;applied_values.append(vector)
            except MissingContext as error:reason=str(error)
            source=f'statsbomb_spatial_context_v1:{selection["models_sha256"]}' if applied else f'statsbomb_geometry_v1:{bundle["fallback_model_sha256"]}'
            rows.append({'match_id':mid,'event_id':shot['event_id'],'period':int(shot['period']),'frame':int(shot['frame_end']),
                'team_id':int(shot['team_id']),'xg':value,'xg_source':source})
            audit.append({**rows[-1],'geometry_xg':original,'context_available':context,'new_model_applied':applied,
                'fallback_reason':reason,'outside_training_features':outside})
        print(f'SkillCorner {mid}: scored {len(shots)} shots',flush=True)
    output.mkdir(parents=True)
    with (output/'shot_xg.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'shot_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    def distribution(values):
        a=np.asarray(values,float).reshape(-1,len(FEATURES))
        if not len(a):return {}
        return {name:{'mean':float(a[:,i].mean()),'median':float(np.median(a[:,i])),
            'statsbomb_train_mean':bundle['training_feature_mean'][i],
            'standardized_mean_difference':float((a[:,i].mean()-bundle['training_feature_mean'][i])/max(bundle['training_feature_std'][i],1e-8)),
            'fraction_outside_train_q01_q99':float(((a[:,i]<bundle['training_feature_q01'][i])|(a[:,i]>bundle['training_feature_q99'][i])).mean())}
            for i,name in enumerate(FEATURES)}
    result={'selected_family':family,'n_shots':len(rows),'n_context_available':sum(r['context_available'] for r in audit),
        'n_new_model_applied':sum(r['new_model_applied'] for r in audit),
        'fallback_reasons':dict(Counter(r['fallback_reason'] for r in audit if not r['new_model_applied'])),
        'mean_xg':float(np.mean([r['xg'] for r in rows])),'mean_old_geometry_xg':float(np.mean([r['geometry_xg'] for r in audit])),
        'mean_absolute_xg_change':float(np.mean([abs(r['xg']-r['geometry_xg']) for r in audit])),
        'feature_distribution_all_eligible':distribution(context_values),'feature_distribution_applied':distribution(applied_values),
        'skillcorner_revision':REVISION,'statsbomb_revision':SB_REVISION,'inputs':inputs,'model_sha256':selection['models_sha256'],
        'labels_sha256':digest(output/'shot_xg.csv'),'audit_sha256':digest(output/'shot_audit.json'),
        'source_sha256':source_hashes(),'source':'Experimental hybrid estimated xG; not official StatsBomb or SkillCorner xG',
        'status':'Transfer coverage and feature shift audited; SkillCorner probability calibration remains unverified',
        'fallback_policy':'Frozen geometric teacher if context is absent, misaligned or outside selected feature min/max on StatsBomb training shots; bounds are not a calibration guarantee',
        'existing_epv_labels_replaced':False}
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--root',default='data/statsbomb');s.add_argument('--old-model',default='outputs/xg_v1');s.add_argument('--out',required=True)
    s=sub.add_parser('select');s.add_argument('--data',default='outputs/contextual_xg_data_v1');s.add_argument('--old-model',default='outputs/xg_v1');s.add_argument('--protocol',default='configs/contextual_xg_v1.json');s.add_argument('--out',required=True)
    s=sub.add_parser('evaluate');s.add_argument('--data',default='outputs/contextual_xg_data_v1');s.add_argument('--models',default='outputs/contextual_xg_v1');s.add_argument('--out',required=True)
    s=sub.add_parser('transfer');s.add_argument('--root',default='data/skillcorner');s.add_argument('--models',default='outputs/contextual_xg_v1');s.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.root,a.old_model,a.out);print(json.dumps(r['by_split'],indent=2))
    elif a.command=='select':r=select(a.data,a.old_model,a.protocol,a.out);print('Selected:',r['selected_family'])
    elif a.command=='evaluate':r=evaluate(a.data,a.models,a.out);print(json.dumps(r['metrics_on_same_context_shots'],indent=2))
    else:r=transfer(a.root,a.models,a.out);print(json.dumps({k:v for k,v in r.items() if k.startswith('n_') or k=='fallback_reasons'},indent=2))


if __name__=='__main__':main()
