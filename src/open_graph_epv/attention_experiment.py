"""Fixed-protocol attention/history experiment; selection precedes test scoring."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .attention_model import AttentionEPV
from .evaluation import verify_selection, value_metrics, paired_match_difference
from .source import digest
from .training import load_epv, predict as snapshot_predict, validate_split

MODES=('snapshot','history','repeat')
SOURCES=('attention_model.py','attention_experiment.py','history_data.py','data.py')


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def verify_history(data):
    data=Path(data);manifest=json.loads((data/'manifest.json').read_text())
    if digest(data/'history.npz')!=manifest['history_sha256']:raise ValueError('History dataset changed')
    for name,checksum in manifest['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=checksum:raise ValueError(f'Preparation source changed: {name}')
    return manifest


@torch.no_grad()
def predict(model,x,mask,batch_size=128):
    model.eval()
    return torch.cat([model.symmetric(x[i:i+batch_size],mask[i:i+batch_size]) for i in range(0,len(x),batch_size)]).numpy()


def read_protocol(path):
    protocol=json.loads(Path(path).read_text())
    if protocol['source_sha256']!=source_hashes():raise ValueError('Experiment source differs from registered protocol')
    return protocol


def train(data,protocol_path,mode,seed,output):
    data,output=Path(data),Path(output)
    if output.exists():raise FileExistsError(output)
    protocol=read_protocol(protocol_path);manifest=verify_history(data)
    if manifest['history_sha256']!=protocol['history_sha256']:raise ValueError('Unregistered history data')
    if mode not in protocol['modes'] or seed not in protocol['seeds']:raise ValueError('Unregistered run')
    split=protocol['split']
    with np.load(data/'history.npz',allow_pickle=False) as z:
        validate_split(split,np.unique(z['match_id']))
        tr=np.isin(z['match_id'],split['train']);va=np.isin(z['match_id'],split['val'])
        x,m,y=[torch.from_numpy(z[k][tr].copy()) for k in ('x','mask','y')]
        xv,mv=[torch.from_numpy(z[k][va].copy()) for k in ('x','mask')];yv=z['y'][va].copy()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite training inputs')
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(protocol['cpu_threads'])
    model=AttentionEPV(mode=mode,**protocol['model'])
    count=sum(p.numel() for p in model.parameters())
    if count!=protocol['transformer_parameters']:raise ValueError('Parameter count changed')
    with torch.no_grad():
        model.head[-1].weight.mul_(.01)
        model.head[-1].bias.fill_(float(torch.atanh(y.mean().clamp(-.99,.99))))
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    prior=float(y.mean());history=[];best=float('inf');stale=0;state=None;start=time.perf_counter()
    output.mkdir(parents=True)
    for epoch in range(protocol['max_epochs']):
        model.train();epoch_start=time.perf_counter()
        for indices in torch.randperm(len(x)).split(protocol['batch_size']):
            batch=x[indices].clone()
            sign=torch.where(torch.rand(len(batch))<.5,-1.,1.)[:,None,None]
            batch[...,1]*=sign;batch[...,3]*=sign
            loss=(model(batch,m[indices])-y[indices]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        p=predict(model,xv,mv);score=value_metrics(yv,p,prior)
        history.append({'epoch':epoch+1,'seconds':time.perf_counter()-epoch_start,**score})
        if score['mse']<best:best=score['mse'];state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        progress={'mode':mode,'seed':seed,'epoch':epoch+1,'best_validation_rmse':best**.5,
                  'elapsed_seconds':time.perf_counter()-start,'history':history,'test_evaluated':False}
        temporary=output/'progress.partial';temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(output/'progress.json')
        print(f'{mode} seed={seed} epoch={epoch+1} val_rmse={score["rmse"]:.6f} seconds={history[-1]["seconds"]:.1f}',flush=True)
        if stale>=protocol['patience']:break
    torch.save({'config':model.config,'state_dict':state},output/'model.pt')
    report={'mode':mode,'seed':seed,'config':model.config,'parameter_count':count,'history':history,
        'best_validation':min(history,key=lambda r:r['mse']),'training_mean':prior,'n_train':len(y),'n_val':len(yv),
        'split':split,'history_sha256':manifest['history_sha256'],'protocol_sha256':digest(protocol_path),
        'source_sha256':source_hashes(),'model_sha256':digest(output/'model.pt'),'test_evaluated':False,
        'elapsed_seconds':time.perf_counter()-start,'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    (output/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def load(directory):
    directory=Path(directory);report=json.loads((directory/'training.json').read_text())
    if digest(directory/'model.pt')!=report['model_sha256'] or report['source_sha256']!=source_hashes():
        raise ValueError('Checkpoint or source changed')
    payload=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    if payload['config']!=report['config']:raise ValueError('Model config mismatch')
    model=AttentionEPV(**payload['config']);model.load_state_dict(payload['state_dict'])
    return model.eval()


def select(data,runs,protocol_path,original_data,original_runs,original_selection_path,output):
    data,runs,output=Path(data),Path(runs),Path(output)
    if output.exists():raise FileExistsError(output)
    protocol=read_protocol(protocol_path);manifest=verify_history(data)
    original=json.loads(Path(original_selection_path).read_text());verify_selection(original_data,original_runs,original)
    if manifest['history_sha256']!=protocol['history_sha256'] or original['split']!=protocol['split']:
        raise ValueError('Dataset or split mismatch')
    if digest(original_selection_path)!=protocol['original_selection_sha256']:raise ValueError('Original selection changed')
    torch.set_num_threads(protocol['cpu_threads'])
    with np.load(data/'history.npz',allow_pickle=False) as z:
        use=np.isin(z['match_id'],protocol['split']['val'])
        x,m=[torch.from_numpy(z[k][use]) for k in ('x','mask')];y=z['y'][use]
    records=[];metrics={};prior=original['training_mean']
    for mode in MODES:
        predictions=[];individual=[]
        for seed in protocol['seeds']:
            name=f'{mode}_seed{seed}';directory=runs/name
            report=json.loads((directory/'training.json').read_text())
            if (report['protocol_sha256']!=digest(protocol_path) or report['mode']!=mode or report['seed']!=seed
                or report['history_sha256']!=manifest['history_sha256'] or report['split']!=protocol['split']):
                raise ValueError('Inconsistent attention run')
            p=predict(load(directory),x,m);predictions.append(p)
            individual.append({'seed':seed,**value_metrics(y,p,prior)})
            records.append({'name':name,'mode':mode,'seed':seed,'model_sha256':digest(directory/'model.pt'),
                            'report_sha256':digest(directory/'training.json')})
        metrics[mode]={'ensemble':value_metrics(y,np.mean(predictions,axis=0),prior),'individual':individual}
    for family in ('gnn','ball','set'):
        p=[snapshot_predict(load_epv(Path(original_runs)/r['name']),x[:,-1],m[:,-1]) for r in original['models'] if r['family']==family]
        metrics[family]={'ensemble':value_metrics(y,np.mean(p,axis=0),prior)}
    eligible=('gnn','snapshot','history')  # Repeat is a diagnostic, not a historical-information candidate.
    winner=min(eligible,key=lambda name:metrics[name]['ensemble']['mse'])
    result={'selected_family':winner,'selection_eligible':list(eligible),'validation':metrics,'records':records,
        'protocol':protocol,'protocol_sha256':digest(protocol_path),'history_manifest_sha256':digest(data/'manifest.json'),
        'original_selection_sha256':digest(original_selection_path),'training_mean':prior,'source_sha256':source_hashes(),
        'test_evaluated':False}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def evaluate(data,runs,selection_path,original_data,original_runs,original_selection_path,output):
    data,runs,output=Path(data),Path(runs),Path(output)
    if output.exists():raise FileExistsError(output)
    selection=json.loads(Path(selection_path).read_text());manifest=verify_history(data)
    if selection['source_sha256']!=source_hashes() or digest(data/'manifest.json')!=selection['history_manifest_sha256']:
        raise ValueError('Frozen source/data changed')
    original=json.loads(Path(original_selection_path).read_text());verify_selection(original_data,original_runs,original)
    if digest(original_selection_path)!=selection['original_selection_sha256']:raise ValueError('Original selection changed')
    for r in selection['records']:
        directory=runs/r['name']
        if digest(directory/'model.pt')!=r['model_sha256'] or digest(directory/'training.json')!=r['report_sha256']:
            raise ValueError('Run changed after validation selection')
    torch.set_num_threads(selection['protocol']['cpu_threads'])
    with np.load(data/'history.npz',allow_pickle=False) as z:
        use=np.isin(z['match_id'],selection['protocol']['split']['test'])
        x,m=[torch.from_numpy(z[k][use]) for k in ('x','mask')]
        y,groups,frames=[z[k][use] for k in ('y','match_id','frame')]
    prior=selection['training_mean'];predictions={};metrics={}
    for mode in MODES:
        records=[r for r in selection['records'] if r['mode']==mode]
        values=[predict(load(runs/r['name']),x,m) for r in records]
        predictions[mode]=np.mean(values,axis=0)
        metrics[mode]={'ensemble':value_metrics(y,predictions[mode],prior),
            'individual':[{'seed':r['seed'],**value_metrics(y,p,prior)} for r,p in zip(records,values)]}
    for family in ('gnn','ball','set'):
        records=[r for r in original['models'] if r['family']==family]
        values=[snapshot_predict(load_epv(Path(original_runs)/r['name']),x[:,-1],m[:,-1]) for r in records]
        predictions[family]=np.mean(values,axis=0)
        metrics[family]={'ensemble':value_metrics(y,predictions[family],prior),
            'individual':[{'seed':r['seed'],**value_metrics(y,p,prior)} for r,p in zip(records,values)]}
    contrasts=[('snapshot','gnn'),('history','snapshot'),('history','repeat'),('repeat','snapshot')]
    result={'selected_family':selection['selected_family'],'n_test':len(y),'n_matches':len(np.unique(groups)),
        'metrics':metrics,'training_mean_reference':value_metrics(y,np.full(len(y),prior),prior),
        'comparisons':{f'{a}_minus_{b}':paired_match_difference(y,predictions[a],predictions[b],groups) for a,b in contrasts},
        'per_match':[{'match_id':int(mid),'metrics':{name:value_metrics(y[groups==mid],p[groups==mid],prior) for name,p in predictions.items()}}
                     for mid in np.unique(groups)],
        'selection':selection,'selection_sha256':digest(selection_path),'history_audit':manifest['audit'],
        'limitations':['Previously examined three-match test set: exploratory architectural comparison.',
            'Transferred geometric xG labels; teacher calibration on SkillCorner is unverified.',
            'Common optimizer is a controlled budget, not exhaustive tuning of every architecture.',
            'Temporal encoder uses aligned individual trajectories before spatial attention, not a full historical interaction graph.',
            'Repeat control retains historical visibility for currently visible players; contrast isolates trajectory values conditional on coverage.',
            'Only three match clusters; descriptive intervals omit retraining and xG-source uncertainty.']}
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',target=y,match_id=groups,frame=frames,**predictions)
    result['predictions_sha256']=digest(output/'predictions.npz')
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--mode',choices=MODES,required=True);t.add_argument('--seed',type=int,required=True)
    t.add_argument('--data',default='outputs/history_v1');t.add_argument('--protocol',default='configs/attention_history_v1.json');t.add_argument('--out',required=True)
    for name in ('select','evaluate'):
        t=sub.add_parser(name);t.add_argument('--data',default='outputs/history_v1');t.add_argument('--runs',default='outputs/attention_history_v1')
        t.add_argument('--original-data',default='outputs/graphs_v1');t.add_argument('--original-runs',default='outputs')
        t.add_argument('--original-selection',default='outputs/selection_v1.json');t.add_argument('--out',required=True)
        if name=='select':t.add_argument('--protocol',default='configs/attention_history_v1.json')
        else:t.add_argument('--selection',required=True)
    a=p.parse_args()
    if a.command=='train':train(a.data,a.protocol,a.mode,a.seed,a.out)
    elif a.command=='select':
        r=select(a.data,a.runs,a.protocol,a.original_data,a.original_runs,a.original_selection,a.out)
        print('Selected by validation:',r['selected_family'])
    else:
        r=evaluate(a.data,a.runs,a.selection,a.original_data,a.original_runs,a.original_selection,a.out)
        print(json.dumps({name:v['ensemble']['rmse'] for name,v in r['metrics'].items()},indent=2))


if __name__=='__main__':main()
