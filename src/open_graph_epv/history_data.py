"""Append strictly past, same-period trajectories to frozen snapshot rows."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .data import frame_graph
from .source import digest,match_path

OFFSETS=(-20,-10,-5,0)


def aligned_history(current,buffer,metadata,offsets=OFFSETS):
    """Use the CURRENT possession team's orientation and CURRENT player slots."""
    if offsets[-1]!=0 or any(t>0 for t in offsets) or sorted(set(offsets))!=list(offsets):raise ValueError('Invalid past offsets')
    ids=[p['player_id'] for p in current['player_data']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate current player ID')
    all_x=np.zeros((len(offsets),23,10),dtype=np.float32)
    all_masks=np.zeros((len(offsets),23),dtype=bool)
    for index,offset in enumerate(offsets):
        f=buffer.get(current['frame']+offset)
        if f is None or f.get('period')!=current['period'] or not isinstance(f.get('ball_data'),dict):continue
        # Historical ball possession may differ. All features must remain
        # relative to the team making the prediction at the current frame.
        forced={**f,'possession':current['possession']}
        previous=buffer.get(f['frame']-1)
        if previous is not None and not isinstance(previous.get('ball_data'),dict):previous={**previous,'ball_data':{}}
        graph=frame_graph(forced,previous,metadata,include_extrapolated=False)
        if graph is None:continue
        nodes,mask,_=graph
        mapping={p['player_id']:i+1 for i,p in enumerate(f['player_data'])}
        all_x[index,0]=nodes[0];all_masks[index,0]=mask[0]
        for target,pid in enumerate(ids,1):
            if pid in mapping:
                source=mapping[pid];all_x[index,target]=nodes[source];all_masks[index,target]=mask[source]
    return all_x,all_masks


def prepare(root,data,output):
    root,data,output=Path(root),Path(data),Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((data/'manifest.json').read_text())
    if digest(data/'graphs.npz')!=manifest['graphs_sha256']:raise ValueError('Changed snapshot dataset')
    with np.load(data/'graphs.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    x=np.zeros((len(arrays['y']),len(OFFSETS),23,10),np.float32)
    masks=np.zeros(x.shape[:3],bool);seen=np.zeros(len(x),bool);audit=[];inputs={}
    for mid in np.unique(arrays['match_id']):
        mid=int(mid)
        for suffix in ('match.json','tracking_extrapolated.jsonl'):
            path=match_path(root,mid,suffix);relative=str(path.relative_to(root));checksum=digest(path)
            if manifest['inputs'].get(relative)!=checksum:raise ValueError(f'Changed source {relative}')
            inputs[relative]=checksum
        meta=json.loads(match_path(root,mid,'match.json').read_text())
        indices=np.flatnonzero(arrays['match_id']==mid);lookup={int(arrays['frame'][i]):i for i in indices}
        if len(lookup)!=len(indices):raise ValueError('Duplicate snapshot key')
        buffer={};last=None
        with match_path(root,mid,'tracking_extrapolated.jsonl').open() as stream:
            for line in stream:
                current=json.loads(line);frame=current['frame']
                if last is not None and frame<=last:raise ValueError('Tracking is not strictly ordered')
                buffer[frame]=current
                for old in list(buffer):
                    if old<frame+min(OFFSETS)-1:del buffer[old]
                if frame in lookup:
                    i=lookup[frame];hx,hm=aligned_history(current,buffer,meta)
                    if not np.array_equal(hm[-1],arrays['mask'][i]) or not np.array_equal(hx[-1],arrays['x'][i]):
                        raise ValueError(f'Current snapshot changed at {mid}:{frame}')
                    x[i]=hx;masks[i]=hm;seen[i]=True
                last=frame
        if not seen[indices].all():raise ValueError(f'Missing snapshot frames for {mid}')
        audit.append({'match_id':mid,'n_snapshots':len(indices),
            'observed_nodes_by_offset':masks[indices].sum((0,2)).tolist()})
        print(f'{mid}: {len(indices)} exact snapshot anchors with aligned histories',flush=True)
    if not seen.all():raise ValueError('Incomplete history')
    output.mkdir(parents=True)
    np.savez_compressed(output/'history.npz',x=x,mask=masks,**{k:arrays[k] for k in ('y','match_id','frame')})
    result={'offset_frames':list(OFFSETS),'offset_seconds':[v/10 for v in OFFSETS],
        'snapshot_sha256':manifest['graphs_sha256'],'history_sha256':digest(output/'history.npz'),
        'n_graphs':len(x),'target':manifest['target'],'horizon_seconds':manifest['horizon_seconds'],
        'xg_sha256':manifest['xg_sha256'],'xg_sources':manifest['xg_sources'],'audit':audit,'inputs':inputs,
        'alignment':'Current player IDs, current possession orientation; exact past frames only; no cross-period history',
        'source_sha256':{n:digest(Path(__file__).with_name(n)) for n in ('history_data.py','data.py')}}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
