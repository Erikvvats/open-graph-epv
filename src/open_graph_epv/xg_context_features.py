"""Shared pre-shot spatial features; no provider xG or post-shot information."""
import math
import numpy as np
from .xg import GEOMETRY,geometry

KEEPER=['keeper_distance_to_shooter','keeper_depth','keeper_lateral_offset_from_shot_line',
        'keeper_in_goal_triangle','keeper_distance_to_goal_center']
DEFENSE=['nearest_outfield_defender_distance','defenders_within_3m','defenders_in_goal_triangle',
         'observed_outfield_defenders','smooth_defender_pressure']
FEATURES=GEOMETRY+KEEPER+DEFENSE
FAMILIES={'geometry':list(range(8)),'keeper':list(range(13)),'context':list(range(18))}


class MissingContext(ValueError):pass


def finite_xy(value):
    return isinstance(value,(list,tuple,np.ndarray)) and len(value)>=2 and all(
        isinstance(v,(float,int,np.floating,np.integer)) and np.isfinite(v) for v in value[:2])


def in_triangle(shooter,positions,goal_x):
    positions=np.asarray(positions,float).reshape(-1,2)
    t=(positions[:,0]-shooter[0])/(goal_x-shooter[0])
    return (t>0)&(t<1)&(positions[:,1]>=shooter[1]+t*(-3.66-shooter[1]))&(positions[:,1]<=shooter[1]+t*(3.66-shooter[1]))


def spatial_features(shooter,keeper,defenders,goal_x):
    s,k=np.asarray(shooter,float),np.asarray(keeper,float)
    d=np.asarray(defenders,float).reshape(-1,2)
    if s.shape!=(2,) or k.shape!=(2,) or not len(d) or not np.isfinite(np.r_[s,k,d.ravel(),goal_x]).all():
        raise MissingContext('invalid_spatial_context')
    dx=goal_x-s[0]
    if dx<=0:raise MissingContext('shot_on_or_behind_goal_line')
    goal=np.array([goal_x,0.]);direction=goal-s;distance=np.linalg.norm(direction)
    lateral=abs(direction[0]*(k[1]-s[1])-direction[1]*(k[0]-s[0]))/distance
    distances=np.linalg.norm(d-s,axis=1)
    extra=[float(np.linalg.norm(k-s)),goal_x-k[0],lateral,float(in_triangle(s,[k],goal_x)[0]),
           float(np.linalg.norm(k-goal)),float(distances.min()),float((distances<3).sum()),
           float(in_triangle(s,d,goal_x).sum()),float(len(d)),float(np.exp(-distances/3).sum())]
    return np.r_[geometry(dx,s[1]),extra]


def sb_xy(location):
    if not finite_xy(location):raise MissingContext('invalid_freeze_frame_location')
    return np.array([(float(location[0])-60)*105/120,(float(location[1])-40)*68/80])


def statsbomb_features(event):
    shooter=sb_xy(event.get('location'))
    freeze=event.get('shot',{}).get('freeze_frame') or []
    if not freeze:raise MissingContext('missing_freeze_frame')
    keepers=[];defenders=[];ids=set()
    for player in freeze:
        pid=player.get('player',{}).get('id')
        if pid is not None:
            if pid in ids:raise MissingContext('duplicate_freeze_frame_player')
            ids.add(pid)
        if player.get('teammate') not in (True,False):raise MissingContext('unknown_team_relation')
        if player['teammate']:continue
        xy=sb_xy(player.get('location'))
        position=player.get('position',{}).get('name')
        if not position:raise MissingContext('unknown_opponent_position')
        (keepers if position=='Goalkeeper' else defenders).append(xy)
    if len(keepers)!=1:raise MissingContext('missing_or_ambiguous_keeper')
    if not defenders:raise MissingContext('no_observed_outfield_defenders')
    return spatial_features(shooter,keepers[0],defenders,52.5)


def detected(obj):
    return isinstance(obj,dict) and obj.get('is_detected') is True and finite_xy([obj.get('x'),obj.get('y')])


def skillcorner_features(shot,frame,metadata):
    if shot.get('is_player_possession_end_matched')!='True':raise MissingContext('unmatched_shot_endpoint')
    if frame is None or frame.get('period')!=int(shot['period']) or frame['frame']!=int(shot['frame_end'])-1:
        raise MissingContext('missing_exact_pre_shot_frame')
    team=int(shot['team_id']);pid=int(shot['player_id']);period=int(shot['period'])
    home=metadata['home_team']['id'];away=metadata['away_team']['id']
    if team not in (home,away) or period not in (1,2):raise MissingContext('invalid_team_or_period')
    side=metadata['home_team_side'][period-1]
    if side not in ('left_to_right','right_to_left'):raise MissingContext('invalid_attacking_direction')
    direction=(1 if side=='left_to_right' else -1)*(1 if team==home else -1)
    roster={p['id']:p for p in metadata['players']};tracked={p['player_id']:p for p in frame['player_data']}
    if len(tracked)!=len(frame['player_data']):raise MissingContext('duplicate_tracked_player')
    if pid not in roster or roster[pid]['team_id']!=team:raise MissingContext('shooter_identity_mismatch')
    player=tracked.get(pid);ball=frame.get('ball_data')
    if not detected(player):raise MissingContext('shooter_not_observed')
    if not detected(ball):raise MissingContext('ball_not_observed')
    if math.hypot(player['x']-ball['x'],player['y']-ball['y'])>4:raise MissingContext('shooter_ball_misalignment')
    # Dynamic event endpoint positions are ALREADY attack-normalized.
    shot_xy=np.array([float(shot['x_end']),float(shot['y_end'])])
    observed_xy=direction*np.array([player['x'],player['y']])
    if not np.isfinite(shot_xy).all() or np.linalg.norm(shot_xy-observed_xy)>4:
        raise MissingContext('event_tracking_shooter_misalignment')
    keepers=[];defenders=[]
    for player_id,obj in tracked.items():
        if player_id not in roster:raise MissingContext('unknown_tracked_player')
        if not detected(obj):continue
        meta=roster[player_id]
        if meta['team_id']==team:continue
        xy=direction*np.array([obj['x'],obj['y']])
        (keepers if meta['player_role']['acronym']=='GK' else defenders).append(xy)
    if len(keepers)!=1:raise MissingContext('missing_or_ambiguous_keeper')
    if not defenders:raise MissingContext('no_observed_outfield_defenders')
    return spatial_features(shot_xy,keepers[0],defenders,float(metadata['pitch_length'])/2)
