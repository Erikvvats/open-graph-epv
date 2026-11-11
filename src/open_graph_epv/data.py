"""Causal snapshot graphs and strict first-future-shot xG targets."""
import csv
import json
from pathlib import Path

import numpy as np

from .source import digest, match_path, read_events, shot_events, REVISION

FEATURES = ["x", "y", "vx", "vy", "teammate", "opponent", "ball", "keeper", "detected", "velocity_valid"]
MAX_NODES = 23
FPS = 10


def shot_template(root, match_ids, output):
    fields = ["match_id", "event_id", "period", "frame", "team_id", "xg", "xg_source"]
    with Path(output).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for match_id in match_ids:
            for r in shot_events(read_events(root, match_id)):
                writer.writerow(dict(match_id=match_id, event_id=r["event_id"], period=r["period"],
                                     frame=r["frame_end"], team_id=r["team_id"], xg="", xg_source=""))


def load_xg(path, root, match_ids):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"match_id", "event_id", "period", "frame", "team_id", "xg", "xg_source"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"xG CSV requires {sorted(required)}")
        rows = list(reader)
    supplied = {}
    for r in rows:
        key = (int(r["match_id"]), r["event_id"])
        if key in supplied:
            raise ValueError(f"Duplicate xG label: {key}")
        try:
            value = float(r["xg"])
        except ValueError as error:
            raise ValueError(f"Missing xG for {key}; missing values must not become zero") from error
        if not np.isfinite(value) or not 0 <= value <= 1 or not r["xg_source"].strip():
            raise ValueError(f"Invalid xG or missing provenance for {key}")
        supplied[key] = dict(period=int(r["period"]), frame=int(r["frame"]),
                             team_id=int(r["team_id"]), xg=value, source=r["xg_source"])
    result = {}
    expected_keys = set()
    for mid in match_ids:
        result[int(mid)] = []
        for event in shot_events(read_events(root, mid)):
            key = (int(mid), event["event_id"])
            expected_keys.add(key)
            if key not in supplied:
                raise ValueError(f"Missing shot xG: {key}")
            shot = supplied[key]
            expected = (int(event["period"]), int(event["frame_end"]), int(event["team_id"]))
            if (shot["period"], shot["frame"], shot["team_id"]) != expected:
                raise ValueError(f"xG label is not aligned to SkillCorner shot: {key}")
            result[int(mid)].append(shot)
        result[int(mid)].sort(key=lambda r: (r["period"], r["frame"]))
        times = [(r["period"], r["frame"]) for r in result[int(mid)]]
        if len(times) != len(set(times)):
            raise ValueError(f"Ambiguous simultaneous shot markers in match {mid}")
    selected = {k for k in supplied if k[0] in {int(m) for m in match_ids}}
    if selected != expected_keys:
        raise ValueError("xG CSV includes unknown shot events for selected matches")
    return result


def future_target(shots, frame, period, team_id, period_end, horizon=150):
    """Strictly future (t, t+H], same period, complete horizon required."""
    if frame + horizon > period_end:
        return None
    future = [r for r in shots if r["period"] == period and frame < r["frame"] <= frame + horizon]
    if not future:
        return 0.0
    first = min(future, key=lambda r: r["frame"])
    return first["xg"] * (1 if first["team_id"] == team_id else -1)


def valid_xy(obj):
    return all(isinstance(obj.get(k), (int, float)) and np.isfinite(obj[k]) for k in ("x", "y"))


def frame_graph(frame, previous, metadata, include_extrapolated=False):
    """Only current positions and backward velocity. No event/outcome inputs.

    A node is identified only while constructing a frame; player IDs are never
    model features. Default masks out provider-extrapolated positions.
    """
    period = frame.get("period")
    possession = frame.get("possession") or {}
    group = possession.get("group")
    if period not in (1, 2) or group not in ("home team", "away team"):
        return None
    team = metadata["home_team" if group == "home team" else "away_team"]["id"]
    side = metadata["home_team_side"][period - 1]
    if side not in ("left_to_right", "right_to_left"):
        raise ValueError("Unknown attacking direction")
    direction = (1 if side == "left_to_right" else -1) * (1 if group == "home team" else -1)
    ball = frame["ball_data"]
    if not valid_xy(ball) or (not include_extrapolated and ball.get("is_detected") is not True):
        return None
    players = {p["id"]: p for p in metadata["players"]}
    if len(frame["player_data"]) > 22:
        raise ValueError("More than 22 player records in a frame")
    continuous = (previous is not None and previous.get("period") == period
                  and frame["frame"] - previous["frame"] == 1)
    old = {p["player_id"]: p for p in previous["player_data"]} if continuous else {}
    old_ball = previous["ball_data"] if continuous else {}
    nodes = np.zeros((MAX_NODES, len(FEATURES)), dtype=np.float32)
    mask = np.zeros(MAX_NODES, dtype=bool)
    half = np.array([metadata["pitch_length"], metadata["pitch_width"]], dtype=float) / 2
    if not np.isfinite(half).all() or (half <= 0).any():
        raise ValueError("Invalid pitch dimensions")
    entries = [(ball, old_ball, 0, False, True)]
    seen = set()
    for p in frame["player_data"]:
        pid = p["player_id"]
        if pid in seen:
            raise ValueError("Duplicate tracked player")
        seen.add(pid)
        if pid not in players:
            raise ValueError(f"Unknown tracked player: {pid}")
        meta = players[pid]
        entries.append((p, old.get(pid, {}), meta["team_id"], meta["player_role"]["acronym"] == "GK", False))
    for i, (obj, prior, player_team, keeper, is_ball) in enumerate(entries):
        detected = obj.get("is_detected") is True
        if not valid_xy(obj) or (not include_extrapolated and not detected):
            continue
        xy = np.array([obj["x"], obj["y"]])
        velocity_valid = valid_xy(prior) and (include_extrapolated or prior.get("is_detected") is True)
        velocity = ((xy - np.array([prior["x"], prior["y"]])) * FPS if velocity_valid else np.zeros(2))
        # Unrealistic one-frame jumps are missing motion, not clipped evidence.
        if np.linalg.norm(velocity) > (50 if is_ball else 15):
            velocity_valid, velocity = False, np.zeros(2)
        nodes[i] = [*(direction * xy / half), *(direction * velocity / 10),
                    float(not is_ball and player_team == team), float(not is_ball and player_team != team),
                    float(is_ball), float(keeper), float(detected), float(velocity_valid)]
        mask[i] = True
    if mask.sum() < 3:
        return None
    return nodes, mask, team


def iter_graphs(root, match_id, stride=20, include_extrapolated=False):
    if stride < 1:
        raise ValueError("stride must be positive")
    metadata = json.loads(match_path(root, match_id, "match.json").read_text())
    ends = {p["period"]: p["end_frame"] for p in metadata["match_periods"]}
    previous = None
    with match_path(root, match_id, "tracking_extrapolated.jsonl").open() as f:
        for line in f:
            frame = json.loads(line)
            if previous is not None and frame["frame"] <= previous["frame"]:
                raise ValueError("Tracking frames must be strictly increasing")
            if frame["frame"] % stride == 0:
                graph = frame_graph(frame, previous, metadata, include_extrapolated)
                if graph is not None:
                    yield graph, frame["frame"], frame["period"], ends[frame["period"]]
            previous = frame


def prepare(root, match_ids, xg_csv, output, stride=20, horizon_seconds=15, include_extrapolated=False):
    if horizon_seconds <= 0 or not np.isfinite(horizon_seconds):
        raise ValueError("horizon_seconds must be finite and positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    shots = load_xg(xg_csv, root, match_ids)  # Validate every label before creating output.
    x, masks, targets, games, frames = [], [], [], [], []
    for mid in match_ids:
        for (nodes, mask, team), frame, period, end in iter_graphs(root, mid, stride, include_extrapolated):
            target = future_target(shots[int(mid)], frame, period, team, end, round(horizon_seconds * FPS))
            if target is not None:
                x.append(nodes); masks.append(mask); targets.append(target); games.append(int(mid)); frames.append(frame)
    if not x:
        raise ValueError("No eligible graphs")
    output.mkdir(parents=True)
    np.savez_compressed(output / "graphs.npz", x=np.array(x), mask=np.array(masks),
                        y=np.array(targets, dtype=np.float32), match_id=np.array(games), frame=np.array(frames))
    manifest = {"revision": REVISION, "features": FEATURES, "target": "signed first future shot xG",
                "horizon_seconds": horizon_seconds, "stride_frames": stride, "fps": FPS,
                "include_extrapolated": include_extrapolated, "xg_sha256": digest(xg_csv),
                "xg_sources": sorted({s["source"] for v in shots.values() for s in v}),
                "matches": sorted(set(games)), "n_graphs": len(x), "inputs": {}}
    for mid in match_ids:
        for suffix in ("match.json", "dynamic_events.csv", "tracking_extrapolated.jsonl"):
            path = match_path(root, mid, suffix)
            manifest["inputs"][str(path.relative_to(root))] = digest(path)
    manifest["graphs_sha256"] = digest(output / "graphs.npz")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
