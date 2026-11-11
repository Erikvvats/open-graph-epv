"""Pinned, checksummed downloads, including Git LFS tracking objects."""
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "4340d274572876239c154c90bc507a9b3250a656"
REPOSITORY = "SkillCorner/opendata"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"
MEDIA = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root, relative):
    """Never overwrite an existing file that lacks matching provenance."""
    root = Path(root)
    target = root / relative
    manifest_path = root / "download_manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                {"repository": REPOSITORY, "revision": REVISION, "files": {}})
    if manifest["revision"] != REVISION:
        raise ValueError("Data directory belongs to another revision; choose a new directory")
    record = manifest["files"].get(relative)
    if target.exists():
        if record and digest(target) == record["sha256"]:
            return target
        raise ValueError(f"Unverified existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    expected_hash = expected_size = None
    url = RAW + relative
    try:
        with urlopen(url, timeout=60) as response:
            prefix = response.read(256)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                pointer = (prefix + response.read()).decode().splitlines()
                expected_hash = next(s.split("sha256:")[1] for s in pointer if s.startswith("oid "))
                expected_size = int(next(s.split()[1] for s in pointer if s.startswith("size ")))
            else:
                with temporary.open("wb") as out:
                    out.write(prefix)
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        out.write(block)
        if expected_hash:
            url = MEDIA + relative
            with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
            if temporary.stat().st_size != expected_size or digest(temporary) != expected_hash:
                raise ValueError("Git LFS object failed size/SHA256 verification")
        checksum = digest(temporary)
        temporary.replace(target)
        manifest["files"][relative] = {"sha256": checksum, "bytes": target.stat().st_size, "url": url}
        temp_manifest = manifest_path.with_suffix(".partial")
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temp_manifest.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download(root, match_ids, tracking=False):
    fetch(root, "LICENSE")
    fetch(root, "README.md")
    matches = json.loads(fetch(root, "data/matches.json").read_text())
    available = {int(m["id"]) for m in matches}
    requested = sorted(available if match_ids == ["all"] else {int(m) for m in match_ids})
    if not set(requested) <= available:
        raise ValueError("Requested match not in the pinned upstream match catalogue")
    for match_id in requested:
        for suffix in ("match.json", "dynamic_events.csv", "phases_of_play.csv"):
            fetch(root, f"data/matches/{match_id}/{match_id}_{suffix}")
        if tracking:
            fetch(root, f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl")
        print(f"Downloaded and verified match {match_id}", flush=True)


def match_path(root, match_id, suffix):
    return Path(root) / "data" / "matches" / str(match_id) / f"{match_id}_{suffix}"


def read_events(root, match_id):
    with match_path(root, match_id, "dynamic_events.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def shot_events(rows):
    """Only possession-ending shots; lead_to_shot is a future-derived flag."""
    shots = [r for r in rows if r["event_type"] == "player_possession" and r["end_type"] == "shot"]
    keys = [(r["match_id"], r["event_id"]) for r in shots]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate shot event IDs")
    return shots


def audit(root, match_ids):
    result = {"revision": REVISION, "matches": [], "target_ready": False}
    for match_id in match_ids:
        rows = read_events(root, match_id)
        columns = list(rows[0]) if rows else []
        result["matches"].append({
            "match_id": int(match_id), "events": len(rows), "shot_markers": len(shot_events(rows)),
            "xg_columns": [c for c in columns if c.lower() in ("xg", "shot_xg", "expected_goals")],
            "other_model_columns": [c for c in columns if "epv" in c or c.startswith("xshot")],
        })
    result["target_note"] = "An explicit, complete shot-xG CSV is required. xshot and EPV are not shot xG."
    return result
