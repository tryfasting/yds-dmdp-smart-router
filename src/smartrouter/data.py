"""
Service-log merge with explicit keys and cardinality checks.

Reproduces the join logic of the team's merge notebook (01_데이터 병합.ipynb, cells 6/15/20/23)
without its dtype-dependent drop_duplicates step. The historical full_merged_v3.csv is treated as a
stored artifact (hash-checked), not regenerated: its later cleaning notebook input (v2) is missing.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

USAGE_FILE = "1. 사용자별 사용량 데이터.json"
CLIENT_FILE = "2. 사용자 클라이언트 데이터.json"
EVENT_FILE = "3. 이벤트 데이터.json"

# SHA-256 of the stored historical artifact (bookend-yeardream-proj/data/processed/full_merged_v3.csv).
V3_SHA256 = "06de38eb5dad4de1d8553329cc44a4f5a1ad5609df6339213f06d25b13c2c01b"


@dataclass(frozen=True)
class MergeReport:
    usage_rows: int
    usage_missing_id: int
    client_rows: int
    client_unique_ids: int
    event_rows: int
    event_unique_distinct_ids: int
    user_merged_rows: int
    user_merged_null_key: int
    user_merged_duplicate_keys: int
    event_null_key: int
    merged_rows: int
    events_matched: int
    events_unmatched: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_oid(value: Any) -> Optional[str]:
    """MongoDB export ids arrive as {"$oid": "..."}; plain strings are kept."""
    if isinstance(value, dict):
        return value.get("$oid")
    if isinstance(value, str):
        return value
    return None


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    usage = pd.read_json(raw_dir / USAGE_FILE)
    clients = pd.read_json(raw_dir / CLIENT_FILE, lines=True)
    events = pd.read_json(raw_dir / EVENT_FILE, lines=True)
    return usage, clients, events


def merge_logs(
    usage: pd.DataFrame, clients: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, MergeReport]:
    usage = usage.copy()
    events = events.copy()
    usage["user_key"] = usage["_id"].map(extract_oid).astype("object")
    events["user_key"] = events["user_id"].map(extract_oid).astype("object")

    # 1) usage <-> client attributes: each user id must appear at most once on each side.
    user_merged = pd.merge(
        usage, clients, left_on="user_key", right_on="distinct_id", how="outer", validate="one_to_one"
    )
    user_merged["merge_key"] = user_merged["user_key"].fillna(user_merged["distinct_id"])
    users_keyed = user_merged.dropna(subset=["merge_key"])

    # 2) events -> users: logged-in user id first, anonymous distinct_id otherwise.
    events["merge_key"] = events["user_key"].fillna(events["distinct_id"])
    merged = pd.merge(
        events,
        users_keyed.drop(columns=["user_key"]),
        on="merge_key",
        how="left",
        suffixes=("_event", "_user"),
        validate="many_to_one",
        indicator=True,
    )
    if len(merged) != len(events):
        raise AssertionError("Event rows changed during left join")

    report = MergeReport(
        usage_rows=len(usage),
        usage_missing_id=int(usage["user_key"].isna().sum()),
        client_rows=len(clients),
        client_unique_ids=int(clients["distinct_id"].nunique()),
        event_rows=len(events),
        event_unique_distinct_ids=int(events["distinct_id"].nunique()),
        user_merged_rows=len(user_merged),
        user_merged_null_key=int(user_merged["merge_key"].isna().sum()),
        user_merged_duplicate_keys=int(users_keyed["merge_key"].duplicated().sum()),
        event_null_key=int(events["merge_key"].isna().sum()),
        merged_rows=len(merged),
        events_matched=int((merged["_merge"] == "both").sum()),
        events_unmatched=int((merged["_merge"] == "left_only").sum()),
    )
    return merged.drop(columns=["_merge"]), report


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_v3(path: Path) -> dict[str, Any]:
    """Hash-check the stored v3 artifact and report its shape (no content is printed)."""
    digest = sha256_file(path)
    shape = pd.read_csv(path, usecols=[0]).shape[0], len(pd.read_csv(path, nrows=0).columns)
    return {"path": str(path), "sha256": digest, "matches_recorded_hash": digest == V3_SHA256, "rows": shape[0], "columns": shape[1]}


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
