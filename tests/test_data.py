import pandas as pd
import pytest

from smartrouter.data import extract_oid, merge_logs


def synthetic_logs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usage = pd.DataFrame({"_id": [{"$oid": "u1"}, {"$oid": "u2"}, None], "count": [3, 1, 2]})
    clients = pd.DataFrame({"distinct_id": ["u1", "anon9"], "properties": [{"os": "mac"}, {"os": "win"}]})
    events = pd.DataFrame(
        {
            "event_name": ["run_paraphrasing", "selected_paraphrasing", "run_paraphrasing", "run_paraphrasing"],
            "distinct_id": ["dev1", "anon9", "dev2", "ghost"],
            "user_id": [{"$oid": "u1"}, None, {"$oid": "u2"}, None],
            "time": [1, 2, 3, 4],
        }
    )
    return usage, clients, events


def test_extract_oid():
    assert extract_oid({"$oid": "abc"}) == "abc"
    assert extract_oid("plain") == "plain"
    assert extract_oid(None) is None


def test_merge_keeps_event_rows_and_reports_matches():
    merged, report = merge_logs(*synthetic_logs())
    assert len(merged) == 4
    assert report.user_merged_rows == 4  # u1 (both), u2 (usage only), null-id usage, anon9 (client only)
    assert report.user_merged_null_key == 1
    assert report.events_matched == 3  # u1, anon9, u2
    assert report.events_unmatched == 1  # ghost


def test_duplicate_client_id_violates_one_to_one():
    usage, clients, events = synthetic_logs()
    clients = pd.concat([clients, clients.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        merge_logs(usage, clients, events)
