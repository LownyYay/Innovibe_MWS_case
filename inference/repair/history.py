"""Carry active state onsets across past calls of an explicitly identified series."""
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
from temporal import iso_day, resolve_boundary

ATTRS = {"service_state", "ticket_status", "tariff"}


def normalize(value):
    return " ".join(re.findall(r"[а-яёa-z0-9]+", (value or "").lower().replace("ё", "е")))


def similarity(left, right):
    a, b = normalize(left), normalize(right)
    if a == b:
        return 1.0
    negatives = lambda text: bool(re.search(r"\b(?:не|нет|без)\b", text))
    if negatives(a) != negatives(b):
        return 0.0
    at = {w[:5] for w in a.split() if len(w) > 2}
    bt = {w[:5] for w in b.split() if len(w) > 2}
    overlap = len(at & bt) / max(1, max(len(at), len(bt)))
    return max(overlap, SequenceMatcher(None, a, b).ratio())


def object_similarity(left, right):
    a, b = left.get("object", {}), right.get("object", {})
    if a.get("type") != b.get("type"):
        return 0.0
    if a.get("type") == "client":
        return 1.0
    x, y = a.get("name") or "", b.get("name") or ""
    if not x or not y:
        return 0.0
    xn, yn = set(re.findall(r"\d+", x)), set(re.findall(r"\d+", y))
    if xn and yn and xn != yn:
        return 0.0
    return similarity(x, y)


def account(row):
    series = row.get("series")
    return str(series["account_id"]) if isinstance(series, dict) and series.get("account_id") else None


def chronology(row):
    value = datetime.fromisoformat(row["started_at"])
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp(), row["series"].get("call_index", 0), row["call_id"]


def finalize(row, prediction, raw_facts):
    from apply_repair import chain_states
    events = []
    turns = {t["idx"]: t["text"] for t in row["turns"]}
    for i, (fact, raw) in enumerate(zip(prediction["facts"], raw_facts, strict=True)):
        end = raw.get("valid_to")
        if fact.get("valid_to") is not None or not isinstance(end, dict) or end.get("status") != "explicit":
            continue
        try:
            value, _ = resolve_boundary(end, turns, iso_day(row["started_at"]), "to")
        except (ValueError, OverflowError):
            continue
        if fact["valid_from"][:10] <= value[:10]:
            fact["valid_to"] = value
            events.append({"call_id": row["call_id"], "fact_index": i, "field": "valid_to",
                           "before": None, "after": value, "method": "explicit_end_after_history",
                           "turn_idx": end["turn_idx"], "expression": end["expression"]})
    events += chain_states(row, prediction, raw_facts)
    return events


def apply_history(rows, indexed, raw_indexed, repair_events, policy):
    """Mutate own prediction copies only. No gold, future calls or cross-account lookup."""
    if any("facts" in row for row in rows):
        raise ValueError("History accepts public calls without reference facts only")
    events = []
    accounts = defaultdict(list)
    for row in rows:
        key = account(row)
        if key is not None:
            accounts[key].append(row)
    confidence = {(e["call_id"], e["fact_index"], e["field"]): e.get("confidence", 1.0)
                  for e in repair_events}
    for group in accounts.values():
        group.sort(key=chronology)
        prior = []
        for row in group:
            cid = row["call_id"]
            current = indexed[cid]["facts"]
            raw = raw_indexed[cid]
            if raw.get("error") or raw.get("generation_error"):
                continue
            import json
            raw_facts = json.loads(raw["text"])["facts"]
            for i, (fact, raw_fact) in enumerate(zip(current, raw_facts, strict=True)):
                if fact["attribute"] not in ATTRS or fact.get("polarity", 1) <= 0:
                    continue
                status = raw_fact.get("valid_from")
                if not isinstance(status, dict) or status.get("status") != "unresolved":
                    continue
                scope = policy["scope"]
                is_fallback = fact["valid_from"][:10] == row["started_at"][:10]
                if scope == "fallback" and not is_fallback:
                    continue
                if scope == "weak" and not is_fallback and confidence.get((cid, i, "valid_from"), 0.0) >= 0.8:
                    continue
                matches = []
                for old_row, old_fact in prior:
                    # Within equal timestamps, series call_index determines the
                    # ordering; never use a later timestamp or a later index.
                    if chronology(old_row)[0] > chronology(row)[0]:
                        continue
                    if chronology(old_row)[0] == chronology(row)[0] and old_row["series"].get("call_index", 0) >= row["series"].get("call_index", 0):
                        continue
                    if old_fact["attribute"] != fact["attribute"]:
                        continue
                    if object_similarity(fact, old_fact) < policy["similarity"]:
                        continue
                    if similarity(fact["value"], old_fact["value"]) < policy["similarity"]:
                        continue
                    onset = old_fact["valid_from"]
                    if onset[:10] > old_row["started_at"][:10]:
                        continue
                    if fact.get("valid_to") and onset[:10] > fact["valid_to"][:10]:
                        continue
                    matches.append((old_row, old_fact))
                if not matches:
                    continue
                latest_time = max(chronology(r) for r, f in matches)
                latest = [(r, f) for r, f in matches if chronology(r) == latest_time]
                dates = {f["valid_from"][:10] for r, f in latest}
                if len(dates) != 1:
                    continue
                old_row, old_fact = latest[-1]
                value = old_fact["valid_from"]
                if value[:10] == fact["valid_from"][:10]:
                    continue
                events.append({"call_id": cid, "fact_index": i, "field": "valid_from",
                               "before": fact["valid_from"], "after": value,
                               "method": "past_series_active_state", "prior_call_id": old_row["call_id"],
                               "prior_started_at": old_row["started_at"]})
                fact["valid_from"] = value
            events += finalize(row, indexed[cid], raw_facts)
            # Replace the cache for each mentioned object/attribute by its
            # latest active state(s). This prevents jumping across an observed
            # intervening state change to an older same-valued state.
            for fact in current:
                if fact["attribute"] not in ATTRS or fact.get("polarity", 1) <= 0:
                    continue
                prior = [(old_row, old_fact) for old_row, old_fact in prior
                         if old_fact["attribute"] != fact["attribute"] or object_similarity(fact, old_fact) < policy["similarity"]]
            active = [f for f in current if f["attribute"] in ATTRS and f.get("polarity", 1) > 0
                      and f.get("valid_to") is None and f["valid_from"][:10] <= row["started_at"][:10]]
            for fact in active:
                peers = [f for f in active if f["attribute"] == fact["attribute"] and object_similarity(fact, f) >= policy["similarity"]]
                if not peers:
                    continue
                latest_onset = max(f["valid_from"][:10] for f in peers)
                newest = [f for f in peers if f["valid_from"][:10] == latest_onset]
                # Equal-date contradictory active states are not reliable history.
                if len({normalize(f["value"]) for f in newest}) != 1:
                    continue
                if fact["valid_from"][:10] == latest_onset:
                    prior.append((row, fact.copy()))
    return events
