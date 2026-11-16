"""Joint interval decoding from frozen unary date scores and state transitions."""
from datetime import date
import math

from candidates import cues, options
from ranker import probability
from history import object_similarity, similarity
from temporal import iso_day, resolve_boundary


def logit(p):
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def candidate_cache(row, prediction, raw, artifact):
    """Extract all scores once; neither gold nor any other call is inspected."""
    import json
    if "facts" in row:
        raise ValueError("Joint decoder accepts public calls without reference facts only")
    if raw.get("error") or raw.get("generation_error"):
        return []
    raw_facts = json.loads(raw["text"])["facts"]
    if len(raw_facts) != len(prediction["facts"]):
        raise ValueError("Prediction/raw fact counts differ")
    found = cues(row)
    turns = {t["idx"]: t["text"] for t in row["turns"]}
    base = iso_day(row["started_at"])
    cache = []
    for i, (fact, rf) in enumerate(zip(prediction["facts"], raw_facts, strict=True)):
        if fact["attribute"] not in ("ticket_status", "tariff", "service_state") or fact.get("polarity", 1) <= 0:
            continue
        successors = []
        idx = (fact.get("source") or {}).get("turn_idx", -999)
        for j, nxt in enumerate(prediction["facts"]):
            if j == i or nxt["attribute"] != fact["attribute"] or nxt.get("polarity", 1) <= 0:
                continue
            if object_similarity(fact, nxt) < 1.0 or similarity(fact["value"], nxt["value"]) >= 0.8:
                continue
            if (nxt.get("source") or {}).get("turn_idx", -999) < idx:
                continue
            day = nxt["valid_from"][:10]
            if day <= row["started_at"][:10]:
                successors.append(day)
        record = {"i": i, "attr": fact["attribute"], "old": (fact["valid_from"][:10], (fact.get("valid_to") or "")[:10] or None),
                  "successors": sorted(set(successors)), "fields": {}, "variable": {}}
        for field in ("valid_from", "valid_to"):
            old_value = fact.get(field)
            old_day = old_value[:10] if old_value else None
            status = rf.get(field)
            variable = isinstance(status, dict) and status.get("status") == "unresolved"
            candidates = options(row, fact, field, prediction["facts"], found)
            scored = []
            known_end = None
            if field == "valid_to" and isinstance(status, dict) and status.get("status") == "explicit":
                try:
                    known_end, _ = resolve_boundary(status, turns, base, "to")
                except (ValueError, OverflowError):
                    pass
            if not variable:
                value = known_end if known_end else old_value
                scored = [{"day": value[:10] if value else None, "iso": value,
                           "score": 0.0, "p": 1.0, "support": None, "old": value == old_value}]
            else:
                in_pool = any(d == old_day for d, _, _ in candidates)
                # An onset obtained from earlier series history is outside this
                # call's candidate pool; keep this independently supported anchor.
                if field == "valid_from" and not in_pool:
                    variable = False
                    scored = [{"day": old_day, "iso": old_value, "score": 0.0, "p": 1.0, "support": None, "old": True}]
                else:
                    for day, features, evidence in candidates:
                        p = probability(features, artifact["model"])
                        if day != old_day and day is not None and not evidence and day not in successors:
                            continue
                        if day != old_day and p < (0.05 if field == "valid_from" else 0.03):
                            continue
                        support = min(evidence, key=lambda h: (abs(h["idx"] - idx), h["start"])) if evidence else None
                        iso = old_value if day == old_day else support["iso"] if support else day
                        scored.append({"day": day, "iso": iso, "score": logit(p), "p": p,
                                       "support": support, "old": day == old_day})
                    if not any(c["day"] == old_day for c in scored):
                        scored.append({"day": old_day, "iso": old_value, "score": 0.0, "p": 0.5, "support": None, "old": True})
            record["fields"][field] = scored
            record["variable"][field] = variable
        cache.append(record)
    return cache


def decode_cached(row, prediction, cache, policy):
    """Mutate prediction dates only; return a complete provenance trail."""
    changes = []
    for rec in cache:
        if rec["attr"] not in policy["attributes"]:
            continue
        starts = rec["fields"]["valid_from"]
        ends = rec["fields"]["valid_to"]
        if policy["mode"] == "end_only":
            starts = [c for c in starts if c["day"] == rec["old"][0]]
        if not starts or not ends:
            continue

        def score(start, end):
            a, b = start["day"], end["day"]
            if a is None or (b is not None and b < a):
                return -math.inf
            value = start["score"] + end["score"]
            if rec["variable"]["valid_to"] and b is not None:
                value += policy["closed_bonus"]
                next_states = [d for d in rec["successors"] if d > a]
                if next_states:
                    next_day = min(next_states)
                    if b == next_day:
                        value += policy["transition_bonus"]
                    elif b > next_day:
                        value -= policy["transition_bonus"]
            return value

        old_start = next((c for c in starts if c["day"] == rec["old"][0]), None)
        old_end = next((c for c in ends if c["day"] == rec["old"][1]), None)
        old_score = score(old_start, old_end) if old_start and old_end else -math.inf
        pairs = [(score(a, b), a, b) for a in starts for b in ends]
        best_score, a, b = max(pairs, key=lambda x: (x[0], int(x[1]["old"]) + int(x[2]["old"])))
        if best_score == -math.inf or best_score < old_score + policy["gain"]:
            continue
        fact = prediction["facts"][rec["i"]]
        for field, candidate in (("valid_from", a), ("valid_to", b)):
            old = fact.get(field)
            if (old or "")[:10] == (candidate["iso"] or "")[:10]:
                continue
            fact[field] = candidate["iso"]
            support = candidate["support"]
            changes.append({"call_id": row["call_id"], "fact_index": rec["i"], "field": field,
                            "before": old, "after": candidate["iso"], "method": "joint_interval_decoder",
                            "score_gain": best_score - old_score if math.isfinite(old_score) else None,
                            "candidate_score": candidate["p"],
                            "turn_idx": support["idx"] if support else None,
                            "expression": support["expression"] if support else None})
    return changes
