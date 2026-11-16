"""Restore unresolved dates from public inputs with a frozen train-only linker."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
import zipfile

from candidates import cues, options, tokens, overlap
from ranker import choose
from temporal import resolve_boundary, iso_day


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def non_temporal(fact):
    result = {k: v for k, v in fact.items() if k not in ("valid_from", "valid_to")}
    if "object" in result:
        result["object"] = {k: v for k, v in result["object"].items() if k != "id"}
    return result


def chain_states(row, prediction, raw_facts):
    """Close a clearly historical single-valued state at a later state's onset."""
    events = []
    facts = prediction["facts"]
    groups = {}
    for i, fact in enumerate(facts):
        if fact["attribute"] not in ("service_state", "ticket_status", "tariff") or fact.get("polarity", 1) <= 0:
            continue
        obj = fact.get("object", {})
        key = (obj.get("type"), obj.get("name"), fact["attribute"])
        if not obj.get("name"):
            continue
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        for i in indices:
            old = facts[i]
            raw_end = raw_facts[i].get("valid_to")
            if old.get("valid_to") is not None or not isinstance(raw_end, dict) or raw_end.get("status") != "unresolved":
                continue
            start = old["valid_from"][:10]
            successors = []
            for j in indices:
                if j == i:
                    continue
                new = facts[j]
                if overlap(tokens(old["value"]), tokens(new["value"])) >= 0.8:
                    continue
                new_start = new["valid_from"][:10]
                if not (start < new_start <= row["started_at"][:10]):
                    continue
                if (new.get("source") or {}).get("turn_idx", -999) < (old.get("source") or {}).get("turn_idx", -999):
                    continue
                successors.append(new)
            if successors:
                following = min(successors, key=lambda f: f["valid_from"])
                old["valid_to"] = following["valid_from"]
                events.append({"call_id": row["call_id"], "fact_index": i, "field": "valid_to",
                               "before": None, "after": old["valid_to"], "method": "single_valued_state_transition",
                               "next_value": following["value"], "next_source": following.get("source")})
    return events


def repair_call(row, prediction, raw, artifact):
    if "facts" in row:
        raise ValueError("Repair accepts public input without reference facts only")
    output = copy.deepcopy(prediction)
    if raw.get("error") or raw.get("generation_error"):
        return output, []
    raw_facts = json.loads(raw["text"])["facts"]
    if len(raw_facts) != len(output["facts"]):
        raise ValueError("Raw output and prediction fact counts differ")
    public_facts = [non_temporal(f) for f in output["facts"]]
    found = cues(row)
    events = []
    for fi, (fact, raw_fact) in enumerate(zip(output["facts"], raw_facts, strict=True)):
        if non_temporal(fact) != non_temporal(raw_fact):
            raise ValueError("Raw output and prediction non-temporal content differs")
        if fact["attribute"] == "action":
            continue
        before = copy.deepcopy(fact)
        proposed = []
        for field in ("valid_from", "valid_to"):
            status = raw_fact.get(field)
            if not isinstance(status, dict) or status.get("status") != "unresolved":
                continue
            choices = options(row, public_facts[fi], field, public_facts, found)
            if artifact.get("n_features") and any(len(x) != artifact["n_features"] for _, x, _ in choices):
                raise ValueError("Date model and feature code versions differ")
            day, confidence, margin, evidence = choose(choices, artifact["model"])
            if confidence < artifact["thresholds"][field] or margin < artifact["margin"]:
                continue
            value = day
            support = None
            if evidence:
                source_idx = (fact.get("source") or {}).get("turn_idx", -999)
                role = "deadline" if fact["attribute"] == "agreement" and field == "valid_from" else "from"
                support = min(evidence, key=lambda h: (abs(h["idx"] - source_idx), h["role"] != role, h["start"]))
                value = support["iso"]
            if (before.get(field) or "")[:10] == (value or "")[:10]:
                continue
            fact[field] = value
            proposed.append({"field": field, "before": before.get(field), "after": value,
                             "confidence": confidence, "margin": margin,
                             "turn_idx": support["idx"] if support else None,
                             "expression": support["expression"] if support else None,
                             "coarse_interval": support["coarse"] if support else False})
        if before.get("valid_to") is not None and fact["valid_from"][:10] > before["valid_to"][:10]:
            # Never discard a previously valid end because the new onset proposal conflicts.
            fact["valid_from"] = before["valid_from"]
        if artifact.get("restore_explicit_end", False) and fact.get("valid_to") is None:
            raw_end = raw_fact.get("valid_to")
            if isinstance(raw_end, dict) and raw_end.get("status") == "explicit":
                turns = {t["idx"]: t["text"] for t in row["turns"]}
                try:
                    end, _ = resolve_boundary(raw_end, turns, iso_day(row["started_at"]), "to")
                    if fact["valid_from"][:10] <= end[:10]:
                        fact["valid_to"] = end
                        proposed.append({"field": "valid_to", "before": before.get("valid_to"), "after": end,
                                         "method": "restore_explicit_end_after_onset_repair",
                                         "turn_idx": raw_end["turn_idx"], "expression": raw_end["expression"]})
                except (ValueError, OverflowError):
                    pass
        if fact.get("valid_to") is not None and fact["valid_from"][:10] > fact["valid_to"][:10]:
            fact["valid_to"] = None
        assert non_temporal(fact) == non_temporal(before)
        for event in proposed:
            if fact.get(event["field"]) != before.get(event["field"]):
                events.append({"call_id": row["call_id"], "fact_index": fi, **event,
                               "after": fact.get(event["field"])})
    if artifact.get("chain_states", False):
        events += chain_states(row, output, raw_facts)
    return output, events


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=base / "inputs" / "public_calls.jsonl")
    parser.add_argument("--pred", type=Path, default=base / "inputs" / "original_predictions.jsonl")
    parser.add_argument("--raw", type=Path, default=base / "inputs" / "raw_outputs.jsonl")
    parser.add_argument("--model", type=Path, default=base / "date_ranker.json")
    parser.add_argument("--data", type=Path, default=base / "scoring")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="Process calls independently")
    args = parser.parse_args()
    args.out = args.out or base / ("repaired_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = read_jsonl(args.inputs)
    predictions = read_jsonl(args.pred)
    raw = read_jsonl(args.raw)
    indexed = {r["call_id"]: r for r in predictions}
    raw_indexed = {r["call_id"]: r for r in raw}
    ids = {r["call_id"] for r in rows}
    assert len(ids) == len(rows) == len(predictions) == len(indexed) == len(raw) == len(raw_indexed)
    assert ids == indexed.keys() == raw_indexed.keys()
    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    expected_features = artifact.get("feature_source_sha256")
    if expected_features and hashlib.sha256(Path(__file__).with_name("candidates.py").read_bytes()).hexdigest() != expected_features:
        raise ValueError("Feature source hash does not match the frozen model")
    events = []
    out_predictions = args.out / "predictions.jsonl"
    fixed_indexed = {}
    for i, row in enumerate(rows, 1):
        result, changes = repair_call(row, indexed[row["call_id"]], raw_indexed[row["call_id"]], artifact)
        fixed_indexed[row["call_id"]] = result
        events += changes
        if i % 50 == 0 or i == len(rows):
            print(f"Date repair: {i}/{len(rows)} calls; {len(events)} changed boundaries", flush=True)
    if artifact.get("history_policy") and not args.no_history:
        from history import apply_history
        history_events = apply_history(rows, fixed_indexed, raw_indexed, events, artifact["history_policy"])
        events += history_events
        print(f"Past-series history: {len(history_events)} additional boundary changes", flush=True)
    if artifact.get("joint_policy"):
        from joint import candidate_cache, decode_cached
        joint_events = []
        for row in rows:
            prediction = fixed_indexed[row["call_id"]]
            cache = candidate_cache(row, prediction, raw_indexed[row["call_id"]], artifact)
            joint_events += decode_cached(row, prediction, cache, artifact["joint_policy"])
        events += joint_events
        print(f"Joint interval decoder: {len(joint_events)} additional boundary changes", flush=True)
    with out_predictions.open("w", encoding="utf-8") as stream:
        for row in rows:
            result = fixed_indexed[row["call_id"]]
            assert [non_temporal(f) for f in result["facts"]] == [non_temporal(f) for f in indexed[row["call_id"]]["facts"]]
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    repair_seconds = time.monotonic() - started
    (args.out / "changes.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"n_calls": len(rows), "changed_boundaries": len(events), "repair_seconds": repair_seconds,
               "policy": artifact["selection"], "thresholds": artifact["thresholds"],
               "history_policy": artifact.get("history_policy") if not args.no_history else None,
               "joint_policy": artifact.get("joint_policy"),
               "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in (("public_calls", args.inputs), ("original_predictions", args.pred), ("raw_outputs", args.raw), ("date_ranker", args.model))}}
    if not args.no_score:
        # Reference annotations are accessed only here, after predictions have
        # already been fixed and written. They are not passed to repair_call.
        for name, path in (("before", args.pred), ("after", out_predictions)):
            output = args.out / f"{name}_metrics.json"
            subprocess.run([sys.executable, str((args.data / "case_metrics.py").resolve()),
                            str((args.data / "val.jsonl").resolve()), str(path.resolve()),
                            "--json", str(output)], cwd=args.data, check=True)
            summary[name] = json.loads(output.read_text(encoding="utf-8"))
        for key in ("n_calls", "n_gold_facts", "n_pred_facts", "fact_precision", "fact_recall", "grounding_turn", "grounding_span", "n_temporal"):
            assert summary["before"][key] == summary["after"][key], f"Unexpected change: {key}"
    (args.out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = args.out.with_name(args.out.name + ".zip")
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as package:
        for path in args.out.iterdir():
            if path.is_file():
                package.write(path, path.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Results:", args.out, flush=True)
    print("ZIP:", zip_path, flush=True)


if __name__ == "__main__":
    main()
