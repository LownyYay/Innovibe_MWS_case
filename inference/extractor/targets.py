"""Project v2.1 gold facts into span supervision without modifying the corpus."""
from collections import defaultdict
import re

from temporal import (
    FIELDS, UNRESOLVED_CUE, boundary_role, iso_day, mentions, resolve_boundary,
)


STATE_ATTRIBUTES = {"service_state", "tariff", "ticket_status"}


def quote_bounds(fact, text):
    source = fact["source"]
    quote = source.get("quote")
    if not quote:
        return None
    span = source.get("span")
    if span and 0 <= span[0] < span[1] <= len(text) and text[slice(*span)] == quote:
        return tuple(span)
    position = text.find(quote)
    if position >= 0 and text.find(quote, position + 1) < 0:
        return position, position + len(quote)
    return None


def temporal_targets(fact, turns, base, matches):
    idx = fact["source"]["turn_idx"]
    text = turns.get(idx, "")
    bounds = quote_bounds(fact, text)
    local_matches = matches.get(idx, ())
    scope = text[slice(*bounds)] if bounds else text
    scoped_matches = list(mentions(scope)) if bounds else local_matches
    labels = {}
    for field in FIELDS:
        gold = fact.get(field)
        if field == "valid_to" and gold is None:
            labels[field] = None
            continue
        gold_day = iso_day(gold) if gold is not None else None
        role = boundary_role(fact["attribute"], field)
        candidates = []
        for hit in local_matches:
            expression = hit.group()
            in_quote = bounds and bounds[0] <= hit.start() < hit.end() <= bounds[1]
            in_value = re.search(r"(?<!\w)" + re.escape(expression) + r"(?!\w)", fact["value"], re.I)
            if not (in_quote or in_value):
                continue
            evidence = {"status": "explicit", "turn_idx": idx, "expression": expression}
            try:
                value, _ = resolve_boundary(evidence, turns, base, role)
            except (ValueError, OverflowError):
                continue
            if iso_day(value) == gold_day:
                candidates.append(evidence)

        if len(candidates) == 1:
            labels[field] = candidates[0]
        elif (field == "valid_from" and gold_day == base and scope
              and not scoped_matches and not UNRESOLVED_CUE.search(scope)):
            labels[field] = {"status": "implicit"}
        else:
            labels[field] = {"status": "unresolved"}
    if labels["valid_from"] == labels["valid_to"] and labels["valid_from"]["status"] == "explicit":
        labels["valid_to"] = {"status": "unresolved"}
    return labels


def link_state_transitions(facts, labels, turns, base):
    """Use a unique grounded next-state onset as the previous state's end label."""
    onsets = defaultdict(list)
    for fact, boundaries in zip(facts, labels, strict=True):
        attr, object_id = fact["attribute"], fact["object"].get("id")
        onset = boundaries["valid_from"]
        if attr in STATE_ATTRIBUTES and object_id and fact["polarity"] == 1 and onset["status"] == "explicit":
            onsets[(object_id, attr, iso_day(fact["valid_from"]))].append((fact["value"], onset))

    for fact, boundaries in zip(facts, labels, strict=True):
        if (boundaries["valid_to"] != {"status": "unresolved"}
                or not fact.get("valid_to") or fact["polarity"] != 1):
            continue
        end_day = iso_day(fact["valid_to"])
        key = (fact["object"].get("id"), fact["attribute"], end_day)
        candidates = []
        for value, evidence in onsets.get(key, ()):
            if value == fact["value"] or evidence == boundaries["valid_from"] or evidence in candidates:
                continue
            resolved, _ = resolve_boundary(evidence, turns, base, "to")
            if iso_day(resolved) == end_day:
                candidates.append(evidence)
        if len(candidates) == 1:
            boundaries["valid_to"] = candidates[0]


def target_facts(row, facts):
    turns = {turn["idx"]: turn["text"] for turn in row["turns"]}
    matches = {idx: list(mentions(text)) for idx, text in turns.items()}
    base = iso_day(row["started_at"])
    labels = [temporal_targets(fact, turns, base, matches) for fact in facts]
    link_state_transitions(facts, labels, turns, base)
    return [{**fact, **boundaries} for fact, boundaries in zip(facts, labels, strict=True)]

