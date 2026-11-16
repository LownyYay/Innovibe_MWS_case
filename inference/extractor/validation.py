"""Convert model JSON to v2.1 service facts at the response boundary."""
from temporal import FIELDS, boundary_role, iso_day, resolve_boundary


def validate_structure(fact, turns, index):
    if not isinstance(fact, dict):
        raise ValueError(f"fact {index}: not an object")
    obj, source = fact.get("object"), fact.get("source")
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        raise ValueError(f"fact {index}: invalid object.name")
    if not all(isinstance(fact.get(key), str) and fact[key] for key in ("attribute", "value")):
        raise ValueError(f"fact {index}: invalid attribute/value")
    if type(fact.get("polarity")) is not int or fact["polarity"] not in (1, -1):
        raise ValueError(f"fact {index}: invalid polarity")
    if not isinstance(source, dict) or type(source.get("turn_idx")) is not int or source["turn_idx"] not in turns:
        raise ValueError(f"fact {index}: invalid source.turn_idx")
    span = source.get("span")
    if span is not None and (
        not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
        or span[0] < 0 or span[1] < span[0]
    ):
        raise ValueError(f"fact {index}: invalid source.span")


def decode_prediction(obj, row, *, legacy_iso=False):
    """Return public {call_id, facts} and recoverable temporal errors."""
    if not isinstance(obj, dict) or not isinstance(obj.get("facts"), list):
        raise ValueError("expected object with facts array")
    turns = {turn["idx"]: turn["text"] for turn in row["turns"]}
    base = iso_day(row["started_at"])
    fallback = {"valid_from": base.isoformat(), "valid_to": None}
    issues = []
    facts = []
    for index, predicted in enumerate(obj["facts"]):
        validate_structure(predicted, turns, index)
        fact = {**predicted, "object": {key: value for key, value in predicted["object"].items() if key != "id"}}
        for field in FIELDS:
            value = predicted.get(field)
            fact[field] = fallback[field]
            try:
                if field not in predicted:
                    raise ValueError("missing temporal field")
                if field == "valid_to" and value is None:
                    continue
                if legacy_iso:
                    iso_day(value)
                    fact[field] = value
                    continue
                status = value.get("status") if isinstance(value, dict) else None
                if field == "valid_from" and status == "implicit":
                    continue
                if status == "unresolved":
                    raise ValueError("unresolved temporal expression")
                role = boundary_role(predicted["attribute"], field)
                fact[field], _ = resolve_boundary(value, turns, base, role)
            except (ValueError, OverflowError) as error:
                issues.append({"fact_index": index, "field": field, "error": str(error)})
        if not legacy_iso and fact["valid_to"] is not None and iso_day(fact["valid_from"]) > iso_day(fact["valid_to"]):
            fact["valid_to"] = None
            issues.append({"fact_index": index, "field": "valid_to", "error": "end precedes start"})
        facts.append(fact)
    return {"call_id": row["call_id"], "facts": facts}, issues

