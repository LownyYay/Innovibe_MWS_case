"""Model I/O shared by SFT, evaluation and the service."""
import json
from pathlib import Path

from targets import target_facts

CONTRACT = json.loads(Path(__file__).with_name("inference_contract.json").read_text(encoding="utf-8"))
SYSTEM = CONTRACT["prompt"]["system"]
INPUT_FIELDS = CONTRACT["prompt"]["input_fields"]
CHAT = CONTRACT["chat"]["inference_kwargs"]


def messages(row):
    payload = {key: row[key] for key in INPUT_FIELDS}
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def facts_without_object_ids(facts):
    """Return new fact and object mappings while preserving every other field."""
    return [
        {**fact, "object": {key: value for key, value in fact["object"].items() if key != "id"}}
        for fact in facts
    ]


def training_target(row):
    return json.dumps(
        {"facts": facts_without_object_ids(target_facts(row, row["facts"]))},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def check_tokenizer(tokenizer):
    expected = CONTRACT["tokenizer"]
    if tokenizer.chat_template != CONTRACT["chat"]["template"]:
        raise ValueError("Unexpected chat template for CompactFactExtraction v2.1")
    if tokenizer.eos_token_id != expected["eos_token_id"]:
        raise ValueError("Unexpected EOS token for CompactFactExtraction v2.1")
