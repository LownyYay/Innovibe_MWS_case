"""Validate the immutable v2.1 training and scoring inputs."""
import hashlib
import json
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("training_config.json").read_text(encoding="utf-8"))
# The outer package includes these runtime inputs. Schema/FAQ hashes are provenance.
REQUIRED = {"train.jsonl", "val.jsonl", "case_metrics.py", "catalog.json",
            "baseline_pred.jsonl", "baseline_metrics.json"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_data(directory):
    directory = Path(directory)
    for name in sorted(REQUIRED):
        path = directory / name
        if not path.is_file() or digest(path) != CONFIG["data_sha256"][name]:
            raise ValueError(f"Missing file or v2.1 SHA-256 mismatch: {path}")
    return CONFIG["data_sha256"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]
