"""Download the exact base snapshot required by training and evaluation."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from data_v21 import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../base_model"))
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("inference_contract.json").read_text(encoding="utf-8"))
    snapshot_download(
        repo_id=contract["base_model"],
        revision=contract["revision"],
        local_dir=str(args.output),
    )
    for name, expected in contract["tokenizer"]["sha256_source"].items():
        if digest(args.output / name) != expected:
            raise ValueError(f"Downloaded snapshot hash mismatch: {name}")
    print(f"Base model ready: {args.output.resolve()}")


if __name__ == "__main__":
    main()
