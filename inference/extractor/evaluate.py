"""Evaluate a span adapter or replay saved run02 model output on CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

from data_v21 import read_jsonl, verify_data
from model_io import CONTRACT, CHAT, check_tokenizer, messages
from validation import decode_prediction


def generated_outputs(rows, model_path, adapter_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from train import load_base_tokenizer

    adapter_path = Path(adapter_path)
    adapter_contract = adapter_path.parent / "inference_contract.json"
    if not adapter_contract.is_file() or json.loads(adapter_contract.read_text(encoding="utf-8")) != CONTRACT:
        raise ValueError("Expected a span adapter with its exported inference_contract.json")
    load_base_tokenizer(model_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    check_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter_path).eval()
    generation = GenerationConfig(**CONTRACT["generation"])
    for row in rows:
        inputs = tokenizer.apply_chat_template(
            messages(row), tokenize=True, **CHAT, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            result = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=generation, do_sample=False,
            )
        tokens = result[0, inputs.shape[1]:]
        exhausted = len(tokens) == generation.max_new_tokens and tokens[-1].item() != tokenizer.eos_token_id
        yield {
            "call_id": row["call_id"],
            "text": tokenizer.decode(tokens, **CONTRACT["output"]["decode"]),
            "generation_error": "generation reached max_new_tokens" if exhausted else None,
        }


def saved_outputs(rows, path):
    indexed = {}
    for item in read_jsonl(path):
        key = item["call_id"]
        if key in indexed:
            raise ValueError(f"duplicate call_id in raw outputs: {key}")
        indexed[key] = item
    if indexed.keys() != {row["call_id"] for row in rows}:
        raise ValueError("raw outputs must cover all and only the validation calls")
    return [indexed[row["call_id"]] for row in rows]


def evaluate(rows, outputs, out, *, legacy_iso=False):
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    structural_errors = []
    temporal_errors = Counter()
    temporal_calls = 0
    with (
        (out / "predictions.jsonl").open("w", encoding="utf-8") as pred,
        (out / "raw_outputs.jsonl").open("w", encoding="utf-8") as raw,
    ):
        for row, output in zip(rows, outputs, strict=True):
            call_id, text = row["call_id"], output["text"]
            reason = None
            issues = []
            try:
                if output.get("generation_error"):
                    raise ValueError(output["generation_error"])
                prediction, issues = decode_prediction(json.loads(text), row, legacy_iso=legacy_iso)
            except (ValueError, TypeError, KeyError) as error:
                reason = str(error)
                structural_errors.append({"call_id": call_id, "error": reason})
                prediction = {"call_id": call_id, "facts": []}
            if issues:
                temporal_calls += 1
                temporal_errors.update(f'{issue["field"]}: {issue["error"]}' for issue in issues)
            pred.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            raw.write(json.dumps({
                "call_id": call_id, "text": text, "error": reason, "temporal_issues": issues,
            }, ensure_ascii=False) + "\n")
    summary = {
        "n_calls": len(rows), "n_errors": len(structural_errors), "errors": structural_errors,
        "n_calls_with_temporal_issues": temporal_calls, "temporal_issues": dict(temporal_errors),
        "elapsed_seconds": time.monotonic() - start,
    }
    (out / "validation_errors.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def score(data, out):
    for name, prediction in (("candidate", out / "predictions.jsonl"), ("baseline", data / "baseline_pred.jsonl")):
        metrics_path = out / f"{name}_metrics.json"
        subprocess.run(
            [sys.executable, str(data / "case_metrics.py"), str(data / "val.jsonl"), str(prediction),
             "--json", str(metrics_path)], check=True,
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics["metrics_version"] != CONTRACT["case_version"]:
            raise ValueError("unexpected scorer version")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--raw-outputs", type=Path)
    parser.add_argument("--legacy-iso", action="store_true", help="Replay saved ISO targets from the previous adapter")
    args = parser.parse_args()
    if args.raw_outputs is not None:
        if args.model or args.adapter:
            parser.error("use --raw-outputs or --model with --adapter")
        if args.raw_outputs.resolve() == (args.out / "raw_outputs.jsonl").resolve():
            parser.error("use a different output directory for replay")
    elif not (args.model and args.adapter) or args.legacy_iso:
        parser.error("provide --model and --adapter, or --raw-outputs for replay")
    verify_data(args.data)
    rows = read_jsonl(args.data / "val.jsonl")
    outputs = (
        saved_outputs(rows, args.raw_outputs)
        if args.raw_outputs is not None
        else generated_outputs(rows, args.model, args.adapter)
    )
    evaluate(rows, outputs, args.out, legacy_iso=args.legacy_iso)
    score(args.data, args.out)


if __name__ == "__main__":
    main()

