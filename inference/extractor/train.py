import argparse, json, os, subprocess, sys
from pathlib import Path
import torch
from torch.nn.functional import cross_entropy
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, PeftModel, get_peft_model

from data_v21 import CONFIG, digest, read_jsonl, verify_data
from model_io import CONTRACT, CHAT, check_tokenizer, messages, training_target

SETTINGS = CONFIG["training"]

LOSS_CHUNK_TOKENS = SETTINGS["loss_chunk_tokens"]


class Qwen3ForSFT(Qwen3ForCausalLM):
    def forward(
        self, input_ids=None, attention_mask=None, labels=None,
        num_items_in_batch=None, **kwargs,
    ):
        if labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        # Hidden state at position t predicts the label at t+1.
        shifted_labels = labels[:, 1:]
        supervised = shifted_labels.ne(-100)
        target_hidden = hidden[:, :-1][supervised]
        targets = shifted_labels[supervised]
        if targets.numel() == 0:
            raise ValueError('Batch contains no supervised tokens')

        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), LOSS_CHUNK_TOKENS):
            stop = start + LOSS_CHUNK_TOKENS
            chunk = (target_hidden[start:stop], targets[start:stop])
            if torch.is_grad_enabled():
                # Recompute each chunk during backward so vocabulary logits do not accumulate.
                loss_sum = loss_sum + checkpoint(self._chunk_loss, *chunk, use_reentrant=False)
            else:
                loss_sum = loss_sum + self._chunk_loss(*chunk)

        denominator = targets.numel() if num_items_in_batch is None else num_items_in_batch
        return CausalLMOutputWithPast(loss=loss_sum / denominator)

    def _chunk_loss(self, hidden, targets):
        logits = self.lm_head(hidden)
        return cross_entropy(logits.float(), targets, reduction='sum')


def encode(tok, row):
    m=messages(row)
    target=training_target(row)
    prefix=tok.apply_chat_template(m,tokenize=True,**CHAT)
    full=tok.apply_chat_template(m+[{'role':'assistant','content':target}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
    if full[:len(prefix)] != prefix: raise ValueError('Chat template prefix mismatch; refusing incorrect loss mask')
    if tok.eos_token_id not in full[len(prefix):]:
        raise ValueError('Assistant target lacks EOS')
    return {'input_ids':full,'labels':[-100]*len(prefix)+full[len(prefix):]}

class Data(Dataset):
    def __init__(self, rows, tok, limit):
        self.items=[]; lengths=[]
        for row in rows:
            x=encode(tok,row); lengths.append(len(x['input_ids'])); self.items.append(x)
        print('Token length max / mean:',max(lengths),sum(lengths)/len(lengths),flush=True)
        if max(lengths)>limit: raise ValueError(f'Max sequence {max(lengths)} exceeds MAX_LENGTH={limit}; increase it, no truncation applied')
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

class Collate:
    def __init__(self, pad): self.pad=pad
    def __call__(self, batch):
        n=max(len(x['input_ids']) for x in batch)
        return {'input_ids':torch.tensor([x['input_ids']+[self.pad]*(n-len(x['input_ids'])) for x in batch]), 'attention_mask':torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in batch]), 'labels':torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in batch])}

class CheckpointCallback(TrainerCallback):
    def __init__(self, identity):
        self.identity = identity

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "run_identity.json"
            path.write_text(json.dumps(self.identity, indent=2), encoding="utf-8")


def load_base_tokenizer(path):
    for name, expected in CONTRACT["tokenizer"]["sha256_source"].items():
        if digest(Path(path) / name) != expected:
            raise ValueError(f"Unexpected base snapshot file: {name}")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    check_tokenizer(tok)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True, help='Path to the pinned local Qwen snapshot')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-steps', type=int, choices=range(10, 21), default=SETTINGS['smoke_steps'])
    parser.add_argument('--save-steps', type=int, default=SETTINGS['save_steps'])
    parser.add_argument('--resume', default=None)
    parser.add_argument('--init-adapter', type=Path, help='Warm-start from the exported v2.1 run01 main/adapter')
    parser.add_argument('--max-length', type=int, default=SETTINGS['max_length'])
    parser.add_argument('--epochs', type=float, default=SETTINGS['epochs'])
    parser.add_argument('--batch', type=int, default=SETTINGS['batch_per_gpu'])
    parser.add_argument('--accum', type=int, default=SETTINGS['smoke_accum'])
    parser.add_argument('--seed', type=int, default=SETTINGS['seed'])
    parser.add_argument('--revision', choices=[CONTRACT['revision']], default=CONTRACT['revision'])
    parser.add_argument('--preflight', action='store_true', help='Validate all data and tokenization on CPU, no model weights loaded')
    a = parser.parse_args()
    if a.save_steps < 1:
        parser.error('--save-steps must be positive')
    checkpoint_steps = a.smoke_steps if a.smoke else a.save_steps
    d = Path(a.data)
    data_hashes = verify_data(d)
    tok = load_base_tokenizer(a.model)
    tr = Data(read_jsonl(d / 'train.jsonl'), tok, a.max_length)
    ev = Data(read_jsonl(d / 'val.jsonl'), tok, a.max_length)
    for split, dataset in (("train", tr), ("val", ev)):
        if len(dataset) != CONFIG["counts"][split]:
            raise ValueError(f"Unexpected {split} row count")
    if a.preflight:
        print(f"CPU preflight OK: {len(tr)} train / {len(ev)} val; prefixes, EOS and lengths verified")
        return

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if a.smoke and (world_size != 1 or torch.cuda.device_count() != 1):
        raise ValueError("Smoke requires one visible GPU and one process")
    if a.smoke:
        # Include the longest example from each split to exercise peak memory.
        for dataset, count in ((tr, a.smoke_steps * a.batch * a.accum), (ev, 4)):
            longest = max(range(len(dataset)), key=lambda i: len(dataset.items[i]["input_ids"]))
            first = dataset.items[:count]
            if longest >= count:
                first[-1] = dataset.items[longest]
            dataset.items = first
    set_seed(a.seed)
    out = Path(a.output)
    identity = {
        "case_version": "2.1",
        "train_sha256": data_hashes["train.jsonl"],
        "val_sha256": data_hashes["val.jsonl"],
        "contract_sha256": digest(Path(__file__).with_name("inference_contract.json")),
        "config_sha256": digest(Path(__file__).with_name("training_config.json")),
        "target_source_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("model_io.py", "targets.py", "temporal.py")
        },
        "init_adapter_sha256": {
            name: digest(a.init_adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        } if a.init_adapter else None,
        "training_args": {key: getattr(a, key) for key in
                          ("revision", "seed", "max_length", "epochs", "batch", "accum", "smoke")},
        "world_size": world_size,
        "checkpoint_steps": checkpoint_steps,
    }
    if a.resume:
        marker = Path(a.resume) / "run_identity.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume requires a compatible v2.1 checkpoint and matching training settings")
    elif out.exists() and (any(out.glob("checkpoint-*")) or (out / "adapter").exists()):
        raise ValueError(f"Output already contains a run: {out}. Set --resume or use a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    model=Qwen3ForSFT.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.config.use_cache=False
    if a.init_adapter:
        adapter = json.loads((a.init_adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if (adapter["base_model_name_or_path"], adapter["revision"]) != (CONTRACT["base_model"], CONTRACT["revision"]):
            raise ValueError("Initial adapter targets a different base model or revision")
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        lora = SETTINGS["lora"]
        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            bias="none", task_type="CAUSAL_LM", target_modules=lora["targets"],
        ))
    model.print_trainable_parameters()
    args=TrainingArguments(output_dir=str(out),per_device_train_batch_size=a.batch,per_device_eval_batch_size=1,gradient_accumulation_steps=a.accum,learning_rate=2e-4,num_train_epochs=a.epochs,max_steps=a.smoke_steps if a.smoke else -1,fp16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',lr_scheduler_type='cosine',warmup_ratio=0.03,logging_steps=1 if a.smoke else 10,eval_strategy='steps',save_strategy='steps',eval_steps=checkpoint_steps,save_steps=checkpoint_steps,save_total_limit=3,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',seed=a.seed,data_seed=a.seed,dataloader_num_workers=0,ddp_find_unused_parameters=False,save_safetensors=True,label_names=['labels'],prediction_loss_only=True)
    trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=ev,data_collator=Collate(tok.pad_token_id),processing_class=tok,callbacks=[CheckpointCallback(identity)])
    # PEFT forwards loss kwargs to Qwen3ForSFT. Normalize by the token count
    # for the entire accumulated batch, as in the standard causal LM loss.
    trainer.model_accepts_loss_kwargs = True
    print(f'SFT loss: {LOSS_CHUNK_TOKENS} supervised tokens per logits chunk', flush=True)
    trainer.train(resume_from_checkpoint=a.resume)
    metrics=trainer.evaluate(); trainer.log_metrics('eval',metrics); trainer.save_metrics('eval',metrics); trainer.save_state()
    print(f'Rank {os.environ.get("RANK", "0")}: peak CUDA allocated: '
          f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
    trainer.save_model(str(out/'adapter')); 
    if trainer.is_world_process_zero():
        tok.save_pretrained(out/'adapter')
        manifest = {**CONTRACT, "run_identity": identity, "data_sha256": data_hashes,
                    "torch_runtime": torch.__version__, "cuda_runtime": torch.version.cuda,
                    "selection": "minimum val loss across checkpoints including final step"}
        config = model.peft_config["default"]
        config.base_model_name_or_path = CONTRACT["base_model"]
        config.revision = CONTRACT["revision"]
        config.save_pretrained(out / "adapter")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "inference_contract.json").write_text(json.dumps(CONTRACT, ensure_ascii=False, indent=2), encoding="utf-8")
        (out/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
if __name__=='__main__': main()
