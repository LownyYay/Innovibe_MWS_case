"""CompactFactExtraction HTTP inference for one Russian support call."""
import argparse
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sys
from threading import Lock
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in ('extractor', 'repair', 'temporal_stage'):
    sys.path.insert(0, str(HERE/folder))

import core
from apply_repair import repair_call
from joint import candidate_cache, decode_cached
from model_io import CHAT, CONTRACT, check_tokenizer, messages as extractor_messages
from temporal import boundary_role, iso_day, mentions, resolve_mention
from validation import decode_prediction
from http_schema import ExtractResponse

LOG = logging.getLogger(__name__)
MAX_BODY = 512 * 1024


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def public_row(value):
    if not isinstance(value, dict):
        raise ValueError('Expected a call object')
    call_id, started_at, turns = (value.get(k) for k in ('call_id', 'started_at', 'turns'))
    if not isinstance(call_id, str) or not call_id or not isinstance(started_at, str):
        raise ValueError('call_id and started_at are required')
    datetime.fromisoformat(started_at)
    if not isinstance(turns, list) or not turns:
        raise ValueError('turns must be a non-empty array')
    clean = []
    seen = set()
    for t in turns:
        if (not isinstance(t, dict) or type(t.get('idx')) is not int or
            t['idx'] < 0 or t['idx'] in seen or t.get('speaker') not in ('client', 'operator') or
            not isinstance(t.get('text'), str)):
            raise ValueError('Each turn needs a unique non-negative idx, client/operator speaker and text')
        seen.add(t['idx'])
        clean.append({k: t[k] for k in ('idx','speaker','text')})
    return {'call_id':call_id, 'started_at':started_at, 'turns':clean}


def validate_http_response(row, answer):
    response = ExtractResponse.model_validate(answer)
    if response.call_id != row['call_id']:
        raise ValueError('Response call_id differs from the request')
    turns = {turn['idx']: turn for turn in row['turns']}
    facts = []
    issues = 0
    for fact in response.facts:
        source = fact.source
        turn = turns[source.turn_idx]
        if fact.polarity == -1 and turn['speaker'] != 'client':
            issues += 1
            continue
        if fact.speaker != turn['speaker']:
            fact.speaker = turn['speaker']
            issues += 1
        text = turn['text']
        if source.span is not None:
            start, end = source.span
            if not 0 <= start < end <= len(text):
                source.span = None
                issues += 1
        if source.quote is not None:
            if source.span is None:
                grounded = bool(source.quote.strip()) and source.quote in text
            else:
                start, end = source.span
                grounded = source.quote == text[start:end]
            if not grounded:
                source.quote = None
                issues += 1
        facts.append(fact)
    response.facts = facts
    payload = response.model_dump(exclude_unset=True)
    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return payload, issues


def apply_explicit_repeat(row, prediction, raw):
    """Reconstruct the repeat rule from the documented policy and frozen guard."""
    raw_facts = json.loads(raw['text'])['facts']
    turns = {t['idx']:t['text'] for t in row['turns']}
    base = iso_day(row['started_at'])
    for fact, rf in zip(prediction['facts'], raw_facts, strict=True):
        if fact['attribute'] not in core.ATTRS:
            continue
        ev = rf.get('valid_from')
        if not isinstance(ev, dict) or ev.get('status') != 'explicit':
            continue
        idx, expr = ev.get('turn_idx'), ev.get('expression')
        if idx not in turns or not isinstance(expr, str) or not expr:
            continue
        hits = [match for match in mentions(turns[idx]) if match.group() == expr]
        if len(hits) < 2 or not core.protected(row, rf, 'valid_from'):
            continue
        start = resolve_mention(hits[0], base, boundary_role(fact['attribute'], 'valid_from'))
        old, end = fact['valid_from'], fact.get('valid_to')
        if old[:10] != start[:10] and (end is None or start[:10] <= end[:10]):
            fact['valid_from'] = start


def repair_public(row, pred, raw, ranker):
    pred, _ = repair_call(row, pred, raw, ranker)
    if pred['facts']:
        cache = candidate_cache(row, pred, raw, ranker)
        decode_cached(row, pred, cache, ranker['joint_policy'])
        apply_explicit_repeat(row, pred, raw)
    return pred


class PromptTooLong(ValueError):
    pass


class Generator:
    def __init__(self, base, adapter, device, dtype, generation, limit):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.torch, self.device, self.limit = torch, device, limit
        token = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        check_tokenizer(token)
        self.tokenizer = token
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, torch_dtype=dtype, attn_implementation='sdpa'
        )
        self.model = PeftModel.from_pretrained(model, adapter).to(device).eval()
        self.generation = GenerationConfig(**generation)

    def __call__(self, messages, decode):
        torch = self.torch
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, **CHAT, return_tensors='pt')
        if ids.shape[1] + self.generation.max_new_tokens > self.limit:
            raise PromptTooLong('Prompt and generation reserve exceed the configured context')
        ids = ids.to(self.device)
        with torch.inference_mode():
            result = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                generation_config=self.generation, use_model_defaults=False,
                do_sample=self.generation.do_sample,
            )
        ids = result[0, ids.shape[1]:]
        exhausted = len(ids) >= self.generation.max_new_tokens
        ended = bool(len(ids) and ids[-1].item() == self.tokenizer.eos_token_id)
        return self.tokenizer.decode(ids, **decode), exhausted, ended


class Pipeline:
    def __init__(self, device, mode, adapter, base_model=None):
        import torch

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        record = json.loads((ROOT/'artifact_identity.json').read_text(encoding='utf-8'))
        self.variant = record['trial_id']
        temporal_weights = Path(adapter)/'adapter_model.safetensors'
        if mode == 'full' and (not temporal_weights.is_file() or sha(temporal_weights) != record['adapter_sha256']):
            raise ValueError('Full mode requires the exact verified experiment A adapter')
        extractor = ROOT/'weights/extractor'
        if sha(extractor/'adapter_model.safetensors') != record['extractor_sha256']:
            raise ValueError('Extractor adapter hash mismatch')
        base_model = base_model or os.environ.get('FACT_BASE_MODEL')
        if base_model:
            base = Path(base_model)
        else:
            from huggingface_hub import snapshot_download

            base = Path(snapshot_download(CONTRACT['base_model'], revision=CONTRACT['revision'],
                                          allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']))
        for name, digest in CONTRACT['tokenizer']['sha256_source'].items():
            if sha(base/name) != digest:
                raise ValueError('Pinned base tokenizer/config SHA mismatch: '+name)
        self.extract = Generator(base, extractor, device,
                                 torch.float16 if device == 'cuda' else torch.float32,
                                 CONTRACT['generation'], 16384)
        self.mode = mode
        self.ranker = json.loads((HERE/'repair/date_ranker.json').read_text(encoding='utf-8'))
        if sha(HERE/'repair/candidates.py') != self.ranker['feature_source_sha256']:
            raise ValueError('Frozen repair features mismatch')
        if mode == 'full':
            # Match the effective options recorded in the run01 generation log.
            gen = {**CONTRACT['generation'], 'max_new_tokens':768, 'do_sample':True,
                   'temperature':0.6, 'top_k':20, 'top_p':0.95, 'bos_token_id':151643}
            self.propose = Generator(base, adapter, device,
                                     torch.bfloat16 if device == 'cuda' else torch.float32,
                                     gen, 16384)

    def __call__(self, row):
        text, limit, ended = self.extract(extractor_messages(row), CONTRACT['output']['decode'])
        raw = {'call_id':row['call_id'], 'text':text}
        try:
            if limit and not ended:
                raise ValueError('Extractor reached its generation limit')
            pred, _ = decode_prediction(json.loads(text), row)
        except (ValueError, TypeError, KeyError) as exc:
            LOG.warning('Extractor abstention for %s: %s', row['call_id'], exc)
            return {'call_id':row['call_id'], 'facts':[]}
        if self.mode == 'span':
            return pred
        pred = repair_public(row, pred, raw, self.ranker)
        if self.mode == 'repair' or not core.task_indices(pred['facts']):
            return pred
        try:
            text, limit, _ = self.propose(core.messages(row, pred['facts']), {'skip_special_tokens':True})
            if limit:
                raise ValueError('Temporal adapter reached its generation limit')
            dates = core.parse_dates(text, pred['facts'])
            return core.apply_dates(row, raw, pred, dates)[0]
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            LOG.warning('Temporal abstention for %s: %s', row['call_id'], exc)
            return pred


def handler_for(pipeline):
    inference_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, evidence_issues=None):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            if evidence_issues is not None:
                self.send_header('X-Evidence-Issue-Count', str(evidence_issues))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/healthz':
                self.reply(200, {'status':'ok'})
            elif self.path in ('/health', '/readyz'):
                self.reply(200, {'status':'ready', 'variant':pipeline.variant, 'mode':pipeline.mode})
            else:
                self.reply(404, {'error':'unknown path'})

        def do_POST(self):
            if self.path not in ('/extract', '/v1/extract'):
                self.reply(404, {'error':'unknown path'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size > MAX_BODY:
                    self.reply(413, {'error':'request exceeds 512 KiB'})
                    return
                if size <= 0:
                    raise ValueError('Expected a non-empty JSON request')
                row = public_row(json.loads(self.rfile.read(size)))
            except (ValueError, RecursionError) as exc:
                self.reply(422, {'error':str(exc)})
                return
            start = time.perf_counter()
            try:
                with inference_lock:
                    answer = pipeline(row)
            except PromptTooLong as exc:
                self.reply(413, {'error':str(exc)})
                return
            except Exception:
                LOG.exception('Inference failure')
                self.reply(500, {'error':'inference failed'})
                return
            try:
                answer, evidence_issues = validate_http_response(row, answer)
            except (ValueError, KeyError, RecursionError):
                LOG.warning('Invalid facts contract for %s', row['call_id'])
                self.reply(502, {'error':'model output violates the facts schema'})
                return
            LOG.info('Request finished in %.3fs', time.perf_counter()-start)
            self.reply(200, answer, evidence_issues)
    return Handler


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    ap.add_argument('--mode', choices=('span','repair','full'), default='full')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--adapter', type=Path, default=ROOT/'weights/temporal_A')
    ap.add_argument('--base-model', type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    pipeline = Pipeline(args.device, args.mode, args.adapter, args.base_model)
    with ThreadingHTTPServer((args.host, args.port), handler_for(pipeline)) as server:
        LOG.info('Listening on %s:%d', args.host, args.port)
        server.serve_forever()
