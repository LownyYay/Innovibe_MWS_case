"""Measure batch-1 service inference without model loading or warm-up."""
import argparse
import json
from pathlib import Path
from statistics import mean
import time

from service import Pipeline, ROOT, public_row


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('calls', type=Path, help='Public JSONL (transcript and call metadata)')
    ap.add_argument('--device', choices=('cpu','cuda'), required=True)
    ap.add_argument('--mode', choices=('span','repair','full'), default='full')
    ap.add_argument('--adapter', type=Path, default=ROOT/'weights/temporal_A')
    args = ap.parse_args()
    import resource
    import platform
    import torch

    rows = [public_row(json.loads(line)) for line in args.calls.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        ap.error('provide at least one JSONL row')
    pipe = Pipeline(args.device, args.mode, args.adapter)

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    pipe(rows[0])
    sync()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for row in rows:
        start = time.perf_counter()
        pipe(row)
        sync()
        durations.append(time.perf_counter()-start)
    result = {'device':args.device, 'mode':args.mode, 'calls':len(rows), 'batch_size':1,
              'mean_seconds':mean(durations),
              'p95_seconds':sorted(durations)[(95*len(rows)+99)//100-1],
              'calls_per_hour':3600/mean(durations), 'torch':torch.__version__,
              'platform':platform.platform(),
              'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                   * (1 if platform.system() == 'Darwin' else 1024)}
    if args.device == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(0), gpu_wall_seconds_per_1000_calls=1000*mean(durations),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(json.dumps(result, indent=2))
