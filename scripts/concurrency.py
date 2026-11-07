#!/usr/bin/env python3
"""Concurrency capacity: N parallel streams at a given context size, per recipe.

Usage:
  python3 scripts/concurrency.py --streams 2 4 --prompt-tokens 4096 32768 \
      --max-tokens 256 --out logs/concurrency --runs 1
Stdlib only. Reports per-stream TTFT, per-stream decode tok/s, total throughput,
and any rejections.
"""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:30001').rstrip('/')
MODEL = os.environ.get('SERVED_MODEL_NAME', 'ripple-65b-nvfp4')
FILLER = ('The quick brown fox jumps over the lazy dog beside the quiet river, while '
          'the early riser counts tokens that nobody else will ever read. ')


def headers():
    result = {'Content-Type': 'application/json'}
    key = os.environ.get('SGLANG_API_KEY')
    if key:
        result['Authorization'] = f'Bearer {key}'
    return result


def one_stream(prompt_tokens, max_tokens):
    body = {'model': MODEL,
            'messages': [{'role': 'user', 'content': FILLER * (prompt_tokens // 32 + 1)}],
            'temperature': 0, 'max_tokens': max_tokens, 'stream': True,
            'stream_options': {'include_usage': True}}
    req = urllib.request.Request(BASE + '/v1/chat/completions', headers=headers(),
                                 data=json.dumps(body).encode())
    start = time.monotonic()
    first = None
    last = None
    usage = None
    done = False
    err = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if 'error' in event:
                err = str(event['error'])[:200]
                break
            if event.get('usage'):
                usage = event['usage']
            for choice in (event.get('choices') or []):
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content'):
                    now = time.monotonic()
                    if first is None:
                        first = now
                    last = now
    wall = time.monotonic() - start
    ttft = (first - start) if first else None
    decode = None
    if last and first and usage and usage['completion_tokens'] > 1:
        decode = (usage['completion_tokens'] - 1) / (last - first)
    return {'done': done, 'error': err, 'ttft_seconds': round(ttft, 2) if ttft else None,
            'decode_tok_s': round(decode, 1) if decode else None,
            'wall_seconds': round(wall, 1), 'usage': usage}


def run_case(streams, prompt_tokens, max_tokens, runs):
    rows = []
    for run in range(runs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
            futures = [pool.submit(one_stream, prompt_tokens, max_tokens)
                       for _ in range(streams)]
            start = time.monotonic()
            results = [f.result() for f in futures]
        wall = time.monotonic() - start
        rejected = sum(1 for r in results if r['error'])
        total_out = sum((r['usage'] or {}).get('completion_tokens', 0) for r in results)
        decode_vals = [r['decode_tok_s'] for r in results if r['decode_tok_s']]
        row = {'run': run, 'streams': streams, 'requested_prompt_tokens': prompt_tokens,
               'max_tokens': max_tokens, 'wall_seconds': round(wall, 1),
               'rejections': rejected,
               'min_decode_tok_s': round(min(decode_vals), 1) if decode_vals else None,
               'mean_decode_tok_s': round(sum(decode_vals) / len(decode_vals), 1) if decode_vals else None,
               'total_tok_s': round(total_out / wall, 1) if wall else None,
               'max_ttft_seconds': max((r['ttft_seconds'] for r in results if r['ttft_seconds']), default=None),
               'results': results}
        rows.append(row)
        print(f'streams={streams:>2} prompt={prompt_tokens:>7} run={run} | wall {row["wall_seconds"]:>7.1f}s'
              f' | rejections {rejected} | min decode {row["min_decode_tok_s"]} tok/s'
              f' | total {row["total_tok_s"]} tok/s | max TTFT {row["max_ttft_seconds"]}s')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--streams', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--prompt-tokens', type=int, nargs='+', default=[4096, 32768])
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--out', type=os.fspath, default=None)
    args = parser.parse_args()
    all_rows = []
    for streams in args.streams:
        for prompt_tokens in args.prompt_tokens:
            all_rows.extend(run_case(streams, prompt_tokens, args.max_tokens, args.runs))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'concurrency.json'
        path.write_text(json.dumps({'model': MODEL, 'base': BASE, 'rows': all_rows}, indent=2) + '\n')
        print(f'receipt: {path}')


if __name__ == '__main__':
    main()
