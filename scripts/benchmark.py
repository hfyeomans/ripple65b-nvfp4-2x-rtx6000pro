#!/usr/bin/env python3
"""Throughput probe: TTFT, approximate prefill tok/s, decode tok/s. Stdlib only."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

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


def request(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, headers=headers(), data=json.dumps(body).encode())
    return urllib.request.urlopen(req, timeout=timeout)


def probe(prompt_tokens, max_tokens):
    prompt = FILLER * (prompt_tokens // 32 + 1)
    body = {'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': max_tokens,
            'stream': True, 'stream_options': {'include_usage': True}}
    start = time.monotonic()
    first = None
    last = None
    usage = None
    done = False
    with request('/v1/chat/completions', body) as response:
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
                raise RuntimeError(str(event['error'])[:300])
            if event.get('usage'):
                usage = event['usage']
            choices = event.get('choices') or []
            if any((ch.get('delta') or {}).get('content') or (ch.get('delta') or {}).get('reasoning_content')
                   for ch in choices):
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
    if not done or first is None:
        raise RuntimeError(f'probe stream did not finish with tokens (done={done}, usage={usage})')
    prompt_tokens = usage['prompt_tokens']
    completion_tokens = usage['completion_tokens']
    ttft = first - start
    decode = (completion_tokens - 1) / (last - first) if completion_tokens > 1 else 0.0
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'ttft_seconds': round(ttft, 3),
        'approx_prefill_tok_s': round(prompt_tokens / ttft, 1) if ttft else None,
        'decode_tok_s': round(decode, 1),
        'total_seconds': round(time.monotonic() - start, 1),
        'usage': usage,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prompt-tokens', type=int, nargs='+', default=[4096, 32768],
                    help='prompt sizes to probe (default: 4096 32768)')
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--output', type=os.fspath, default=None)
    args = ap.parse_args()
    rows = []
    for n in args.prompt_tokens:
        row = probe(n, args.max_tokens)
        row['requested_prompt_tokens'] = n
        rows.append(row)
        print(f'{n:>7} prompt tokens | TTFT {row["ttft_seconds"]:>7.2f}s | '
              f'prefill ~{row["approx_prefill_tok_s"]:>7.0f} tok/s | '
              f'decode {row["decode_tok_s"]:>7.1f} tok/s | '
              f'{row["completion_tokens"]} out tokens')
    if args.output:
        from pathlib import Path
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        out = path / 'benchmark.json'
        if out.exists():
            raise ValueError(f'Refusing to overwrite existing receipt: {out}')
        out.write_text(json.dumps({'model': MODEL, 'rows': rows}, indent=2) + '\n')
        print(f'receipt: {out}')


if __name__ == '__main__':
    main()
