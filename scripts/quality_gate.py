#!/usr/bin/env python3
"""Quality gate: same fixed prompt set on two KV profiles, then diff the answers.

Usage:
  python3 scripts/quality_gate.py run --label bf16 --out logs/quality
  python3 scripts/quality_gate.py run --label fp8 --out logs/quality
  python3 scripts/quality_gate.py compare --a logs/quality/bf16 --b logs/quality/fp8 \
      --out logs/quality --rerun-flips
Stdlib only. The prompt set is fixed in PROMPTS below; keep it stable across runs so
receipts stay comparable.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:30000').rstrip('/')
BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:30001').rstrip('/')
MODEL = os.environ.get('SERVED_MODEL_NAME', 'ripple-65b-nvfp4')
TIMEOUT = 600


def headers():
    result = {'Content-Type': 'application/json'}
    key = os.environ.get('SGLANG_API_KEY')
    if key:
        result['Authorization'] = f'Bearer {key}'
    return result


def post(body):
    req = urllib.request.Request(BASE + '/v1/chat/completions', headers=headers(),
                                 data=json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.load(response)


def chat(messages, max_tokens=1024, tools=None, tool_choice=None):
    body = {'model': MODEL, 'messages': messages, 'temperature': 0, 'top_p': 1,
            'max_tokens': max_tokens}
    if tools:
        body['tools'] = tools
    if tool_choice:
        body['tool_choice'] = tool_choice
    return post(body)


def answer_text(msg):
    return (msg.get('content') or '').strip()


def tool_args(msg):
    calls = msg.get('tool_calls') or []
    out = []
    for call in calls:
        try:
            out.append({'name': call['function']['name'],
                        'arguments': json.loads(call['function']['arguments'])})
        except (KeyError, ValueError):
            out.append({'name': (call.get('function') or {}).get('name'), 'arguments': None})
    return out


TOOLS_ADD = [{'type': 'function', 'function': {
    'name': 'add_numbers', 'description': 'Add two integers.', 'strict': True,
    'parameters': {'type': 'object', 'properties': {
        'a': {'type': 'integer'}, 'b': {'type': 'integer'}
    }, 'required': ['a', 'b'], 'additionalProperties': False}}}]

TOOLS_TEMP = [{'type': 'function', 'function': {
    'name': 'get_temperature', 'description': 'Read the temperature in a city.',
    'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius']}
    }, 'required': ['city', 'unit'], 'additionalProperties': False}}}]


def make_item(prompt_id, kind, messages, expected=None, tools=None, tool_choice=None,
              max_tokens=1024):
    return {'id': prompt_id, 'kind': kind, 'messages': messages, 'expected': expected,
            'tools': tools, 'tool_choice': tool_choice, 'max_tokens': max_tokens}


PROMPTS = [
    make_item('ready', 'instruction', [
        {'role': 'user', 'content': 'Reply with exactly the single word: ready'}],
        expected='ready', max_tokens=256),
    make_item('mul', 'math', [
        {'role': 'user', 'content': 'Compute 17 * 24 and reply with only the number.'}],
        expected='408'),
    make_item('palindrome', 'code', [
        {'role': 'user', 'content': 'Write a Python function that checks whether a string '
                                    'is a palindrome. Include a docstring and one test.'}],
        expected='def '),
    make_item('binary_search', 'code', [
        {'role': 'user', 'content': 'Implement binary search in Python iteratively. '
                                    'Return only code.'}],
        expected='def '),
    make_item('trains', 'math', [
        {'role': 'user', 'content': 'A train leaves city A at 60 km/h and another leaves '
                                    'city B 300 km away at 90 km/h, heading toward each '
                                    'other. After how many hours do they meet?'}],
        expected='2'),
    make_item('float', 'reasoning', [
        {'role': 'user', 'content': 'Explain why 0.1 + 0.2 != 0.3 in floating point, '
                                    'in two sentences.'}],
        expected='binary'),
    make_item('primes', 'math', [
        {'role': 'user', 'content': 'What is the sum of the first 10 prime numbers?'}],
        expected='129'),
    make_item('wet_ground', 'logic', [
        {'role': 'user', 'content': 'If it rains the ground gets wet. The ground is wet. '
                                    'Does it follow that it rained? Answer yes or no, '
                                    'then explain briefly.'}],
        expected='no'),
    make_item('derivative', 'math', [
        {'role': 'user', 'content': 'What is the derivative of x^2 * ln(x)?'}],
        expected='2x ln x + x'),
    make_item('hex', 'math', [
        {'role': 'user', 'content': 'Convert 0x2A to decimal and to binary.'}],
        expected='42'),
    make_item('haiku', 'creative', [
        {'role': 'user', 'content': 'Write a haiku about a GPU running out of memory.'}]),
    make_item('sql', 'code', [
        {'role': 'user', 'content': 'Write a SQL CREATE TABLE for a users table with '
                                    'id, email, and created_at columns.'}],
        expected='CREATE TABLE'),
    make_item('prime_big', 'math', [
        {'role': 'user', 'content': 'Is 4294967297 prime? Explain how you checked.'}],
        expected='641'),
    make_item('french', 'language', [
        {'role': 'user', 'content': "Translate 'the quick brown fox' to French."}],
        expected='renard'),
    make_item('tool_add', 'tool', [
        {'role': 'user', 'content': 'Use add_numbers to calculate 23 + 19.'}],
        tools=TOOLS_ADD, tool_choice='auto', expected='add_numbers'),
    make_item('tool_required', 'tool', [
        {'role': 'user', 'content': 'Call get_temperature for Paris in celsius.'}],
        tools=TOOLS_TEMP, tool_choice='required', expected='get_temperature'),
    make_item('longest_line', 'code', [
        {'role': 'user', 'content': 'Write a Python script that reads stdin and prints '
                                    'lines longer than 80 characters. Return only code.'}],
        expected='sys.stdin'),
    make_item('sort', 'code', [
        {'role': 'user', 'content': 'Sort [3, 1, 4, 1, 5, 9, 2, 6] with quicksort. Show '
                                    'the final list only.'}],
        expected='1, 1, 2, 3, 4, 5, 6, 9'),
    make_item('oom', 'knowledge', [
        {'role': 'user', 'content': 'What does CUDA out of memory mean, and name two ways '
                                    'to reduce GPU memory use in inference.'}],
        expected='precision'),
    make_item('json_keys', 'format', [
        {'role': 'user', 'content': 'Return a JSON object with key a equal to the string '
                                    'hello and key b equal to 42. Return only the JSON.'}],
        expected='"a"'),
]


def run(label, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for item in PROMPTS:
        path = out / f'{item["id"]}.json'
        if path.exists():
            records.append(json.loads(path.read_text()))
            print(f'skip {item["id"]} (receipt exists)')
            continue
        start = time.monotonic()
        try:
            r = chat(item['messages'], max_tokens=item['max_tokens'],
                     tools=item['tools'], tool_choice=item['tool_choice'])
            msg = r['choices'][0]['message']
            record = {'id': item['id'], 'kind': item['kind'], 'label': label,
                      'content': answer_text(msg),
                      'reasoning_content': msg.get('reasoning_content') or '',
                      'tool_calls': tool_args(msg),
                      'usage': r.get('usage'), 'seconds': time.monotonic() - start}
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            record = {'id': item['id'], 'kind': item['kind'], 'label': label,
                      'error': str(exc), 'seconds': time.monotonic() - start}
        path.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(f'{"ok " if "error" not in record else "ERR"} {item["id"]}')
    return records


def graded(record, item):
    if 'error' in record:
        return False
    if item['kind'] == 'tool':
        calls = record.get('tool_calls') or []
        return bool(calls) and calls[0]['name'] == item['expected']
    if item['expected'] is None:
        return bool(record['content'])
    return item['expected'] in record['content']


def compare(dir_a, dir_b, out, rerun_flips):
    a_dir, b_dir = Path(dir_a), Path(dir_b)
    by_id = {item['id']: item for item in PROMPTS}
    rows = []
    flips = []
    for item in PROMPTS:
        a = json.loads((a_dir / f'{item["id"]}.json').read_text())
        b = json.loads((b_dir / f'{item["id"]}.json').read_text())
        a_ok = graded(a, item)
        b_ok = graded(b, item)
        same = (a.get('content') == b.get('content')
                and a.get('tool_calls') == b.get('tool_calls'))
        rows.append({'id': item['id'], 'kind': item['kind'],
                     'a_ok': a_ok, 'b_ok': b_ok, 'same': same})
        if not same or a_ok != b_ok:
            flips.append(item['id'])
    total = len(rows)
    same_n = sum(1 for r in rows if r['same'])
    a_pass = sum(1 for r in rows if r['a_ok'])
    b_pass = sum(1 for r in rows if r['b_ok'])
    result = {'a': str(a_dir), 'b': str(b_dir), 'total': total,
              'identical_answers': same_n, 'identical_pct': round(100 * same_n / total, 1),
              'a_graded_pass': a_pass, 'b_graded_pass': b_pass, 'flips': flips}
    print(f'identical answers: {same_n}/{total} ({result["identical_pct"]}%)')
    print(f'graded pass: {Path(dir_a).name}={a_pass}/{total}, '
          f'{Path(dir_b).name}={b_pass}/{total}')
    print(f'flips or grade changes: {flips or "none"}')
    for row in rows:
        if row['id'] in flips:
            print(f'  {row["id"]}: same={row["same"]} a_ok={row["a_ok"]} b_ok={row["b_ok"]}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'compare.json').write_text(json.dumps(result, indent=2) + '\n')
    if rerun_flips and flips:
        print(f'rerunning {len(flips)} flips at temperature 0...')
        flips_by_id = {item['id']: item for item in PROMPTS if item['id'] in flips}
        reruns = {}
        for prompt_id, item in flips_by_id.items():
            answers = []
            for _ in range(2):
                r = chat(item['messages'], max_tokens=item['max_tokens'],
                         tools=item['tools'], tool_choice=item['tool_choice'])
                answers.append(answer_text(r['choices'][0]['message']))
            reruns[prompt_id] = answers
        (out / 'reruns.json').write_text(json.dumps(reruns, indent=2) + '\n')
        print(json.dumps(reruns, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_p = sub.add_parser('run')
    run_p.add_argument('--label', required=True)
    run_p.add_argument('--out', default='logs/quality')
    cmp_p = sub.add_parser('compare')
    cmp_p.add_argument('--a', required=True)
    cmp_p.add_argument('--b', required=True)
    cmp_p.add_argument('--out', default='logs/quality')
    cmp_p.add_argument('--rerun-flips', action='store_true')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.label, args.out)
    else:
        compare(args.a, args.b, args.out, args.rerun_flips)


if __name__ == '__main__':
    main()
