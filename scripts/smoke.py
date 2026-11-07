#!/usr/bin/env python3
"""OpenAI-compatible API checks for the Ripple-65B SGLang server. Stdlib only."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get('BASE_URL', 'http://127.0.0.1:30001').rstrip('/')
MODEL = os.environ.get('SERVED_MODEL_NAME', 'ripple-65b-nvfp4')
TIMEOUT = float(os.environ.get('SMOKE_TIMEOUT', '300'))


def headers():
    result = {'Content-Type': 'application/json'}
    key = os.environ.get('SGLANG_API_KEY')
    if key:
        result['Authorization'] = f'Bearer {key}'
    return result


def request(path, body=None, timeout=TIMEOUT):
    req = urllib.request.Request(BASE + path, headers=headers(),
                                 data=None if body is None else json.dumps(body).encode())
    return urllib.request.urlopen(req, timeout=timeout)


def call(path, body=None, timeout=TIMEOUT):
    with request(path, body, timeout) as response:
        return json.load(response)


def payload(prompt, max_tokens=512):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1, 'max_tokens': max_tokens}


def require_running_container():
    name = os.environ.get('CONTAINER_NAME')
    if name:
        state = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name],
                               text=True, capture_output=True, timeout=10)
        if state.returncode or state.stdout.strip() != 'true':
            raise RuntimeError('Serving container is not running; inspect ./logs.sh')


def ready(wait=0):
    deadline = time.monotonic() + wait
    heartbeat = time.monotonic()
    while True:
        require_running_container()
        try:
            with request('/health', timeout=5) as response:
                if response.status != 200:
                    raise ValueError('Health endpoint is not ready')
            models = call('/v1/models', timeout=5)
            if MODEL not in [m['id'] for m in models.get('data', [])]:
                raise ValueError(f'API does not list expected model {MODEL}')
            require_running_container()
            return models
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'API readiness failed: {exc}') from exc
            if time.monotonic() - heartbeat >= 30:
                print('Waiting for weights, kernel compilation and graph capture...', flush=True)
                heartbeat = time.monotonic()
            time.sleep(min(5, max(0, deadline - time.monotonic())))


def stream(body):
    body = dict(body, stream=True, stream_options={'include_usage': True})
    start = time.monotonic()
    first = None
    content = []
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
                raise ValueError(str(event['error']))
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                text = choice.get('delta', {}).get('content')
                if text:
                    if first is None:
                        first = time.monotonic() - start
                    content.append(text)
    if not done or not content:
        raise ValueError('Stream must return nonempty content and finish with [DONE]')
    return {'content': ''.join(content), 'usage': usage, 'ttft_seconds': first,
            'elapsed_seconds': time.monotonic() - start}


def run_checks(suite=False, output=None):
    results = []

    def save(name, result):
        if output:
            output.mkdir(parents=True, exist_ok=True)
            path = output / (name + '.json')
            if path.exists():
                raise ValueError(f'Refusing to overwrite existing smoke receipt: {path}')
            path.write_text(json.dumps(result, indent=2) + '\n')

    def check(name, fn):
        start = time.monotonic()
        try:
            value = fn()
            save(name, value)
            result = {'check': name, 'passed': True, 'seconds': time.monotonic() - start}
            print('PASS ' + name)
        except (KeyError, IndexError, TypeError, ValueError, OSError, RuntimeError) as exc:
            result = {'check': name, 'passed': False, 'error': str(exc),
                      'seconds': time.monotonic() - start}
            print('FAIL ' + name + ': ' + str(exc), file=sys.stderr)
        results.append(result)

    def completion():
        r = call('/v1/chat/completions', payload('Reply with exactly the single word: ready'))
        save('completion-response', r)
        text = r['choices'][0]['message']['content']
        if not isinstance(text, str) or text.strip().lower() != 'ready':
            raise ValueError('Completion did not obey the ready instruction; inspect completion-response.json')
        return r

    check('completion', completion)

    if suite:
        def reasoning_check():
            r = call('/v1/chat/completions', payload('Compute 17 * 24 and reply with only the number.', 1024))
            save('reasoning-response', r)
            msg = r['choices'][0]['message']
            text = msg.get('content') or ''
            reasoning = msg.get('reasoning_content') or ''
            if '408' not in text:
                raise ValueError('Expected 408 in the final answer content')
            return {'answer_ok': True, 'reasoning_chars': len(reasoning),
                    'usage': r.get('usage')}

        def tool_check():
            b = payload('Use add_numbers to calculate 23 + 19.', 1024)
            b['tools'] = [{'type': 'function', 'function': {
                'name': 'add_numbers', 'description': 'Add two integers.', 'strict': True,
                'parameters': {'type': 'object', 'properties': {
                    'a': {'type': 'integer'}, 'b': {'type': 'integer'}
                }, 'required': ['a', 'b'], 'additionalProperties': False}}}]
            b['tool_choice'] = 'auto'
            r = call('/v1/chat/completions', b)
            save('tools-response', r)
            tools = r['choices'][0]['message'].get('tool_calls') or []
            if len(tools) != 1 or tools[0]['function']['name'] != 'add_numbers':
                raise ValueError('Expected one add_numbers tool call')
            args = json.loads(tools[0]['function']['arguments'])
            if (not isinstance(args, dict) or args.get('a') != 23 or args.get('b') != 19):
                raise ValueError('Tool arguments failed validation')
            # Round trip: append the call and the tool result, expect the computed sum.
            r2 = call('/v1/chat/completions', {
                'model': MODEL,
                'messages': [
                    {'role': 'user', 'content': 'Use add_numbers to calculate 23 + 19.'},
                    r['choices'][0]['message'],
                    {'role': 'tool', 'tool_call_id': tools[0]['id'], 'content': '42'},
                ],
                'temperature': 0, 'max_tokens': 1024})
            save('tools-roundtrip-response', r2)
            text = r2['choices'][0]['message'].get('content') or ''
            if '42' not in text:
                raise ValueError('Round trip did not produce the computed sum')
            return {'call': tools[0], 'roundtrip_ok': True, 'usage': r2.get('usage')}

        check('reasoning-math', reasoning_check)
        check('tools', tool_check)
        check('streaming', lambda: stream(payload('Write one sentence about the Moon.')))
    return all(r['passed'] for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='store_true', help='run the full suite')
    parser.add_argument('--wait', type=int, default=0, help='wait up to N seconds for readiness')
    parser.add_argument('--output', type=os.fspath, default=None,
                        help='write JSON receipts into this directory')
    args = parser.parse_args()
    ready(args.wait)
    ok = run_checks(args.suite, Path(args.output) if args.output else None)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
