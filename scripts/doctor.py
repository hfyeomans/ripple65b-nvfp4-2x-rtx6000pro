#!/usr/bin/env python3
"""Preflight checks for the Ripple-65B NVFP4 SGLang recipe. Python standard library only."""
import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((REPO_ROOT / 'config/model-lock.json').read_text())


def run(cmd, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check(name, fn):
    try:
        detail = fn()
        print(f'PASS {name}' + (f': {detail}' if detail else ''))
        return True
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'FAIL {name}: {exc}', file=sys.stderr)
        return False


def gpu_info():
    out = run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,driver_version',
               '--format=csv,noheader,nounits'])
    if out.returncode:
        raise RuntimeError('nvidia-smi failed')
    info = {}
    for line in out.stdout.strip().splitlines():
        idx, total, free, driver = [f.strip() for f in line.split(',')]
        info[int(idx)] = (int(total), int(free), driver)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', default=os.environ.get('GPUS', '0,1'))
    ap.add_argument('--min-free-mib', type=int,
                    default=int(os.environ.get('MIN_GPU_FREE_MIB', '88000')))
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', '30001')))
    ap.add_argument('--image', default=os.environ.get(
        'IMAGE', 'lmsysorg/sglang:v0.5.19@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9'))
    ap.add_argument('--model-dir', default=os.environ.get('MODEL_SNAPSHOT_DIR', ''))
    ap.add_argument('--draft-dir', default=os.environ.get('DRAFT_SNAPSHOT_DIR', ''))
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    hub = os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface')
    model_dir = Path(args.model_dir or (hub
                                        + f'/hub/models--ricursiveAI--Ripple-65B-Preview-NVFP4/snapshots/{LOCK["target"]["revision"]}'))
    draft_dir = Path(args.draft_dir or (hub
                                        + f'/hub/models--ricursiveAI--Ripple-65B-Preview-DFlash2/snapshots/{LOCK["draft"]["revision"]}'))
    ok = True

    def docker_daemon():
        out = run(['docker', 'info'])
        if out.returncode:
            raise RuntimeError('Docker daemon unavailable')
        return 'running'

    def image_present():
        out = run(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'])
        if out.returncode:
            raise RuntimeError(f'image {args.image} not present; run ./prepare.sh')
        return out.stdout.strip()[:12]

    def gpus_free():
        info = gpu_info()
        seen = []
        for idx in gpus:
            if idx not in info:
                raise RuntimeError(f'GPU {idx} not visible to nvidia-smi')
            total, free, _ = info[idx]
            seen.append(f'{idx}:{free}MiB free/{total}MiB')
            if free < args.min_free_mib:
                raise RuntimeError(f'GPU {idx} has {free} MiB free (< {args.min_free_mib}); stop other GPU users')
        return ', '.join(seen)

    def model_complete():
        target = LOCK['target']
        if not model_dir.is_dir():
            raise RuntimeError(f'NVFP4 snapshot not found: {model_dir}')
        cfg = json.loads((model_dir / 'config.json').read_text())
        if cfg.get('architectures') != ['RicursiveForCausalLM']:
            raise RuntimeError(f'config.json architectures: {cfg.get("architectures")}')
        quant = cfg.get('quantization_config') or {}
        if quant.get('quant_method') != 'modelopt' or quant.get('quant_algo') != 'NVFP4':
            raise RuntimeError(f'expected ModelOpt NVFP4 quantization, got {quant.get("quant_method")}/{quant.get("quant_algo")}')
        for name in ('tokenizer.json', 'chat_template.jinja', 'model.safetensors.index.json'):
            if not (model_dir / name).is_file():
                raise RuntimeError(f'missing {name}')
        idx = json.loads((model_dir / 'model.safetensors.index.json').read_text())
        shards = sorted(set(idx['weight_map'].values()))
        if len(shards) != target['shards'] or idx['metadata']['total_size'] != target['total_size_bytes']:
            raise RuntimeError(f'index mismatch: {len(shards)} shards, '
                               f'{idx["metadata"]["total_size"]} bytes (lock expects '
                               f'{target["shards"]} shards, {target["total_size_bytes"]} bytes)')
        for shard in shards:
            if not (model_dir / shard).is_file():
                raise RuntimeError(f'missing shard {shard}')
        return f'{len(shards)} shards, {target["total_size_bytes"] / 2**30:.1f} GiB, revision {target["revision"]}'

    def draft_complete():
        if not draft_dir.is_dir():
            raise RuntimeError(f'DFlash2 draft snapshot not found: {draft_dir}')
        cfg = json.loads((draft_dir / 'config.json').read_text())
        if cfg.get('architectures') != ['DFlash2DraftModel']:
            raise RuntimeError(f'draft config.json architectures: {cfg.get("architectures")}')
        if (draft_dir / 'model.safetensors').stat().st_size <= 0:
            raise RuntimeError('draft model.safetensors missing or empty')
        return f'DFlash2DraftModel, revision {LOCK["draft"]["revision"]}'

    def port_free():
        host = os.environ.get('HOST', '0.0.0.0')
        bind = '127.0.0.1' if host in ('0.0.0.0', '::') else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            if s.connect_ex((bind, args.port)) == 0:
                raise RuntimeError(f'port {args.port} is already in use on {bind}')
        return f'{args.port} free'

    ok &= check('docker daemon', docker_daemon)
    ok &= check('serving image', image_present)
    ok &= check(f'gpus free ({args.gpus})', gpus_free)
    ok &= check('nvfp4 snapshot', model_complete)
    ok &= check('dflash2 draft', draft_complete)
    ok &= check('api port', port_free)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
