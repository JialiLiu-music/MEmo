#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/music-emotion-project')
AUDIO = ROOT / 'data/audio_wav/deam_0076.wav'
OUT = ROOT / 'data/structure_allin1_singleprocess_smoke'
DEMIX = OUT / 'demix'
SPEC = OUT / 'spec'
META = OUT / 'smoke_meta.json'


def mark(stage: str, **extra: object) -> None:
    payload = {
        'stage': stage,
        'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        **extra,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mark('start', audio=str(AUDIO), output=str(OUT), pid=os.getpid())

    import torch
    mark(
        'torch',
        version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_build=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    import natten
    mark('natten', version=getattr(natten, '__version__', 'unknown'))

    import allin1
    mark('allin1_imported')

    result = allin1.analyze(
        AUDIO,
        out_dir=OUT,
        model='harmonix-fold0',
        device='cuda',
        demix_dir=DEMIX,
        spec_dir=SPEC,
        keep_byproducts=True,
        overwrite=True,
        multiprocess=False,
    )
    elapsed = round(time.time() - started, 3)
    mark('analyze_returned', elapsed_sec=elapsed, result_type=type(result).__name__)

    json_files = sorted(str(p) for p in OUT.glob('*.json'))
    META.write_text(json.dumps({'elapsed_sec': elapsed, 'json_files': json_files}, indent=2), encoding='utf-8')
    mark('done', elapsed_sec=elapsed, json_files=json_files)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        mark('exception', error=repr(exc))
        raise