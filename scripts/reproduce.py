#!/usr/bin/env python3
"""Run the steps of configs/pipeline.json in order (each step is `python -m nextact.<module> ...`).

Examples:
  python scripts/reproduce.py --dry-run                      # print every command
  python scripts/reproduce.py --only lexical,lexical_analysis,exposure   # CPU-only stages
  python scripts/reproduce.py --only icl --corpora sgd       # one GPU stage for one corpus

Environment: NEXTACT_DATA, NEXTACT_PRIVATE, NEXTACT_RESULTS (see src/nextact/paths.py and README.md).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def commands(config, only=None, corpora=None):
    for stage in config['stages']:
        if only and stage['stage'] not in only:
            continue
        chosen = [c for c in stage['corpora'] if c is not None and (not corpora or c in corpora)]
        for step in stage['steps']:
            per_corpus = any('{corpus}' in s for s in step)
            for corpus in (chosen if per_corpus else [None]):
                args = [s.replace('{corpus}', corpus) for s in step] if per_corpus else list(step)
                yield stage['stage'], stage['device'], [sys.executable, '-m', f'nextact.{args[0]}', *args[1:]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default=str(REPO / 'configs' / 'pipeline.json'))
    ap.add_argument('--only', default=None, help='comma-separated stage names')
    ap.add_argument('--corpora', default=None, help='comma-separated corpora (sgd, star, ami)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    config = json.loads(Path(a.config).read_text())
    only = set(a.only.split(',')) if a.only else None
    corpora = set(a.corpora.split(',')) if a.corpora else None
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO / 'src')] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    for stage, device, cmd in commands(config, only, corpora):
        print(f'[{stage}, {device}] ' + ' '.join(cmd[1:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
