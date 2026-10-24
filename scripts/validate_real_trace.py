#!/usr/bin/env python3
"""Validate collected trace causality and report descriptive statistics (no model requests)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from swarmleak.data.schemas import Trace
from swarmleak.experiments.runner import trace_is_synthetic


def validate(path: Path) -> dict:
    tr = Trace.load_json(path)
    if trace_is_synthetic(tr) or not tr.source.startswith('llm:'):
        raise ValueError('expected a real LLM trace')
    if tr.generator.get('scheduler') != 'causal_events_v2':
        raise ValueError('expected causal generator v2')
    if not tr.messages or any(not 0 <= m.t < tr.duration for m in tr.messages):
        raise ValueError('messages outside mission interval')
    if any(not 0 <= t < tr.duration for t in tr.plan_changes):
        raise ValueError('plan ground truth outside mission interval')
    state = {}
    activations = sorted(tr.generator['task_activations'], key=lambda a: a['t'])
    index = 0
    reports = sorted(tr.generator['report_tasks'], key=lambda r: r['trigger_t'])
    for report in reports:
        while index < len(activations) and activations[index]['t'] <= report['trigger_t']:
            a = activations[index]
            state[a['worker']] = a['task']
            index += 1
        if report['task'] != state.get(report['worker'], 'hold position'):
            raise ValueError('report uses a future or stale task')
    calls = [json.loads(line) for line in path.with_suffix('.calls.jsonl').read_text().splitlines()]
    if any(c.get('finish_reason') == 'length' for c in calls):
        raise ValueError('truncated LLM output in collection')
    if len(calls) != tr.generator['model_calls']:
        raise ValueError('call audit count does not match trace metadata')
    by_kind = {}
    for kind in sorted({m.kind for m in tr.messages}):
        msgs = [m for m in tr.messages if m.kind == kind]
        sizes = np.array([m.size for m in msgs])
        gaps = np.diff([m.t for m in msgs])
        by_kind[kind] = {'count': len(msgs), 'size_bytes_median': float(np.median(sizes)),
                         'size_bytes_p95': float(np.percentile(sizes, 95)),
                         'iat_s_median': float(np.median(gaps)) if len(gaps) else None}
    return {'trace': str(path), 'validated': True, 'model': tr.generator['model'], 'n': tr.n_agents,
            'duration': tr.duration, 'messages': len(tr.messages), 'plan_changes': len(tr.plan_changes),
            'model_calls': len(calls), 'call_latency_s_median': float(np.median([c['latency_s'] for c in calls])),
            'finish_reasons': dict(Counter(c.get('finish_reason') for c in calls)), 'by_kind': by_kind}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', type=Path, nargs='+')
    args = ap.parse_args()
    print(json.dumps([validate(path) for path in args.paths], indent=2))
