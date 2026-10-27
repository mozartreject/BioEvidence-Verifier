"""Audit a blinded relation reviewer on a fixed, stratified SciFact dev subset.

This is a development diagnostic, not a held-out clinical accuracy claim.
Each response is cached so an interrupted run resumes without repeating calls.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np

from core import LABELS, claim_label
from relation_review import review_relation

ROOT = Path(__file__).resolve().parent
CLAIMS = Path.home() / 'Downloads/phase2/data/scifact/claims_dev.jsonl'
CASES = ROOT / 'clean-selector-run/cases/dev'
MODEL = ROOT / 'clean-selector-combiner/experimental-model.joblib'


def sample_cases(per_class=10):
    claims = [json.loads(line) for line in CLAIMS.read_text().splitlines()]
    grouped = {i: [] for i in range(3)}
    for claim in claims:
        label = claim_label(claim)
        if label is not None and (CASES / f"{claim['id']}.json").exists():
            grouped[label].append(claim)
    rng = random.Random(20261027)
    return [claim for label in range(3) for claim in rng.sample(grouped[label], per_class)]


def summarize(rows):
    completed = [r for r in rows if r.get('review')]
    return {
        'completed': len(completed),
        'classifier_accuracy': sum(r['classifier'] == r['gold'] for r in completed) / len(completed) if completed else None,
        'reviewer_accuracy': sum(r['review'] == r['gold'] for r in completed) / len(completed) if completed else None,
        'by_class': {name: {
            'count': sum(r['gold'] == name for r in completed),
            'classifier_correct': sum(r['gold'] == name and r['classifier'] == name for r in completed),
            'reviewer_correct': sum(r['gold'] == name and r['review'] == name for r in completed),
        } for name in LABELS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=10)
    parser.add_argument('--output', type=Path, default=ROOT/'relation-review-audit.json')
    args = parser.parse_args()
    cases = sample_cases(args.per_class)
    saved = json.loads(args.output.read_text()) if args.output.exists() else {}
    model = joblib.load(MODEL)
    feature_data = json.loads((ROOT/'clean-selector-combiner/dev-features.json').read_text())
    features = dict(zip(feature_data['ids'], feature_data['features']))
    for index, claim in enumerate(cases, 1):
        key = str(claim['id'])
        if key in saved:
            continue
        record = json.loads((CASES/f'{key}.json').read_text())
        gold = LABELS[claim_label(claim)]
        classifier = LABELS[int(model.predict(np.asarray(features[key])[None,:])[0])]
        try:
            reviewed = review_relation(record['claim'], record['pairs'])
            outcome = dict(id=key, claim=record['claim'], gold=gold, classifier=classifier,
                           review=reviewed['verdict'], citations=reviewed['citations'])
        except Exception as exc:
            outcome = dict(id=key, claim=record['claim'], gold=gold, classifier=classifier,
                           review=None, error=f'{type(exc).__name__}: {exc}')
        saved[key] = outcome
        args.output.write_text(json.dumps(saved, indent=2))
        print(f'{index}/{len(cases)} {key}: gold={gold} classifier={classifier} review={outcome["review"]}', flush=True)
    report = summarize(list(saved.values()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
