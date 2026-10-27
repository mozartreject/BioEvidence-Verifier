"""Validate model label semantics before interpreting any output columns."""
CANONICAL_NLI = ('contradiction', 'neutral', 'entailment')


def canonical_order(id2label, num_labels=3):
    if num_labels != 3 or len(id2label) != 3:
        raise ValueError('Expected exactly three NLI labels')
    normalized = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    if set(normalized) != {0, 1, 2} or set(normalized.values()) != set(CANONICAL_NLI):
        raise ValueError('NLI config must uniquely declare contradiction, neutral, entailment at indices 0..2')
    reverse = {label: index for index, label in normalized.items()}
    return [reverse[label] for label in CANONICAL_NLI]
