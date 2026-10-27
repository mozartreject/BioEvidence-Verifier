"""Conservative second retrieval pass when the first answer cannot be justified.

The retry never changes the claim or accepts a decisive model class without
the same passage and independent-relation checks used for the local result.
"""

from evidence_guard import validate_decisive_evidence
from relation_review import require_agreement


def _assessment(claim, result, reviewer):
    if result.get('verdict') not in ('SUPPORTED', 'REFUTED'):
        return None, None
    try:
        validate_decisive_evidence(result)
        review = reviewer(claim, result)
        if not isinstance(review, dict) or review.get('verdict') != result['verdict']:
            raise ValueError('Independent relation review did not confirm the proposed verdict')
        return review, None
    except ValueError as exc:
        return None, exc


def select_result(claim, local_result, live_predict, reviewer=require_agreement):
    """Return (result, review, trace); retain local output if retry is unverified."""
    local_review, local_error = _assessment(claim, local_result, reviewer)
    if local_review is not None:
        return local_result, local_review, {'attempted': False, 'selected': 'local'}
    try:
        live_result = live_predict(claim)
        live_review, live_error = _assessment(claim, live_result, reviewer)
    except Exception as exc:
        return local_result, None, {'attempted': True, 'selected': 'local', 'reason': f'live retrieval unavailable: {type(exc).__name__}'}
    if live_review is not None:
        return live_result, live_review, {'attempted': True, 'selected': 'live', 'reason': 'independently reviewed decisive evidence'}
    return local_result, None, {'attempted': True, 'selected': 'local', 'reason': str(live_error) if live_error else 'live result was not decisive'}
