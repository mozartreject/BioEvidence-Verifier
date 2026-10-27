"""Reject decisive aggregate verdicts with uniformly neutral pairwise evidence."""
def validate_decisive_evidence(result):
    if result.get('verdict') not in ('SUPPORTED','REFUTED'):
        return
    pairs=result.get('evidence',[])
    if not pairs:
        raise ValueError('Evidence validation failed: no passages justify a decisive verdict. No verified conclusion is available.')
    neutral=[]
    direction=2 if result["verdict"]=="SUPPORTED" else 0
    matching=[]
    import math
    for pair in pairs:
        raw=pair.get('raw',[])
        if len(raw)!=3 or any(not math.isfinite(float(x)) or not 0<=float(x)<=1 for x in raw):
            raise ValueError('Evidence validation failed: invalid pairwise probabilities.')
        if abs(sum(raw)-1)>1e-4:
            raise ValueError('Evidence validation failed: pairwise probabilities do not sum to one.')
        neutral.append(raw[1]>=max(raw[0],raw[2]))
        matching.append(raw[direction]>max(raw[i] for i in range(3) if i!=direction))
    if all(neutral):
        raise ValueError('Evidence validation failed: all retrieved passages were classified as neutral, but the aggregate model proposed a decisive verdict. No verified conclusion is available; the claim needs directly relevant evidence.')

    if not any(matching):
        raise ValueError("Evidence validation failed: no retrieved passage favors the proposed verdict. No verified conclusion is available; review the evidence.")
