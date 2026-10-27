"""Experimental combiner inference, deliberately not wired into production."""
import json
from pathlib import Path
import numpy as np
from core import LABELS, feature_vector

def predict(pairs, artifact_path):
    artifact=json.loads(Path(artifact_path).read_text())
    if artifact.get('status')!='experimental_not_deployed':raise ValueError('Unknown artifact contract')
    if artifact['classes']!=[0,1,2]:raise ValueError('Unexpected class mapping')
    x=feature_vector(pairs)
    mean=np.asarray(artifact['mean']);scale=np.asarray(artifact['scale'])
    coef=np.asarray(artifact['coefficients']);intercept=np.asarray(artifact['intercept'])
    if mean.shape!=x.shape or scale.shape!=x.shape or coef.shape!=(3,len(x)) or intercept.shape!=(3,):
        raise ValueError('Feature/model shape mismatch')
    if not all(np.isfinite(a).all() for a in [mean,scale,coef,intercept]) or np.any(scale<=0):
        raise ValueError('Invalid artifact values')
    logits=np.einsum('ij,j->i',coef,(x-mean)/scale,optimize=False)+intercept
    scores=np.exp(logits-logits.max());scores/=scores.sum()
    return dict(verdict=LABELS[int(scores.argmax())],scores=dict(zip(LABELS,scores.tolist())),
                fingerprint=artifact['fingerprint'],experimental=True,calibrated=False)
