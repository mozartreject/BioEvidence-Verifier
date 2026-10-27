"""Generic relevance diagnostics; no benchmark annotations or IDs in features."""
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from core import feature_vector
NEGATIONS={'not','no','without','neither','nor','never','absence','absent','lack','lacks'}

def tokens(s):return set(re.findall(r'[a-z0-9]+',s.lower()))
def overlap_features(claim,title,passage):
    q=tokens(claim)-ENGLISH_STOP_WORDS;t=tokens(title);p=tokens(passage)
    qn=set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)',claim));pn=set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)',passage))
    return [len(q&p)/max(1,len(q)),len(q&t)/max(1,len(q)),len(q&p)/max(1,len(q|p)),
        float(bool(tokens(claim)&NEGATIONS)),float(bool(p&NEGATIONS)),
        len(qn&pn)/max(1,len(qn)),float(bool(qn-pn)),np.log1p(len(p)),np.log1p(len(q))]

def extract(record,passage_scores):
    pairs=sorted(record['pairs'],key=lambda p:p['relevance'],reverse=True)
    byid=dict(zip([p['document_id'] for p in record['pairs']],passage_scores))
    extra=[]
    for p in pairs:
        score=float(byid[p['document_id']]);raw=np.asarray(p['raw'])
        relevance=1/(1+np.exp(-np.clip(score,-50,50)))
        extra.extend([score,relevance,*overlap_features(record['claim'],p['title'],p['selected']['text']),*(raw*relevance)])
    value=np.concatenate([feature_vector(pairs),extra])
    if not np.isfinite(value).all():raise ValueError('Nonfinite passage features')
    return value
