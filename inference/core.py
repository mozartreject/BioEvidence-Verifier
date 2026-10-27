"""Label-independent sparse retrieval and evidence-combination features."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

LABELS = ('REFUTED', 'NEI', 'SUPPORTED')

class SparseBM25:
    def __init__(self, texts):
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', dtype=np.float64)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        length = np.asarray(counts.sum(axis=1)).ravel()
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((len(texts) - df + .5) / (df + .5))
        rows = np.repeat(np.arange(len(texts)), np.diff(counts.indptr))
        counts.data = counts.data * 2.5 / (counts.data + 1.5 * (.25 + .75 * length[rows] / length.mean())) * idf[counts.indices]
        self.matrix = counts.tocsc()

    def search(self, claim, k=30):
        query = self.vectorizer.transform([claim])
        scores = np.asarray(self.matrix[:, query.indices].sum(axis=1)).ravel()
        # Stable ties and removal of zero-score documents.
        order = np.argsort(-scores, kind='stable')
        return [(int(i), float(scores[i])) for i in order[:k] if scores[i] > 0]

def feature_vector(pairs):
    """Fixed-size, rank-aware features. No IDs, claim text, or annotations."""
    if len(pairs) != 3: raise ValueError('Exactly three scored documents required')
    pairs = sorted(pairs, key=lambda p: p['relevance'], reverse=True)
    raw = np.asarray([p['raw'] for p in pairs], dtype=float)
    rel = np.asarray([p['relevance'] for p in pairs], dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(rel).all(): raise ValueError('Nonfinite input')
    if np.any(raw < 0) or not np.allclose(raw.sum(axis=1), 1): raise ValueError('Invalid probabilities')
    if np.any(rel < 0) or np.any(rel > 1): raise ValueError('Invalid relevance')
    discounted = raw * rel[:, None]; discounted[:, 1] += 1 - rel
    return np.concatenate([raw.ravel(), discounted.ravel(), rel,
        raw.mean(axis=0), raw.max(axis=0), discounted.mean(axis=0),
        np.bincount(raw.argmax(axis=1), minlength=3) / 3,
        [rel[0]-rel[1], rel[1]-rel[2]]])

def baselines(pairs):
    pairs = sorted(pairs, key=lambda p: p['relevance'], reverse=True)
    raw = np.asarray([p['raw'] for p in pairs]); rel = np.asarray([p['relevance'] for p in pairs])
    p = raw * rel[:, None]; p[:, 1] += 1-rel
    return {'mean': int(p.mean(axis=0).argmax()), 'top_document': int(p[0].argmax())}

def claim_label(c):
    labels = {a['label'] for annotations in c.get('evidence', {}).values() for a in annotations}
    if len(labels) > 1 or labels - {'SUPPORT', 'CONTRADICT'}: return None
    return {'SUPPORT': 2, 'CONTRADICT': 0}.get(next(iter(labels), ''), 1)

def linked_groups(claims):
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x: parents[x] = find(parents[x])
        return parents[x]
    for c in claims:
        key = 'claim:' + str(c['id']); find(key)
        for doc in set(map(str, c.get('cited_doc_ids', []))) | set(c.get('evidence', {})):
            parents[find('doc:' + doc)] = find(key)
    return [find('claim:' + str(c['id'])) for c in claims]
