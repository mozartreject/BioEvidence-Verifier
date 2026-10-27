#!/usr/bin/env python3
"""Training-selected combiner with explicit selected-passage relevance features."""
import os,json,hashlib,warnings
from pathlib import Path
import numpy as np
from core import claim_label,linked_groups
from passage_features import extract
from run import metrics
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'clean-selector-combiner'

def main():
    OUT.mkdir(exist_ok=True)
    import fcntl
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(4)
    from sentence_transformers import CrossEncoder
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    warnings.filterwarnings('error',category=RuntimeWarning)
    data=Path.home()/'Downloads/phase2/data/scifact';excluded=set(json.loads((ROOT/'clean-selector-run/leakage-audit.json').read_text())['excluded_ids'])
    # Only training labels are read before model selection is frozen.
    train=[json.loads(s) for s in (data/'claims_train.jsonl').read_text().splitlines()]
    train=[c for c in train if claim_label(c) is not None and str(c['id']) not in excluded]
    ranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',device='cpu',max_length=384)
    def features(split,ids):
        records=[json.loads((ROOT/f'clean-selector-run/cases/{split}'/f'{cid}.json').read_text()) for cid in ids]
        pairs=[(r['claim'],p['title']+'\n'+p['selected']['text']) for r in records for p in r['pairs']]
        scores=np.asarray(ranker.predict(pairs,batch_size=16,show_progress_bar=False)).reshape(-1,3)
        X=np.asarray([extract(r,s) for r,s in zip(records,scores)])
        (OUT/f'{split}-features.json').write_text(json.dumps(dict(ids=list(ids),features=X.tolist(),passage_scores=scores.tolist())))
        return X
    X=features('train',[str(c['id']) for c in train]);y=np.asarray([claim_label(c) for c in train]);groups=np.asarray(linked_groups(train))
    # Fixed baseline hyperparameters, not selected using in-sample selector features.
    factories={'boosted_trees':lambda:GradientBoostingClassifier(n_estimators=120,learning_rate=.05,max_depth=2,min_samples_leaf=8,random_state=42)}
    def infer(model,name,x):
        if name=='linear':
            scaler,clf=model.steps[0][1],model.steps[1][1]
            logits=np.einsum('ij,kj->ik',(x-scaler.mean_)/scaler.scale_,clf.coef_,optimize=False)+clf.intercept_
            assert np.isfinite(logits).all();return clf.classes_[logits.argmax(axis=1)]
        return model.predict(x)
    selected='boosted_trees'
    (OUT/'selection.json').write_text(json.dumps(dict(selected=selected,policy='Fixed baseline combiner hyperparameters. No OOF claim: selector was fitted on part of these training claims. Development labels not used.'),indent=2))
    model=factories[selected]();kwargs=dict(sample_weight=compute_sample_weight('balanced',y)) if selected=='boosted_trees' else {};model.fit(X,y,**kwargs)
    import joblib
    joblib.dump(model,OUT/'experimental-model.joblib')
    dev=[json.loads(s) for s in (data/'claims_dev.jsonl').read_text().splitlines()];dev=[c for c in dev if claim_label(c) is not None]
    pred=infer(model,selected,features('dev',[str(c['id']) for c in dev]));score=metrics([claim_label(c) for c in dev],pred)
    result=dict(selected=selected,development=score,status='experimental_not_deployed',scope='Reused SciFact development labels; all 35183 documents searched upstream; expanded-corpus annotation limitations remain.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2));(OUT/'predictions.json').write_text(json.dumps(pred.tolist()))
    files=[Path(__file__),ROOT/'passage_features.py',ROOT/'core.py',ROOT/'clean-selector-run/leakage-audit.json',data/'claims_train.jsonl',data/'claims_dev.jsonl']+sorted((ROOT/'clean-selector-run/cases').rglob('*.json'))
    (OUT/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
