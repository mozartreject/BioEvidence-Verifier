#!/usr/bin/env python3
"""Standalone experimental full-corpus classifier; no server/deployment changes."""
import argparse,json,os,pickle,copy
from functools import lru_cache
from pathlib import Path
import numpy as np
from core import SparseBM25,LABELS
from passage_features import extract
from nli_contract import canonical_order
from live_query import compact_terms
ROOT=Path(__file__).resolve().parent

class Candidate:
    def __init__(self, live_pubmed=False, trained_selector=False, verifier_path=None, combiner_dir=None):
        self.live_pubmed=live_pubmed
        self.trained_selector=trained_selector
        self.artifact_dir=Path(combiner_dir).resolve() if combiner_dir else ROOT/('combiner' if trained_selector else 'combiner')
        self._cached_predict=lru_cache(maxsize=16)(self._predict)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch,joblib,importlib.util
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        from sentence_transformers import CrossEncoder
        torch.set_num_threads(4)
        self.torch=torch
        metadata=Path(os.environ.get('BIOVERIFY_CORPUS_METADATA', str(Path.home()/'Downloads/phase2/data/faiss_index_35k/abstract_metadata.pkl'))).expanduser()
        self.docs=pickle.load(metadata.open('rb'))['abstracts']
        assert len(self.docs)==35183
        self.retriever=SparseBM25([d['text'] for d in self.docs])
        self.ranker=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',device='cpu',max_length=384)
        self.selector=CrossEncoder(str(Path(os.environ.get('BIOVERIFY_SELECTOR_MODEL', str(ROOT.parent/'external-models/selector'))).expanduser()),device='cpu',max_length=384) if trained_selector else self.ranker
        name=str(Path(verifier_path).resolve()) if verifier_path else 'pritamdeka/PubMedBERT-MNLI-MedNLI'
        self.tokenizer=AutoTokenizer.from_pretrained(name,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(name,local_files_only=True).eval()
        self.order=canonical_order(self.model.config.id2label,self.model.config.num_labels)
        spec=importlib.util.spec_from_file_location('sentences',ROOT/'selector/sentences.py')
        sent=importlib.util.module_from_spec(spec);spec.loader.exec_module(sent);self.choose=sent.choose
        selection=json.loads((self.artifact_dir/'selection.json').read_text())
        if selection['selected']!='boosted_trees':raise ValueError('Candidate expects the frozen boosted-tree artifact')
        self.combiner=joblib.load(self.artifact_dir/'experimental-model.joblib')
        if self.combiner.classes_.tolist()!=[0,1,2]:raise ValueError('Invalid label mapping')

    def predict(self,claim):
        if not isinstance(claim,str) or not claim.strip():raise ValueError('A nonempty claim is required')
        # Cache only frozen local inference. Live sources must retain their own TTL.
        result=self._predict(claim) if self.live_pubmed else self._cached_predict(claim)
        return copy.deepcopy(result)

    def predict_live(self,claim):
        """Run a request-local PubMed retrieval without duplicating model weights."""
        return copy.deepcopy(self._predict(claim,live_pubmed=True))

    def clear_cache(self):
        """Explicitly invalidate this instance; process restarts already create a fresh cache."""
        self._cached_predict.cache_clear()

    def _predict(self,claim,live_pubmed=None):
        if not isinstance(claim,str) or not claim.strip():raise ValueError('A nonempty claim is required')
        if len(self.tokenizer.encode(claim,add_special_tokens=False))>240:raise ValueError('Split the claim into shorter statements (under 240 model tokens)')
        found=self.retriever.search(claim,30)
        docs=self.docs
        trace={}
        if self.live_pubmed if live_pubmed is None else live_pubmed:
            import sys
            sys.path.insert(0,str(Path(os.environ.get('BIOVERIFY_PROJECT_ROOT', str(ROOT.parent))).expanduser()))
            from backend.services.pubmed_search import PubMedSearch,search_terms
            client=PubMedSearch(ROOT/'live-experiment/cache',limit=20)
            records,primary_trace=client.search(claim)
            trace={'live_queries':[primary_trace]}
            compact=compact_terms(search_terms(claim))
            if compact:
                extra,secondary_trace=client.search(compact)
                trace['live_queries'].append(secondary_trace)
                records.extend(extra)
            # Request-local candidate pool: never mutate the frozen corpus/index.
            docs=list(self.docs)
            existing={self.docs[j]['id'] for j,_ in found}
            for record in records:
                if record['id'] not in existing:
                    found.append((len(docs),0.0));docs.append(record);existing.add(record['id'])
        if len(found)<3:return dict(verdict='NEI',reason='Insufficient retrieved candidates',experimental=True,evidence=[])
        scores=np.asarray(self.ranker.predict([(claim,docs[j]['text']) for j,_ in found],batch_size=16,show_progress_bar=False)).reshape(-1)
        if not np.isfinite(scores).all():raise ValueError('Invalid reranking outputs')
        chosen=np.argsort(-scores,kind='stable')[:3];pairs=[];premises=[]
        ranked=np.argsort(-scores,kind='stable')
        trace['candidates']=[dict(document_id=docs[found[k][0]]['id'],
            bm25_rank=int(k)+1 if k<30 else None,rerank_rank=rank+1,
            reranker_score=float(scores[k]),selected=bool(k in chosen),
            reranker_input_tokens=len(self.ranker.tokenizer.encode(claim,docs[found[k][0]]['text'],truncation=False)),
            reranker_max_length=self.ranker.max_length)
            for rank,k in enumerate(ranked)]
        trace['nli_label_order']=dict(model_id2label=self.model.config.id2label,canonical_indices=self.order)
        trace['truncated_reranker_candidates']=sum(c['reranker_input_tokens']>c['reranker_max_length'] for c in trace['candidates'])
        for k in chosen:
            j,bm25=found[k];d=docs[j];title=d.get('title','');text=d['text']
            if title and text.startswith(title):text=text[len(title):].lstrip()
            selected=self.choose(claim,dict(text=text,title=title),self.tokenizer,self.selector)
            premises.append(title+'\n'+selected['sentence']['text'])
            pairs.append(dict(document_id=d['id'],source=d.get('source'),retrieval_origin=d.get('retrieval_origin','local_corpus'),title=title,selected=selected['sentence'],
                              relevance=float(1/(1+np.exp(-np.clip(scores[k],-50,50)))),reranker_score=float(scores[k]),bm25_score=bm25,
                              selector_score=selected['score'],selector_candidate_count=selected['candidate_count']))
        batch=self.tokenizer(premises,[claim]*3,truncation='only_first',max_length=512,padding=True,return_tensors='pt')
        trace['nli_input_tokens']=[len(self.tokenizer.encode(p,claim,truncation=False)) for p in premises]
        trace['nli_truncated_inputs']=sum(n>512 for n in trace['nli_input_tokens'])
        with self.torch.inference_mode():
            logits=self.model(**batch).logits[:,self.order]
            if not self.torch.isfinite(logits).all():raise ValueError('Nonfinite NLI logits')
            raw=logits.softmax(-1).cpu().numpy()
        trace['canonical_nli_logits']=logits.cpu().tolist()
        for p,probs in zip(pairs,raw):p['raw']=probs.tolist()
        passage_scores=self.ranker.predict([(claim,p) for p in premises],batch_size=16,show_progress_bar=False)
        record=dict(claim=claim,pairs=pairs);features=extract(record,passage_scores)
        probabilities=self.combiner.predict_proba(features[None,:])[0]
        if not np.isfinite(probabilities).all():raise ValueError('Invalid classifier outputs')
        return dict(claim=claim,verdict=LABELS[int(probabilities.argmax())],scores=dict(zip(LABELS,probabilities.tolist())),
            evidence=pairs,experimental=True,calibrated=False,corpus_documents=len(self.docs),
            retrieval_trace=trace,features=features.tolist(),scope='Experimental text classification; used by the opt-in dashboard wrapper; not an explanation/image evaluation.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trained-selector',action='store_true');ap.add_argument('--live-pubmed',action='store_true');ap.add_argument('--verifier-path',type=Path);ap.add_argument('--combiner-dir',type=Path);ap.add_argument('--claim');ap.add_argument('--parity-check',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    if not args.claim and not args.parity_check:ap.error('Provide --claim or --parity-check')
    if args.parity_check and (args.verifier_path or args.combiner_dir or args.live_pubmed):ap.error('Use the experiment-specific parity script for model overrides')
    model=Candidate(live_pubmed=args.live_pubmed,trained_selector=args.trained_selector,verifier_path=args.verifier_path,combiner_dir=args.combiner_dir)
    if args.parity_check:
        saved=json.loads((model.artifact_dir/'dev-features.json').read_text());pred=json.loads((model.artifact_dir/'predictions.json').read_text());checks=[]
        for i in [0,len(saved['ids'])//2,len(saved['ids'])-1]:
            cid=saved['ids'][i];record=json.loads((ROOT/('clean-selector-run' if args.trained_selector else 'full-run')/'cases/dev'/f'{cid}.json').read_text());r=model.predict(record['claim'])
            assert [p['document_id'] for p in r['evidence']]==[p['document_id'] for p in record['pairs']]
            error=float(np.max(np.abs(np.asarray(r['features'])-saved['features'][i])))
            assert error<1e-4,('Feature mismatch',cid,error)
            assert r['verdict']==LABELS[pred[i]],'Verdict mismatch'
            checks.append(dict(id=cid,verdict=r['verdict'],max_feature_difference=error))
        result=dict(status='PASS',checks=checks,scope='Three deterministic live/cache consistency cases; not an accuracy evaluation')
    else:result=model.predict(args.claim)
    if args.output:args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
