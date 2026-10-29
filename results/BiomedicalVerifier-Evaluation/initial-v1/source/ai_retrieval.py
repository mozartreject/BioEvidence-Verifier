"""Retrieve real PubMed abstracts for the independent AI decision path."""
from pathlib import Path
import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from web_evidence import WebSearch
from backend.services.pubmed_search import PubMedSearch, search_terms
from live_query import compact_terms

def retrieve_evidence(claim, engine):
    client = PubMedSearch(Path(__file__).resolve().parent / 'live-experiment/cache', limit=20)
    queries = [claim]
    compact = compact_terms(search_terms(claim))
    if compact and compact != search_terms(claim):
        queries.append(compact)
    records, traces = {}, []
    web_enabled = os.environ.get('BIOVERIFY_WEB_SEARCH') == '1'
    web_trace = {'enabled':web_enabled}
    def web_search():
        try:
            return WebSearch(Path(__file__).resolve().parent/'web-cache').search(claim)
        except Exception as exc:
            return [], {'enabled':True,'error':type(exc).__name__,'warning':'Web search unavailable; only successfully retrieved sources can be used.'}
    pool=ThreadPoolExecutor(max_workers=1)
    web_future=pool.submit(web_search) if web_enabled else None
    for query in queries:
        try:
            found, trace = client.search(query)
        except Exception as exc:
            found, trace = [], {'query':query,'error':type(exc).__name__,'warning':'PubMed search unavailable.'}
        traces.append(trace)
        for record in found:
            records[record['id']] = record
    if web_future:
        web_records, web_trace = web_future.result()
        for record in web_records: records[record['id']]=record
    pool.shutdown(wait=True)
    documents = list(records.values())
    # Retain complete relevant paragraphs and their neighbors within the AI context budget.
    terms=set(re.findall(r'\w+',search_terms(claim).lower()))
    for d in documents:
        paragraphs=d['text'].split('\n')
        if len(d['text'])>7000:
            ranked=sorted(range(len(paragraphs)), key=lambda i:len(terms & set(re.findall(r'\w+',paragraphs[i].lower()))), reverse=True)
            chosen=set();length=0
            for i in ranked:
                for j in (i,max(0,i-1),min(len(paragraphs)-1,i+1)):
                    if j not in chosen and length+len(paragraphs[j])<=7000:
                        chosen.add(j);length+=len(paragraphs[j])+1
            d={**d,'text':'\n'.join(paragraphs[j] for j in sorted(chosen)) or d['text'][:7000]}
            records[d['id']]=d
    documents=list(records.values())
    if not documents:
        raise ValueError('No readable evidence was retrieved from the enabled search sources. Try a more specific claim or retry later.')
    ranks = np.asarray(engine.ranker.predict([(claim, d['title'] + '\n' + d['text']) for d in documents], batch_size=16, show_progress_bar=False)).reshape(-1)
    if not np.isfinite(ranks).all():
        raise ValueError('Invalid relevance ranking')
    evidence = []
    order=list(np.argsort(-ranks,kind='stable'))
    selected=[]
    # Include the best fetched web page and PubMed paper when both are available.
    if web_enabled:
        for origin in ('live_web','live_pubmed'):
            match=next((k for k in order if documents[k].get('retrieval_origin')==origin),None)
            if match is not None:selected.append(match)
    for k in order:
        if len(selected)>=3:break
        if k not in selected:selected.append(k)
    selected.sort(key=lambda k:-ranks[k])
    for k in selected:
        d = documents[k]
        evidence.append(dict(document_id=d['id'], title=d['title'], source=d['source'], source_url=d.get('source_url'), retrieved_at_utc=d.get('retrieved_at_utc'),
            retrieval_origin=d.get('retrieval_origin','live_pubmed'), review_text=d['text'], selected={'text': d['text']},
            bm25_score=0.0, reranker_score=float(ranks[k]), relevance=float(1 / (1 + np.exp(-np.clip(ranks[k], -50, 50))))))
    return dict(evidence=evidence, scores={}, retrieval_trace={'live_queries': traces,'web_search':web_trace})
