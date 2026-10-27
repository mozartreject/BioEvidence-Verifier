"""Bounded PubMed search with explicit provenance and an offline-safe cache.

Search terms are transmitted to NCBI only when this client is enabled by the
adapter. Never accepts arbitrary URLs. Never assigns a fact-checking verdict.
"""
from __future__ import annotations
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
import requests

_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

def search_terms(claim):
    # Remove generic assertion words for SEARCH only; the original claim is
    # unchanged for reranking and verification, including its negation.
    words = re.findall(r"[\w]+(?:[-/][\w]+)*", claim.lower())
    stop = set('a an the is are was were be been being do does did not no all any can cannot will would should may might to of for in on by with and or that this it its cause causes cure cures prevent prevents effective ineffective effectiveness treatment treats treat'.split())
    terms = [w for w in words if w not in stop]
    # Disambiguate the common disease name from temperature treatments.
    terms = ['common cold' if w == 'colds' else w for w in terms]
    return ' '.join(dict.fromkeys(terms))[:400]

def parse_articles(xml):
    root = ET.fromstring(xml)
    records = []
    stamp = datetime.now(timezone.utc).isoformat()
    for article in root.findall('.//PubmedArticle'):
        citation = article.find('MedlineCitation')
        if citation is None: continue
        pmid = citation.findtext('PMID', '').strip()
        title = citation.find('Article/ArticleTitle')
        paragraphs = citation.findall('Article/Abstract/AbstractText')
        if not pmid.isdigit() or title is None or not paragraphs: continue
        publication_types = [n.text or '' for n in citation.findall('Article/PublicationTypeList/PublicationType')]
        if any(t in {'Retracted Publication', 'Retraction of Publication'} for t in publication_types): continue
        title_text = ''.join(title.itertext()).strip()
        abstract = '\n'.join((p.get('Label','') + ': ' if p.get('Label') else '') + ''.join(p.itertext()).strip() for p in paragraphs).strip()
        if not title_text or not abstract: continue
        date = citation.find('Article/Journal/JournalIssue/PubDate')
        date_text = ' '.join(date.itertext()).strip() if date is not None else None
        records.append({'id':f'pubmed:{pmid}', 'title':title_text, 'text':abstract, 'source':'pubmed', 'score':None,
                        'source_url':f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', 'publication_date':date_text,
                        'retrieved_at_utc':stamp, 'retrieval_origin':'live_pubmed', 'publication_types':publication_types})
    return records

class PubMedSearch:
    def __init__(self, cache_dir, limit=12, timeout=15, ttl=86400, session=None):
        self.cache_dir = Path(cache_dir)
        self.limit = max(1,min(20,int(limit)))
        self.timeout = timeout
        self.ttl = ttl
        self.session = session or requests.Session()

    def _request(self, endpoint, params):
        global _LAST_REQUEST
        # Keep aggregate requests from this process below the unkeyed 3/s limit.
        with _LOCK:
            delay = .4 - (time.monotonic() - _LAST_REQUEST)
            if delay > 0: time.sleep(delay)
            _LAST_REQUEST = time.monotonic()
            response = self.session.get(BASE+endpoint, params={'tool':'BiomedicalVerifier', **params}, timeout=self.timeout)
            response.raise_for_status()
            return response

    def search(self, claim):
        query = search_terms(claim)
        if not query: return [], {'query':query, 'cache_hit':False, 'reason':'No usable search terms'}
        key = hashlib.sha256(f'v1:{self.limit}:{query}'.encode()).hexdigest()
        path = self.cache_dir / f'{key}.json'
        try:
            cached = json.loads(path.read_text())
            if time.time()-cached['saved_at'] < self.ttl and cached['query']==query:
                return cached['records'], {'query':query, 'cache_hit':True, 'reason':'cached PubMed response'}
        except (OSError, ValueError, KeyError, TypeError): pass
        response = self._request('esearch.fcgi', {'db':'pubmed','term':query,'retmode':'json','retmax':self.limit,'sort':'relevance'})
        payload = response.json()
        if payload.get('error') or 'esearchresult' not in payload: raise ValueError('PubMed search returned an error')
        ids = [str(i) for i in payload['esearchresult'].get('idlist',[]) if str(i).isdigit()][:self.limit]
        records = []
        if ids:
            response = self._request('efetch.fcgi', {'db':'pubmed','id':','.join(ids),'retmode':'xml','rettype':'abstract'})
            records = parse_articles(response.content)
            records = [r for r in records if r['id'].split(':')[-1] in ids]
        reason = 'live PubMed search'
        try:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
            cache = {'saved_at':time.time(),'query':query,'records':records}
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(cache,ensure_ascii=False)); temporary.replace(path)
        except OSError:
            reason += '; cache unavailable'
        return records, {'query':query,'cache_hit':False,'reason':reason}
