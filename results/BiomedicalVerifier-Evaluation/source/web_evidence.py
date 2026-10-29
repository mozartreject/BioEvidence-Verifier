"""Public web search and bounded full-page retrieval from medical authorities."""
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from xml.etree import ElementTree as ET
import requests

DOMAINS = ('who.int','cdc.gov','nhs.uk','nih.gov','medlineplus.gov','nice.org.uk',
           'cochrane.org','mayoclinic.org','clevelandclinic.org','gov.uk','ecdc.europa.eu',
           'cancer.gov','cancer.org','heart.org','diabetes.org')

def safe_url(url):
    try:
        p = urlsplit(url)
        host = (p.hostname or '').lower()
        if p.scheme != 'https' or p.username or p.password or p.port not in (None,443): return None
        if not any(host == d or host.endswith('.'+d) for d in DOMAINS): return None
        return urlunsplit(('https', p.netloc.lower(), p.path or '/', p.query, ''))
    except ValueError: return None

class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[];self.all=[];self.main=[];self.title=[]
    def handle_starttag(self,tag,attrs):
        if tag not in ('area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'):
            self.stack.append(tag)
        if tag in ('p','div','li','h1','h2','h3','section','br'): self.add('\n')
    def handle_endtag(self,tag):
        if tag in self.stack:
            i=len(self.stack)-1-self.stack[::-1].index(tag);self.stack=self.stack[:i]
        if tag in ('p','div','li','h1','h2','h3','section'): self.add('\n')
    def add(self,s):
        if any(t in self.stack for t in ('script','style','nav','footer','header','noscript','svg','form')): return
        self.all.append(s)
        if 'main' in self.stack or 'article' in self.stack: self.main.append(s)
    def handle_data(self,s):
        if 'title' in self.stack: self.title.append(s)
        self.add(s)
    def text(self):
        raw=' '.join(self.main if len(''.join(self.main))>300 else self.all)
        return '\n'.join(line for line in (re.sub(r'\s+',' ',p).strip() for p in raw.split('\n')) if line)

class WebSearch:
    def __init__(self, cache_dir, session=None):
        self.cache=Path(cache_dir);self.session=session or requests
    def fetch(self,url,title):
        current=safe_url(url)
        if not current: return None
        for _ in range(4):
            r=self.session.get(current,timeout=(4,10),allow_redirects=False,stream=True,
                headers={'User-Agent':'BiomedicalVerifierResearch/1.0'})
            try:
                if r.status_code in (301,302,303,307,308):
                    from urllib.parse import urljoin
                    current=safe_url(urljoin(current,r.headers.get('Location','')))
                    if not current:return None
                    continue
                r.raise_for_status()
                if 'text/html' not in r.headers.get('Content-Type','').lower():return None
                chunks=[];size=0
                for chunk in r.iter_content(32768):
                    size+=len(chunk)
                    if size>2_000_000:return None
                    chunks.append(chunk)
                page=PageText();page.feed(b''.join(chunks).decode(r.encoding or 'utf-8',errors='replace'))
                body=page.text()
                if len(body)<350:return None
                return dict(id='web:'+hashlib.sha256(current.encode()).hexdigest()[:20],
                    title=' '.join(page.title).strip() or title,text=body[:18000],source=urlsplit(current).hostname,
                    source_url=current,retrieval_origin='live_web',retrieved_at_utc=datetime.now(timezone.utc).isoformat())
            finally:r.close()
        return None
    def search(self,claim):
        key=hashlib.sha256(claim.encode()).hexdigest();path=self.cache/(key+'.json')
        try:
            cached=json.loads(path.read_text())
            if time.time()-cached['saved_at']<86400:
                return cached['records'],dict(provider='Bing RSS',cache_hit=True,query=claim,errors=[])
        except (OSError,ValueError,KeyError,TypeError):pass
        r=self.session.get('https://www.bing.com/search',params={'q':claim+' WHO CDC NHS medical evidence','format':'rss'},timeout=(4,15))
        r.raise_for_status()
        root=ET.fromstring(r.content)
        candidates=[];seen=set()
        for item in root.findall('./channel/item'):
            url=safe_url(item.findtext('link',''))
            if url and url not in seen:
                seen.add(url);candidates.append((url,item.findtext('title','')))
        def read(item):
            try:return self.fetch(*item)
            except (requests.RequestException,ValueError):return None
        with ThreadPoolExecutor(max_workers=4) as pool:
            records=[d for d in pool.map(read,candidates[:8]) if d]
        if records:
            try:
                self.cache.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(saved_at=time.time(),records=records)))
            except OSError:pass
        return records,dict(provider='Bing RSS',cache_hit=False,query=claim,eligible_pages=len(candidates),fetched_pages=len(records),errors=[])
