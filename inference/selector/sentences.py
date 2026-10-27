"""Inference accepts claim/document text only; never benchmark annotations."""
import math,re

def sentence_units(text,tokenizer,budget):
    boundaries=[0]+[m.end() for m in re.finditer(r'\n+|(?<=[.!?])\s+(?=[A-Z])',text)]+[len(text)]
    units=[]
    for a,b in zip(boundaries,boundaries[1:]):
        offsets=tokenizer(text[a:b],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        for i in range(0,len(offsets),budget):
            part=offsets[i:i+budget]
            if part:units.append(dict(start=a+part[0][0],end=a+part[-1][1]))
    return units

def choose(claim,document,tokenizer,ranker):
    text=document['text'];title=document.get('title','')
    budget=min(320,512-len(tokenizer.encode(claim,add_special_tokens=False))-len(tokenizer.encode(title,add_special_tokens=False))-16)
    if budget<32:raise ValueError('Insufficient token budget for evidence context')
    units=sentence_units(text,tokenizer,budget)
    if not units:raise ValueError('Document has no sentence candidates')
    scores=[float(s) for s in ranker.predict([(claim,title+'\n'+text[u['start']:u['end']]) for u in units],batch_size=16,show_progress_bar=False)]
    if not all(math.isfinite(s) for s in scores):raise ValueError('Non-finite selector score')
    best=max(range(len(units)),key=lambda i:scores[i]);u=units[best]
    # Choose the SAME central sentence for both conditions; context is the only change.
    left=right=best
    for neighbor in [best-1,best+1]:
        if not 0<=neighbor<len(units):continue
        a,b=min(left,neighbor),max(right,neighbor)
        if len(tokenizer.encode(text[units[a]['start']:units[b]['end']],add_special_tokens=False))<=budget:left,right=a,b
    a,b=units[left]['start'],units[right]['end']
    return dict(sentence=dict(start=u['start'],end=u['end'],text=text[u['start']:u['end']]),
                context=dict(start=a,end=b,text=text[a:b]),score=scores[best],candidate_count=len(units),token_budget=budget)
