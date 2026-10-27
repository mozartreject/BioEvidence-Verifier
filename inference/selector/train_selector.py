#!/usr/bin/env python3
"""Train a sentence-evidence selector; never read development/test labels.

Labels mean annotated evidence relevance, NOT supported/refuted claim verdicts.
"""
import argparse,contextlib,hashlib,json,math,os,random,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL='cross-encoder/ms-marco-MiniLM-L-6-v2'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def tokens(text):return set(re.findall(r'[a-z0-9]+',text.lower()))

def build_data(data,seed=42):
    corpus={str(d['doc_id']):d for d in map(json.loads,(data/'corpus.jsonl').read_text().splitlines())}
    claims=list(map(json.loads,(data/'claims_train.jsonl').read_text().splitlines()))
    # Connected document groups keep all claims about one annotated document,
    # including negated variants and multi-document claims, in the same split.
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    for c in claims:
        ids=list(c.get('evidence',{}))
        for d in ids[1:]:parents[find(d)]=find(ids[0])
        for d in ids:find(d)
    groups=sorted({find(d) for d in parents});random.Random(seed).shuffle(groups)
    valid=set(groups[:max(1,round(.2*len(groups)))])
    result={'train':[],'validation':[]}
    for c in claims:
        query=tokens(c['claim'])
        for did,annotations in c.get('evidence',{}).items():
            if did not in corpus:raise ValueError('Missing corpus document '+did)
            doc=corpus[did];positives={i for a in annotations for i in a['sentences']}
            negatives=[i for i in range(len(doc['abstract'])) if i not in positives]
            negatives.sort(key=lambda i:(-len(query & tokens(doc['abstract'][i])),i))
            split='validation' if find(did) in valid else 'train'
            # Validation includes all sentences; train emphasizes hard, same-paper negatives.
            chosen=sorted(positives|set(negatives if split=='validation' else negatives[:max(3,3*len(positives))]))
            for i in chosen:
                if doc['abstract'][i].strip():result[split].append(dict(claim_id=str(c['id']),document_id=did,sentence_index=i,
                    claim=c['claim'],text=doc['title']+'\n'+doc['abstract'][i],label=int(i in positives)))
    a={r['document_id'] for r in result['train']};b={r['document_id'] for r in result['validation']}
    assert not a&b,'Document leakage between training and validation'
    assert not ({r['claim_id'] for r in result['train']} & {r['claim_id'] for r in result['validation']}),'Claim leakage'
    return result

def ranking_metrics(rows,scores):
    groups={}
    for r,s in zip(rows,scores):groups.setdefault((r['claim_id'],r['document_id']),[]).append((float(s),r['label']))
    hits=[];aps=[]
    for pairs in groups.values():
        pairs.sort(key=lambda item:item[0],reverse=True);positives=sum(y for _,y in pairs)
        if not positives:continue
        hits.append(pairs[0][1]);found=0;ap=0.
        for rank,(_,label) in enumerate(pairs,1):
            if label:found+=1;ap+=found/rank
        aps.append(ap/positives)
    return dict(groups=len(aps),top1_evidence_hit_rate=sum(hits)/len(hits) if hits else None,
                mean_average_precision=sum(aps)/len(aps) if aps else None)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path.home()/'Downloads/phase2/data/scifact')
    ap.add_argument('--output',type=Path,default=ROOT/'trained-v2')
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--smoke',action='store_true',help='Tiny execution check; produces no deployment model')
    args=ap.parse_args()
    if args.epochs<1 or args.batch_size<1:ap.error('epochs and batch-size must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    data=build_data(args.data)
    if args.smoke:
        data={k:v[:32] for k,v in data.items()}
    configuration=dict(script=digest(Path(__file__)),training_labels=digest(args.data/'claims_train.jsonl'),corpus=digest(args.data/'corpus.jsonl'),
                       model=MODEL,epochs=args.epochs,batch_size=args.batch_size,device=args.device,smoke=args.smoke,seed=42,learning_rate=2e-5,max_length=384)
    fingerprint=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fingerprint:raise SystemExit('Configuration changed. Use a NEW output directory.')
    write(manifest,dict(fingerprint=fingerprint,configuration=configuration))
    write(out/'data.json',data)
    stats={k:dict(sentences=len(v),positive=sum(r['label'] for r in v),documents=len({r['document_id'] for r in v}),claims=len({r['claim_id'] for r in v})) for k,v in data.items()}
    write(out/'data-summary.json',stats);print(json.dumps(stats,indent=2),flush=True)
    if args.prepare_only:return
    import fcntl
    lock=(out/'train.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This training directory is already in use.')
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    random.seed(42);torch.manual_seed(42)
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Device:',device,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,local_files_only=True).to(device)
    if model.config.num_labels!=1:raise ValueError('Expected a single relevance logit')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    start=0;best=-1.;history=[]
    checkpoint=out/'last.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):values[key]=value.to(device)
        start=state['epoch'];best=state['best'];history=state['history']
    def encode(rows):return tokenizer([r['claim'] for r in rows],[r['text'] for r in rows],padding=True,truncation=True,max_length=384,return_tensors='pt').to(device)
    def validate():
        model.eval();scores=[]
        with torch.inference_mode():
            for i in range(0,len(data['validation']),args.batch_size):scores.extend(model(**encode(data['validation'][i:i+args.batch_size])).logits.reshape(-1).float().cpu().tolist())
        return ranking_metrics(data['validation'],scores)
    if not history:
        baseline=validate();history.append(dict(epoch=0,validation=baseline));best=baseline['mean_average_precision'];print('Before training:',baseline,flush=True)
        if not args.smoke:model.save_pretrained(out/'best');tokenizer.save_pretrained(out/'best')
        write(out/'history.json',history)
    for epoch in range(start,args.epochs):
        torch.manual_seed(42+epoch);indices=list(range(len(data['train'])));random.Random(42+epoch).shuffle(indices)
        model.train();total=0.;steps=0;began=time.perf_counter()
        for offset in range(0,len(indices),args.batch_size):
            batch=[data['train'][i] for i in indices[offset:offset+args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            logits=model(**encode(batch)).logits.reshape(-1)
            targets=torch.tensor([r['label'] for r in batch],dtype=logits.dtype,device=device)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; stopping without promotion')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach().cpu());steps+=1
            if steps%50==0:print(f'Epoch {epoch+1}: {min(offset+args.batch_size,len(indices))}/{len(indices)} sentences; loss={total/steps:.4f}',flush=True)
        val=validate();improved=val['mean_average_precision']>best
        if improved:
            best=val['mean_average_precision']
            if not args.smoke:model.save_pretrained(out/'best');tokenizer.save_pretrained(out/'best')
        row=dict(epoch=epoch+1,loss=total/steps,validation=val,seconds=time.perf_counter()-began,improved_over_previous_best=improved)
        history.append(row);write(out/'history.json',history)
        temporary=out/'last.tmp';torch.save(dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,history=history),temporary);temporary.replace(checkpoint)
        print(json.dumps(row),flush=True)
    write(out/'result.json',dict(smoke_only=args.smoke,best_validation_map=best,history=history,
        status='diagnostic_only_not_deployed',next_step='Evaluate selected model on the same cached 300 development claims, including NEI, before considering integration.'))
    print('Complete. Results:',out/'result.json',flush=True)
if __name__=='__main__':main()
