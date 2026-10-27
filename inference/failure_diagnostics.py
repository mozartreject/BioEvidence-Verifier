"""Request-local diagnostics, never an accepted clinical verdict."""
from contextvars import ContextVar
failure_context=ContextVar('biomedical_failure_context',default=None)

def build_diagnostics(claim,result,error):
    evidence=[]
    for p in result.get('evidence',[]):
        raw=p.get('raw',[])
        evidence.append(dict(evidence_id=p['document_id'],title=p.get('title',''),text=p.get('selected',{}).get('text',''),
            source=p.get('source'),source_url=('https://pubmed.ncbi.nlm.nih.gov/'+p['document_id'].split(':',1)[1]+'/' if p['document_id'].startswith('pubmed:') and p['document_id'].split(':',1)[1].isdigit() else None),
            probabilities=dict(zip(['refuted','nei','supported'],raw)),
            pairwise_label=['REFUTED','NEI','SUPPORTED'][max(range(3),key=lambda i:raw[i])] if len(raw)==3 else None,
            relevance=p.get('relevance')))
    review=getattr(error,'review',None)
    neutral=bool(evidence) and all(e['pairwise_label']=='NEI' for e in evidence)
    return dict(claim=claim,accepted_verdict=None,classification_status='REJECTED',
        proposed_verdict=result.get('verdict'),proposed_scores=result.get('scores'),
        failure_stage='Evidence-relation review' if review or 'relation review' in str(error) else 'Evidence consistency check',
        explanation=('All selected passages were scored neutral. The aggregate prediction was rejected because these passages do not justify a decisive conclusion.' if neutral else str(error)),
        scientific_meaning='This is a failure of this pipeline to justify an answer, not evidence of scientific uncertainty about the claim.',
        retrieval=dict(corpus_documents=result.get('corpus_documents'),live_search=any(p.get('retrieval_origin')=='live_pubmed' for p in result.get('evidence',[])),selected_documents=len(evidence)),
        evidence=evidence,relation_review=review,
        limitations=['Passage scores and relevance are uncalibrated model outputs.','Retrieved papers may be incomplete or irrelevant.','A rejected proposed verdict is not a verified answer.'])
