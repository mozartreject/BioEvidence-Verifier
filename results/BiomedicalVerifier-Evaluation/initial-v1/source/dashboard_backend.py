"""Opt-in dashboard integration; original project files are never edited."""
import hashlib,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
PROJECT=Path.home()/'Downloads/BiomedicalVerifier'
sys.path.insert(0,str(PROJECT))
from backend.schemas import (EvidenceRecord,Phase2Result,Phase4Result,PhaseTiming,ModelArtifact,
    ClassProbabilities,PairwiseVerification,Verdict,verdict_to_user_facing)
from candidate import Candidate
from ai_retrieval import retrieve_evidence
from evidence_guard import validate_decisive_evidence
from relation_review import require_agreement
from failure_diagnostics import failure_context,build_diagnostics
from retrieval_rescue import select_result
from claim_input import retrieval_claim
from api_reviewer import configured as api_reviewer_configured,require_api_agreement,review_claim as api_review_claim

class Shared:
    def __init__(self):self.engine=None;self.result=None;self.claim=None;self.review=None;self.rescue_trace=None
    def load(self):
        if self.engine is None:self.engine=Candidate(trained_selector=True)

class ImageStatusCapture:
    """Keep Phase 1 status available when a later evidence check rejects a verdict."""
    def __init__(self,adapter):self.adapter=adapter
    def __getattr__(self,name):return getattr(self.adapter,name)
    def infer(self,request):
        result=self.adapter.infer(request)
        context=failure_context.get()
        if context is not None and request.image_path:
            context['image_analysis']=dict(has_image=bool(result.has_image),status=str(result.status.value),
                verdict_effect='none',scope='Experimental image/text diagnostic; not a medical-image diagnosis')
        return result

class Retrieval:
    def __init__(self,shared):self.shared=shared;self.retriever=SimpleNamespace(docs=[])
    def load(self):self.shared.load();self.retriever.docs=self.shared.engine.docs
    def run(self,claim,claim_id):
        self.shared.result=None;self.shared.claim=None;self.shared.review=None;self.shared.rescue_trace=None
        started=datetime.now(timezone.utc);clock=time.perf_counter()
        effective_claim=retrieval_claim(claim)
        ai_final=os.environ.get('BIOVERIFY_AI_FINAL')=='1'
        result=(retrieve_evidence(effective_claim,self.shared.engine) if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else self.shared.engine.predict_live(effective_claim)) if ai_final else self.shared.engine.predict(effective_claim)
        if not ai_final and os.environ.get('BIOVERIFY_RESCUE_LIVE')=='1':
            result,self.shared.review,self.shared.rescue_trace=select_result(effective_claim,result,self.shared.engine.predict_live)
        if not result.get('evidence'):raise ValueError('Not enough retrieval candidates to run the experimental pipeline')
        self.shared.result=result;self.shared.claim=claim
        docs=[]
        for p in result['evidence']:
            url='https://pubmed.ncbi.nlm.nih.gov/'+p['document_id'].split(':',1)[1]+'/' if p['source']=='pubmed' else p.get('source_url')
            docs.append(EvidenceRecord(evidence_id=p['document_id'],document_id=p['document_id'],title=p['title'],
                text=p.get('review_text',p['selected']['text']) if ai_final else p['selected']['text'],source=p['source'],source_url=url,
                retrieval_origin=p.get('retrieval_origin','full_35183_bm25_passage_combiner'),bm25_score=p['bm25_score'],
                crossencoder_raw_score=p['reranker_score'],relevance_score=p['relevance']))
        return Phase2Result(status='SUCCESS',evidence=docs,evidence_found=True,queries_executed=[claim],
            retriever_name=('PubMed + medical web search' if os.environ.get('BIOVERIFY_WEB_SEARCH')=='1' else 'PubMed') if ai_final else 'BM25 over 35183 documents',reranker_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
            stop_reason='Up to three fetched research sources ranked for AI review.' if ai_final else 'Three local corpus passages selected.',
            warnings=([t['warning'] for t in result.get('retrieval_trace',{}).get('live_queries',[]) if t.get('warning')]+([result['retrieval_trace']['web_search']['warning']] if result.get('retrieval_trace',{}).get('web_search',{}).get('warning') else []) if ai_final else ['Joint candidate-engine runtime is recorded in Phase 2.'])+(['A visibly clipped opening was repaired for retrieval and model scoring; the original input remains in the request record.'] if effective_claim!=claim else []),
            timing=PhaseTiming(phase='phase2',started_at_utc=started,ended_at_utc=datetime.now(timezone.utc),latency_ms=(time.perf_counter()-clock)*1000))

class Verification:
    def __init__(self,shared):
        self.shared=shared
        artifact=ROOT/'clean-selector-combiner/experimental-model.joblib'
        self.model_artifact=ModelArtifact(component='phase4_passage_combiner',model_name='Supervised evidence selector + PubMedBERT MNLI/MedNLI + passage-aware boosted trees',
            version=hashlib.sha256(artifact.read_bytes()).hexdigest(),checkpoint_path=str(artifact),metadata=dict(
                experimental=True,calibrated=False,development_accuracy=201/300,development_cases=300,
                scope='Retrieval-inclusive reused-development classification score; not a score for images or explanations.',
                hierarchy_policy='Phase 3 provides contextual authority metadata; classifier uses fixed relevance-ranked passage features, not hierarchy weights.'))
    def load(self):pass
    def run(self,claim,evidence_list):
        result=self.shared.result
        if result is None or claim!=self.shared.claim:raise ValueError('Missing or mismatched candidate request')
        byid={p['document_id']:p for p in result['evidence']}
        ids=[e.evidence_id for e in evidence_list]
        if len(ids)!=len(set(ids)) or set(ids)!=set(byid):raise ValueError('Phase 3 changed the selected evidence set; refusing stale classification')
        ai_final=os.environ.get('BIOVERIFY_AI_FINAL')=='1'
        try:
            if ai_final:
                if not api_reviewer_configured():
                    raise ValueError('Evidence validation failed: final AI review is enabled without an API configuration.')
                # The independent model sees the claim and retrieved source text, never
                # the trained classifier's proposed verdict or its class scores.
                api_review=api_review_claim(retrieval_claim(claim),result['evidence'])
                relation_review=None
            else:
                validate_decisive_evidence(result)
                relation_review=self.shared.review if self.shared.review is not None else require_agreement(retrieval_claim(claim),result)
                api_review=require_api_agreement(retrieval_claim(claim),result)
        except Exception as exc:
            error=exc if isinstance(exc,ValueError) else ValueError('Evidence validation failed: live AI review could not be completed. No verified conclusion is available.')
            context=failure_context.get()
            if context is not None:context['diagnostics']=build_diagnostics(claim,result,error)
            raise error from exc
        artifact=self.model_artifact.model_copy(update={
            "component":'live_pubmed_ai_decider' if ai_final else self.model_artifact.component,
            "model_name":(os.environ.get('BIOVERIFY_AI_MODEL','AI API evidence reviewer')+(' over PubMed and medical web evidence' if os.environ.get('BIOVERIFY_WEB_SEARCH')=='1' else ' over live PubMed evidence')) if ai_final else self.model_artifact.model_name,
            "checkpoint_path":None if ai_final else self.model_artifact.checkpoint_path,
            "version":os.environ.get('BIOVERIFY_AI_MODEL') if ai_final else self.model_artifact.version,
            "metadata":{**self.model_artifact.metadata,"relation_review":relation_review,
                "external_api_review":api_review,"decision_source":'live_pubmed_ai_review' if ai_final else 'trained_combiner',
                "base_model_scores":result['scores'],"final_decision_accuracy_validated":False if ai_final else None,
                "development_accuracy":None if ai_final else self.model_artifact.metadata.get('development_accuracy')}},deep=True)
        ai_scores = ai_final and os.environ.get('BIOVERIFY_AI_SCORES') == '1'
        verdict=Verdict(api_review['verdict'] if ai_final else result['verdict'])
        scores=api_review['class_scores'] if ai_scores else result['scores']
        if ai_scores:
            artifact.metadata.pop('base_model_scores', None)
            artifact.metadata['score_source']='ai_evidence_estimates'
            artifact.metadata['score_calibration']='uncalibrated'
        probabilities=ClassProbabilities(refuted=scores['REFUTED'],nei=scores['NEI'],supported=scores['SUPPORTED'])
        pairs=[]
        for did in ([] if ai_scores else ids):
            p=byid[did];raw=p['raw'];label=list(Verdict)[max(range(3),key=lambda i:raw[i])]
            pairs.append(PairwiseVerification(evidence_id=did,probabilities=ClassProbabilities(refuted=raw[0],nei=raw[1],supported=raw[2]),
                predicted_label=label,scored_text=p['title']+'\n'+p['selected']['text']))
        return Phase4Result(status='SUCCESS',verdict=verdict,user_facing_verdict=verdict_to_user_facing(verdict),
            probabilities=probabilities,verifier_score=scores[verdict.value],pairwise_predictions=pairs,evidence_used_ids=ids,
            aggregation_method=('Retrieved PubMed/web evidence followed by quoted-evidence AI review' if ai_final else 'Training-selected passage-aware boosted-tree combiner; authority weights are contextual only'),
            label_mapping=dict(enumerate([Verdict.REFUTED,Verdict.NEI,Verdict.SUPPORTED])),model=artifact,
            warnings=(['Experimental AI decision over live PubMed candidates; accuracy has not been measured on a held-out set.',
                       'Percentages are uncalibrated AI estimates based on retrieved evidence.' if ai_scores else 'Numeric probabilities belong to the trained baseline and do not quantify the AI decision.'] if ai_final else
                      ['Experimental candidate: 67.0% on reused development claims; 75% target not reached. Scores are uncalibrated.',
                       'Pairwise scores are raw NLI outputs. Final scores come from the trained combiner.'])+
                     (['The AI API reviewer found a potentially decisive relation, while the trained classifier selected NEI; the NEI verdict was not automatically overridden.']
                      if not ai_final and api_review and result['verdict']=='NEI' and api_review.get('verdict')!='NEI' else []))

def create_app():
    import importlib
    from backend.api import dependencies
    from backend.pipeline import BiomedicalVerificationPipeline
    from backend.pipeline_config import load_pipeline_config
    module=importlib.import_module('backend.api.app')
    runtime=ROOT/'dashboard-runtime'
    os.environ.update(BIOVERIFY_VERIFIER='biomedical_nli',BIOVERIFY_LIVE_PUBMED='0',BIOVERIFY_PHASE3_TOP_K='3',
        BIOVERIFY_OUTPUTS_DIR=str(runtime/'outputs'),BIOVERIFY_AUDIT_DIR=str(runtime/'audit'),
        BIOVERIFY_HIERARCHY_CACHE=str(runtime/'hierarchy-cache.json'),BIOVERIFY_PIPELINE_VERSION='ai-evidence-review-v1' if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 'passage-combiner-experimental-67.0')
    def initialize():
        if dependencies._pipeline is None:
            shared=Shared();config=load_pipeline_config()
            pipeline=BiomedicalVerificationPipeline(config=config,phase2=Retrieval(shared),phase4=Verification(shared))
            pipeline.phase1=ImageStatusCapture(pipeline.phase1)
            dependencies._pipeline=pipeline
        return dependencies._pipeline
    inference_module=importlib.import_module('backend.api.routes.inference')
    inference_module.UPLOAD_DIR=runtime/'uploads'
    inference_module.UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    module.initialize_pipeline=initialize
    app=module.app
    if not getattr(app.state,'failure_context_installed',False):
        @app.middleware('http')
        async def request_failure_context(request,call_next):
            token=failure_context.set({})
            try:return await call_next(request)
            finally:failure_context.reset(token)
        app.state.failure_context_installed=True
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from fastapi.exception_handlers import http_exception_handler
    from fastapi.responses import JSONResponse
    @app.exception_handler(StarletteHTTPException)
    async def evidence_error_handler(request, exc):
        detail = exc.detail
        message = detail.get('message', '') if isinstance(detail, dict) else ''
        if isinstance(message, str) and 'Evidence validation failed:' in message:
            return JSONResponse(status_code=422, content={'detail': {
                'error': 'EVIDENCE_VALIDATION_FAILED',
                'message': 'The retrieved evidence does not justify the proposed verdict. No verified conclusion is available.',
                'reason': message,
                'retryable': False,
                'diagnostics': {**((failure_context.get() or {}).get('diagnostics') or {}),
                    'image_analysis':(failure_context.get() or {}).get('image_analysis')}}})
        return await http_exception_handler(request, exc)
    # Replace legacy static descriptions with accurate metadata for this mode.
    app.router.routes=[r for r in app.router.routes if getattr(r,'path',None) not in ('/','/api/v1/system/health','/api/v1/system/models')]
    @app.get('/')
    def root():return dict(service='BiomedicalVerifier experimental passage-combiner',api='/api/v1',mode='experimental')
    @app.get('/api/v1/system/health')
    def health():
        p=dependencies.get_pipeline()
        return dict(status='healthy' if p._loaded else 'loading',pipeline_ready=p._loaded,mode='experimental',
            pipeline_version=p.config.runtime.pipeline_version,verifier_backend='ai_evidence_review' if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 'passage_combiner',
            evidence_validation='quoted_evidence_sufficiency_and_ai_scores' if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 'pairwise_direction_and_independent_relation_review',
            relation_review_model='qwen2.5:7b',relation_review_status='final_decision_not_accuracy_validated' if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 'experimental_veto_not_accuracy_validated',
            ai_api_review=dict(enabled=api_reviewer_configured(),model=os.environ.get('BIOVERIFY_AI_MODEL') if api_reviewer_configured() else None,
                policy='final_decision_over_live_pubmed' if os.environ.get('BIOVERIFY_AI_FINAL')=='1' else 'veto_decisive_disagreement',accuracy_validated=False,score_source='ai_evidence_estimates' if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 'trained_combiner'),
            retrieval=dict(document_count=len(p.phase2.retriever.docs),live_pubmed_enabled=os.environ.get('BIOVERIFY_RESCUE_LIVE')=='1',web_search_enabled=os.environ.get('BIOVERIFY_WEB_SEARCH')=='1',web_provider='Bing RSS' if os.environ.get('BIOVERIFY_WEB_SEARCH')=='1' else None,final_candidate_limit=3),
            development_accuracy=None if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else 201/300,target_accuracy=None if os.environ.get('BIOVERIFY_AI_SCORES')=='1' else .75,
            accuracy_scope='Historical unguarded classifier on reused development; current service with relation veto is not accuracy-evaluated',
            current_service_accuracy=None)
    @app.get('/api/v1/system/models')
    def models():
        p=dependencies.get_pipeline()
        return dict(mode='experimental',pipeline_version=p.config.runtime.pipeline_version,
            models=[p.phase4.model_artifact.model_dump(mode='json')],
            scientific_components=dict(phase1='Existing diagnostic image/text branch; does not control verdict',
                phase2='Full 35183-document BM25, MiniLM reranking and supervised passage selection',
                phase3='Existing authority metadata; weights do not alter candidate classification',
                phase4='PubMedBERT plus trained passage-aware combiner',phase5='Existing locked-verdict explanation'))
    return app
