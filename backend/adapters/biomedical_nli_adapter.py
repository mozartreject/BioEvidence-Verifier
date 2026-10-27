"""Biomedical NLI verifier with explicit label mapping and relevance discounting.

This is a separate deployment mode, not the historical Phase 4 checkpoint.
Scores are uncalibrated model outputs. The original adapter remains available.
"""
from datetime import datetime, timezone
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.pipeline_config import PipelineConfig
from backend.schemas import (ClassProbabilities, ModelArtifact, PairwiseVerification,
    Phase4Result, PhaseStatus, PhaseTiming, Verdict, verdict_to_user_facing)

LABELS = [Verdict.REFUTED, Verdict.NEI, Verdict.SUPPORTED]

def relevance_discount(probabilities, relevance):
    """Unrelated evidence contributes uncertainty, never a confident verdict."""
    r = max(0.0, min(1.0, float(relevance or 0.0)))
    return [probabilities[0]*r, probabilities[1]*r+1-r, probabilities[2]*r]

def evidence_passage(evidence, claim, tokenizer):
    """Preserve conclusions/results in structured abstracts before methods.

    For unstructured long abstracts retain both opening and closing passages.
    Selection is claim-label independent and the actual input is exported.
    """
    parts=evidence.text.split('\n')
    def priority(part):
        label=part.split(':',1)[0].upper() if ':' in part else ''
        if len(label)>45: return 2
        return 0 if 'CONCLUSION' in label else 1 if 'RESULT' in label or 'FINDING' in label else 2
    structured=any(priority(p)<2 for p in parts)
    if structured:
        best=min(priority(p) for p in parts)
        parts=[p for p in parts if priority(p)==best]
    passage=evidence.title+'\n'+'\n'.join(parts)
    budget=512-len(tokenizer.encode(claim,add_special_tokens=False))-3
    encoded=tokenizer(passage,add_special_tokens=False,return_offsets_mapping=True)
    if len(encoded['input_ids'])>budget and not structured:
        head=(budget-12)//3;tail=budget-12-head
        offsets=encoded['offset_mapping']
        passage=passage[:offsets[head-1][1]]+'\n[intervening passage omitted]\n'+passage[offsets[-tail][0]:]
    # Export exactly the premise tokens the model will see, without special tokens.
    ids=tokenizer.encode(passage,add_special_tokens=False)[:budget]
    return tokenizer.decode(ids,skip_special_tokens=True), structured

class BiomedicalNliAdapter:
    def __init__(self, config=None):
        self.config = config or PipelineConfig()
        self.loaded = False
        self.model_artifact = None

    def load(self):
        if self.loaded: return
        name = self.config.phase4.nli_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        mapping = {str(v).lower(): int(k) for k,v in self.model.config.id2label.items()}
        if set(mapping) != {'contradiction', 'neutral', 'entailment'}:
            raise ValueError('NLI model must declare contradiction, neutral, entailment labels')
        self.order = [mapping['contradiction'], mapping['neutral'], mapping['entailment']]
        self.device = self.config.runtime.device_override or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_artifact = ModelArtifact(component='phase4_biomedical_nli', model_name=name,
            version=getattr(self.model.config, '_commit_hash', None), metadata={
                'label_mapping': {i:v.value for i,v in enumerate(LABELS)},
                'input_order': 'evidence premise, claim hypothesis',
                'max_tokens':512, 'relevance_discount':True,
                'validation_scope':'development diagnostics; not clinically validated'})
        self.loaded = True

    def run(self, claim, evidence_list):
        self.load()
        started=datetime.now(timezone.utc); clock=time.perf_counter()
        if len(self.tokenizer.encode(claim, add_special_tokens=False)) > 240:
            raise ValueError('Please split this claim into shorter statements (under 240 model tokens each).')
        predictions=[]; weighted=[0.,0.,0.]; total=0.
        warnings=['Biomedical NLI deployment: different from the historical Phase 4 checkpoint. Scores are not calibrated clinical probabilities.']
        for evidence in evidence_list:
            passage, structured = evidence_passage(evidence, claim, self.tokenizer)
            warnings.append(f'{evidence.evidence_id}: scored a bounded passage with ' + ('the conclusion section (or results when no conclusion is present).' if structured else 'opening/closing context retained when needed.'))
            tokens=self.tokenizer(passage,claim,truncation='only_first',max_length=512,return_tensors='pt').to(self.device)
            with torch.inference_mode():
                logits=self.model(**tokens).logits[0][self.order].float().cpu()
                raw=logits.softmax(-1).tolist()
            values=relevance_discount(raw,evidence.relevance_score)
            probs=ClassProbabilities(refuted=values[0],nei=values[1],supported=values[2])
            label=LABELS[max(range(3),key=lambda i:values[i])]
            predictions.append(PairwiseVerification(evidence_id=evidence.evidence_id,probabilities=probs,predicted_label=label,raw_logits=logits.tolist(),scored_text=passage))
            weight=float(evidence.hierarchy_weight if evidence.hierarchy_weight is not None else 1.)
            for i in range(3):weighted[i]+=values[i]*weight
            total+=weight
        values=[v/total for v in weighted] if total else [0.,1.,0.]
        if not evidence_list: warnings.append('No retrieved evidence; returning insufficient evidence.')
        if any(p.predicted_label==Verdict.REFUTED for p in predictions) and any(p.predicted_label==Verdict.SUPPORTED for p in predictions):
            warnings.append('The sources produced conflicting pairwise classifications. Review the individual studies.')
        verdict=LABELS[max(range(3),key=lambda i:values[i])]
        probs=ClassProbabilities(refuted=values[0],nei=values[1],supported=values[2])
        return Phase4Result(status=PhaseStatus.SUCCESS,verdict=verdict,user_facing_verdict=verdict_to_user_facing(verdict),
            probabilities=probs,verifier_score=probs.for_verdict(verdict),pairwise_predictions=predictions,
            evidence_used_ids=[e.evidence_id for e in evidence_list],aggregation_method='hierarchy weighted mean of relevance-discounted biomedical NLI probabilities',
            label_mapping=dict(enumerate(LABELS)),model=self.model_artifact,warnings=warnings,
            timing=PhaseTiming(phase='phase4',started_at_utc=started,ended_at_utc=datetime.now(timezone.utc),latency_ms=(time.perf_counter()-clock)*1000))
