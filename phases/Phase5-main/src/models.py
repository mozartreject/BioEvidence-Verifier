from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

LABELS={0:"REFUTED",1:"NEI",2:"SUPPORTED"}

def normalize_verdict(v:Any)->str:
    if v is None:
        raise ValueError("Verdict missing")
    if isinstance(v,bool):
        raise ValueError("Boolean verdict invalid")
    if isinstance(v,(int,float)) and int(v)==v and int(v) in LABELS:
        return LABELS[int(v)]
    t=str(v).strip().upper()
    aliases={
        "0":"REFUTED","1":"NEI","2":"SUPPORTED",
        "FALSE":"REFUTED","REFUTED":"REFUTED","REFUTE":"REFUTED","CONTRADICTED":"REFUTED",
        "NEI":"NEI","UNPROVEN":"NEI","INSUFFICIENT":"NEI","NOT ENOUGH INFO":"NEI","NOT_ENOUGH_INFO":"NEI",
        "TRUE":"SUPPORTED","SUPPORTED":"SUPPORTED","SUPPORT":"SUPPORTED","ENTAILMENT":"SUPPORTED"
    }
    if t in aliases:
        return aliases[t]
    raise ValueError(f"Unknown verdict: {v!r}")

@dataclass
class Evidence:
    local_id:str
    source_id:str
    title:str
    text:str
    hierarchy:str|None=None
    hierarchy_weight:float|None=None
    retrieval_score:float|None=None
    weighted_score:float|None=None
    source:str|None=None
    metadata:dict=field(default_factory=dict)

    def prompt_block(self)->str:
        meta=[]
        if self.hierarchy: meta.append(f"Hierarchy={self.hierarchy}")
        if self.hierarchy_weight is not None: meta.append(f"HierarchyWeight={self.hierarchy_weight:.3f}")
        if self.retrieval_score is not None: meta.append(f"RetrievalScore={self.retrieval_score:.4f}")
        if self.weighted_score is not None: meta.append(f"WeightedScore={self.weighted_score:.4f}")
        if self.source: meta.append(f"Source={self.source}")
        meta_text="; ".join(meta) if meta else "unavailable"
        return f"{self.local_id}\nTitle: {self.title or '[untitled]'}\nMetadata: {meta_text}\nEvidence text:\n{self.text}"

@dataclass
class ClaimRecord:
    claim_key:str
    claim_text:str
    verdict:str
    confidence:float|None
    dataset:str|None
    evidence:list[Evidence]
    phase4_metadata:dict=field(default_factory=dict)
