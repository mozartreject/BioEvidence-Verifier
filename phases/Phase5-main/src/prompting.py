from __future__ import annotations
import json
from .models import ClaimRecord

SYSTEM_PROMPT="""You are the final explanation component of a biomedical fact-checking system.

The verdict has already been assigned by a separate verifier and is LOCKED.
Your role is explanation, not classification.

Rules:
1. Never change or contradict the locked verdict.
2. Use ONLY the supplied evidence.
3. Every substantive medical statement must be supported by supplied evidence.
4. Cite evidence IDs exactly as [E1], [E2], [E3].
5. Never invent PMIDs, DOIs, URLs, study names, sample sizes, percentages, dates, organizations, quotations, or statistics.
6. Do not use outside medical knowledge.
7. Prefer higher-authority evidence when evidence conflicts.
8. REFUTED means supplied evidence contradicts the claim.
9. SUPPORTED means supplied evidence supports the claim.
10. NEI means supplied evidence is insufficient or inconclusive; NEI does not mean false.
11. Preserve association-versus-causation distinctions.
12. Write 3-5 concise, clinically neutral sentences.
13. Return strict JSON only, without markdown fences.

Required schema:
{
  "verdict": "REFUTED|NEI|SUPPORTED",
  "explanation": "3-5 concise sentences with evidence citations",
  "evidence_ids_used": ["[E1]"],
  "uncertainty_note": ""
}
"""

def generation_prompt(record:ClaimRecord)->str:
    confidence="not provided" if record.confidence is None else f"{record.confidence:.4f}"
    evidence="\n\n".join(x.prompt_block() for x in record.evidence)
    return f"""LOCKED VERDICT: {record.verdict}
VERIFIER CONFIDENCE: {confidence}

CLAIM:
{record.claim_text}

SUPPLIED EVIDENCE:
{evidence}

Generate the explanation now. Return strict JSON only."""

def repair_prompt(record:ClaimRecord,previous:dict,errors:list[str])->str:
    evidence="\n\n".join(x.prompt_block() for x in record.evidence)
    return f"""The previous explanation failed validation.

LOCKED VERDICT: {record.verdict}

CLAIM:
{record.claim_text}

SUPPLIED EVIDENCE:
{evidence}

PREVIOUS OUTPUT:
{json.dumps(previous,ensure_ascii=False)}

VALIDATION ERRORS:
{json.dumps(errors,ensure_ascii=False)}

Repair the output without changing the verdict or inventing facts.
Return strict JSON only."""
