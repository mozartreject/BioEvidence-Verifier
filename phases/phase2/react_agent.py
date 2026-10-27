"""
================================================================================
PHASE 2 REACT AGENT: Local FAISS ONLY. No live API during evaluation.
OLLAMA VERSION — Singleton retriever, deduplication, robust parsing.
================================================================================
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from config import (
    BASE_DIR, OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TAGS_URL,
    TRACES_DIR, MAX_REACT_HOPS, PHASE2_OUTPUT_PATH
)
from validated_retriever import ValidatedHybridRetriever
from evidence_validator import SemanticEvidenceValidator


# ==============================================================================
# AGENT STATE
# ==============================================================================
@dataclass
class AgentState:
    claim: str
    claim_id: str
    true_label: int = None
    scratchpad: List[str] = field(default_factory=list)
    evidence_collected: List[dict] = field(default_factory=list)
    hop_count: int = 0
    max_hops: int = MAX_REACT_HOPS


# ==============================================================================
# OLLAMA LLM
# ==============================================================================
class OllamaLLM:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.url = OLLAMA_URL
        self.tags_url = OLLAMA_TAGS_URL
        
        print(f"Connecting to Ollama ({model_name})...")
        try:
            r = requests.get(self.tags_url, timeout=5)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                print(f"  Ollama running. Available: {models}")
                if not any(model_name in m for m in models):
                    print(f"  WARNING: {model_name} not found. Run: ollama pull {model_name}")
            else:
                print(f"  WARNING: Ollama status {r.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running at localhost:11434. Run 'ollama serve'.")
        except Exception as e:
            print(f"  WARNING: Could not verify Ollama: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["Observation:", "Continue from here."]
            }
        }
        try:
            r = requests.post(self.url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.exceptions.Timeout:
            print("  ERROR: Ollama timeout.")
            return "Thought: Timeout occurred.\nAction: stop('Timeout')"
        except Exception as e:
            print(f"  ERROR: Ollama failed: {e}")
            return f"Thought: Error occurred.\nAction: stop('Error: {e}')"


# ==============================================================================
# REACT AGENT
# ==============================================================================
class ReActAgent:
    def __init__(
        self,
        retriever=None,
    ):
        """
        Phase 2 validated retrieval architecture:

            SciFact BM25 @20
                    +
            BioBERT Dense @20 over 35K corpus
                    |
                    v
               Candidate Union
                    |
                    v
               Cross-Encoder
                    |
                    v
             Locked V3.1 Fusion

        Fusion:
            CE_mm
            + 0.50 * BM25_mm
            + 0.25 * Dense_mm

        The retriever already performs Cross-Encoder scoring
        and final fusion. ReAct MUST NOT rerank these results
        a second time.
        """

        self.llm = OllamaLLM()

        self.retriever = (
            retriever
            if retriever is not None
            else ValidatedHybridRetriever()
        )

        prompt_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "prompts",
            "react_system_prompt.txt"
        )

        with open(
            prompt_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.base_prompt = f.read()

        print(
            f"Loaded ReAct system prompt from: "
            f"{prompt_path}"
        )

    def parse_action(self, text: str) -> Optional[Dict]:
        """
        Robust parsing of Thought/Action from LLM output.
        Returns: {"thought": "...", "tool": "search_pubmed"|"stop", "argument": "..."}
        """
        # Extract thought
        thought_match = re.search(
            r'Thought:\s*(.*?)(?=\n\s*Action:|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else "No thought parsed."
        
        # Extract action
        action_match = re.search(
            r'Action:\s*(.*?)(?=\n|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        action_str = action_match.group(1).strip() if action_match else "STOP"
        
        # Parse tool and argument
        tool = "stop"
        argument = action_str
        
        # Match search_pubmed("...") with various quote styles
        pm_match = re.search(
            r'search_pubmed\(\s*["\']?(.*?)["\']?\s*\)', 
            action_str, 
            re.IGNORECASE
        )
        if pm_match:
            tool = "search_pubmed"
            argument = pm_match.group(1).strip()
        elif "stop" in action_str.lower():
            tool = "stop"
            # Extract reason if in parentheses
            reason_match = re.search(r'stop\(\s*["\']?(.*?)["\']?\s*\)', action_str, re.IGNORECASE)
            argument = reason_match.group(1).strip() if reason_match else action_str
        
        return {
            "thought": thought,
            "tool": tool,
            "argument": argument
        }
    
    def deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """Remove duplicate evidence by ID."""
        seen_ids = set()
        unique = []
        for ev in evidence_list:
            eid = ev.get('id', ev.get('pmid', ''))
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                unique.append(ev)
        return unique
    
    def search_local(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict]:
        """
        Run the locked V3.1 Phase 2 retrieval pipeline.

        ValidatedHybridRetriever already performs:

            SciFact BM25 @20
                    +
            BioBERT Dense @20 over 35K corpus
                    |
                    v
               Candidate Union
                    |
                    v
               Cross-Encoder
                    |
                    v
             Min-Max Fusion

        No additional reranking is performed here.
        """

        results = self.retriever.search(
            query,
            k=k,
        )

        if not results:
            return []

        # Preserve runtime compatibility while keeping the
        # complete V3.1 provenance produced by the retriever.
        for item in results:
            item["score"] = float(
                item.get(
                    "fusion_score",
                    item.get("score", 0.0),
                )
            )

            item["retrieval_score"] = float(
                item.get(
                    "fusion_score",
                    item.get(
                        "retrieval_score",
                        item["score"],
                    ),
                )
            )

        return results

    def search_validated_local(
        self,
        claim: str,
        query: str = None,
        k: int = 10,
        max_validate: int = 5,
    ) -> List[Dict]:
        """
        Retrieve, rerank, and semantically validate local evidence.

        Retrieval scores and CrossEncoder scores are ranking
        signals only. Final evidence acceptance is determined
        by the semantic EvidenceValidator.

        Parameters
        ----------
        claim:
            Original biomedical claim being investigated.

        query:
            Retrieval query. If omitted, the original claim
            is used.

        k:
            Number of CrossEncoder-ranked candidates returned
            by search_local().

        max_validate:
            Maximum number of top-ranked candidates passed to
            semantic validation.

        Returns
        -------
        List[Dict]
            Only documents judged VALID_EVIDENCE.
        """

        if query is None:
            query = claim

        print(
            f"[Validated Retrieval] "
            f"Searching local corpus..."
        )

        ranked = self.search_local(
            query,
            k=k,
        )

        if not ranked:
            print(
                "[Validated Retrieval] "
                "No retrieval candidates found."
            )
            return []

        print(
            f"[Validated Retrieval] "
            f"{len(ranked)} ranked candidates retrieved."
        )

        validator = SemanticEvidenceValidator(
            llm=self.llm,
        )

        valid = validator.valid_only(
            claim,
            ranked,
            max_documents=max_validate,
        )

        print(
            f"[Validated Retrieval] "
            f"{len(valid)}/{min(len(ranked), max_validate)} "
            f"documents accepted as semantic evidence."
        )

        if not valid:
            print(
                "[Validated Retrieval] "
                "STATUS: INSUFFICIENT_LOCAL_EVIDENCE"
            )

        return valid


    def investigate_claim(self, claim: str, claim_id: str, true_label: int = None) -> AgentState:
        """
        Run ReAct loop for one claim.
        Returns AgentState with collected evidence and trace.
        """
        state = AgentState(
            claim=claim, 
            claim_id=claim_id, 
            true_label=true_label, 
            max_hops=MAX_REACT_HOPS
        )
        
        prompt = self.base_prompt.format(claim=claim, max_hops=MAX_REACT_HOPS)
        conversation = prompt + "\n\n"
        
        print(f"\n{'='*70}")
        print(f"CLAIM: {claim[:80]}...")
        print(f"{'='*70}")
        
        while state.hop_count < state.max_hops:
            print(f"\n--- Hop {state.hop_count + 1}/{state.max_hops} ---")
            
            # Build scratchpad context
            scratchpad_text = "\n\n".join(state.scratchpad) if state.scratchpad else "No previous actions."
            full_prompt = conversation + f"\nHere is what has happened so far:\n{scratchpad_text}\n\nContinue from here."
            
            response = self.llm.generate(full_prompt, max_tokens=400, temperature=0.1)
            print(f"LLM: {response[:300]}...")
            
            parsed = self.parse_action(response)
            print(f"Parsed -> Thought: {parsed['thought'][:80]}...")
            print(f"Parsed -> Action: {parsed['tool']}('{parsed['argument'][:60]}...')")
            
            state.scratchpad.append(f"Thought: {parsed['thought']}\nAction: {parsed['tool']}('{parsed['argument']}')")
            
            if parsed['tool'] == 'stop':
                print("Agent stopped.")
                break
            
            if parsed['tool'] == 'search_pubmed':
                results = self.search_local(parsed['argument'], k=3)
                
                # Format observation for LLM
                obs_lines = []
                for i, res in enumerate(results[:3], 1):
                    obs_lines.append(f"Result {i} [{res['source']}] (ID: {res['id']}): {res['title']}")
                    obs_lines.append(f"  {res['text'][:250]}...")
                
                observation = "\n".join(obs_lines) if obs_lines else "No results found."
                
                # Collect evidence
                for res in results:
                    state.evidence_collected.append({
                        'id': res['id'],
                        'title': res.get('title', ''),
                        'text': res.get('text', ''),
                        'score': float(res.get('score', 0.0)),
                        'source': res.get('source', 'unknown'),
                        'query': parsed['argument'],

                        # Locked V3.1 retrieval provenance
                        'bm25_rank': res.get('bm25_rank'),
                        'bm25_raw': res.get('bm25_raw'),
                        'dense_rank': res.get('dense_rank'),
                        'dense_raw': res.get('dense_raw'),
                        'ce_raw': res.get('ce_raw'),
                        'ce_mm': res.get('ce_mm'),
                        'bm25_mm': res.get('bm25_mm'),
                        'dense_mm': res.get('dense_mm'),
                        'fusion_score': res.get('fusion_score'),
                    })
                
                state.scratchpad.append(f"Observation: {observation}")
            else:
                print(f"Unknown tool '{parsed['tool']}'. Stopping.")
                break
            
            state.hop_count += 1
            time.sleep(0.2)
        
        # Deduplicate
        state.evidence_collected = self.deduplicate_evidence(state.evidence_collected)
        
        # Sort by relevance score
        state.evidence_collected.sort(key=lambda x: x['score'], reverse=True)
        
        return state
    
    def save_trace(self, state: AgentState, output_dir: str = None):
        """Save human-readable trace file."""
        if output_dir is None:
            output_dir = TRACES_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        safe_id = re.sub(r'[^\w]', '_', state.claim_id)[:50]
        filepath = os.path.join(output_dir, f"{safe_id}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Claim ID: {state.claim_id}\n")
            f.write(f"Claim: {state.claim}\n")
            if state.true_label is not None:
                label_names = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
                f.write(f"True Label: {label_names.get(state.true_label, 'UNKNOWN')}\n")
            f.write(f"Hops used: {state.hop_count}/{state.max_hops}\n")
            f.write(f"Evidence collected: {len(state.evidence_collected)}\n")
            f.write("="*70 + "\n\n")
            for entry in state.scratchpad:
                f.write(entry + "\n\n")
        
        print(f"Trace saved: {filepath}")


# ==============================================================================
# STANDALONE: Save Phase 2 output in correct structure
# ==============================================================================
def save_phase2_output(all_states: List[AgentState], output_path: str = None):
    """
    Save structured output for Phase 3.
    CRITICAL FIX: Explicit claim association with metadata header.
    """
    if output_path is None:
        output_path = PHASE2_OUTPUT_PATH
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    label_map_reverse = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
    
    claims_output = []
    for state in all_states:
        evidence_list = []
        for ev in state.evidence_collected:
            evidence_list.append({
                "id": ev['id'],
                "title": ev.get('title', ''),
                "text": ev.get('text', '')[:1000],
                "score": float(ev.get('score', 0.0)),
                "source": ev.get('source', 'unknown'),
                "retrieval_query": ev.get('query', ''),

                "retrieval_provenance": {
                    "bm25_rank": ev.get('bm25_rank'),
                    "bm25_raw": ev.get('bm25_raw'),
                    "dense_rank": ev.get('dense_rank'),
                    "dense_raw": ev.get('dense_raw'),
                    "ce_raw": ev.get('ce_raw'),
                    "ce_mm": ev.get('ce_mm'),
                    "bm25_mm": ev.get('bm25_mm'),
                    "dense_mm": ev.get('dense_mm'),
                    "fusion_score": ev.get('fusion_score'),
                }
            })
        
        claims_output.append({
            "claim_id": state.claim_id,
            "claim": state.claim,
            "true_label": state.true_label,
            "true_label_name": label_map_reverse.get(state.true_label, "UNKNOWN"),
            "num_evidence": len(evidence_list),
            "evidence": evidence_list,
            "agent_metadata": {
                "num_hops": state.hop_count,
                "max_hops": state.max_hops
            }
        })
    
    final_output = {
        "metadata": {
            "total_claims": len(claims_output),
            "total_evidence": sum(c["num_evidence"] for c in claims_output),
            "model": OLLAMA_MODEL,
            "corpus": "scifact_local_faiss",
            "max_hops": MAX_REACT_HOPS
        },
        "claims": claims_output
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("PHASE 2 OUTPUT SAVED")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Claims: {final_output['metadata']['total_claims']}")
    print(f"Total evidence: {final_output['metadata']['total_evidence']}")
    
    # BUG FIX VERIFICATION
    if final_output['metadata']['total_claims'] == 0:
        print("🚨 CRITICAL: Output contains 0 claims. DO NOT proceed to Phase 3.")
    else:
        print("✅ Output structure verified. Safe for Phase 3.")
    print(f"{'='*70}")
    
    return output_path
