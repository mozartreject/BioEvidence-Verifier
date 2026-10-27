from __future__ import annotations
from dataclasses import asdict
from tqdm import tqdm
from .llm import OllamaClient
from .grounding import SemanticGrounder, validate_generated
from .prompting import SYSTEM_PROMPT, generation_prompt, repair_prompt

class Phase5Orchestrator:
    def __init__(self,cfg:dict):
        if cfg["generation"]["provider"].lower()!="ollama":
            raise ValueError("This project release supports provider=ollama only")
        self.cfg=cfg
        self.client=OllamaClient(cfg["generation"])
        self.grounder=SemanticGrounder(cfg["grounding"]["semantic_model"])

    def _validate(self,record,output):
        return validate_generated(
            record=record,
            output=output,
            cfg=self.cfg["grounding"],
            min_words=int(self.cfg["generation"]["min_words"]),
            max_words=int(self.cfg["generation"]["max_words"]),
            grounder=self.grounder
        )

    def generate_one(self,record):
        attempts=[]
        try:
            output,meta=self.client.chat(SYSTEM_PROMPT,generation_prompt(record))
            validation=self._validate(record,output)
            attempts.append({"output":output,"validation":validation,"provider":meta})

            for _ in range(int(self.cfg["generation"]["repair_attempts"])):
                if validation["passed"]:
                    break
                output,meta=self.client.chat(
                    SYSTEM_PROMPT,
                    repair_prompt(record,output,validation["errors"])
                )
                validation=self._validate(record,output)
                attempts.append({"output":output,"validation":validation,"provider":meta})

            if not validation["passed"]:
                return None,{
                    "claim_key":record.claim_key,
                    "claim_text":record.claim_text,
                    "locked_verdict":record.verdict,
                    "attempts":attempts,
                    "final_errors":validation["errors"]
                }

            return {
                "claim_key":record.claim_key,
                "claim_text":record.claim_text,
                "verdict":record.verdict,
                "confidence":record.confidence,
                "explanation":str(output["explanation"]).strip(),
                "evidence_ids_used":[str(x) for x in output.get("evidence_ids_used",[])],
                "uncertainty_note":str(output.get("uncertainty_note","")).strip(),
                "validation":validation,
                "generation_metadata":{
                    "attempt_count":len(attempts),
                    "dataset":record.dataset,
                    "phase4_metadata":record.phase4_metadata,
                    "provider":attempts[-1]["provider"],
                    "evidence":[asdict(e) for e in record.evidence]
                }
            },None
        except Exception as e:
            return None,{
                "claim_key":record.claim_key,
                "claim_text":record.claim_text,
                "locked_verdict":record.verdict,
                "fatal_error":repr(e),
                "attempts":attempts
            }

    def run(self,records):
        successes=[]
        failures=[]
        for record in tqdm(records,desc="Phase 5 generation"):
            ok,fail=self.generate_one(record)
            if ok is not None:
                successes.append(ok)
            else:
                failures.append(fail)
        return successes,failures
