# Implementation chapter source map

This map identifies the code that implements each part of the capstone. Describe the **current dashboard route** separately from earlier phase experiments; they are not identical pipelines.

| Chapter topic | Code to inspect | What it implements |
|---|---|---|
| System flow and contracts | `backend/pipeline.py`, `backend/schemas.py`, `backend/pipeline_config.py` | Five-stage orchestration, typed intermediate results, configurable artifact roots |
| Phase 1: representation | `phases/phase1/src/models.py`, `dataset.py`, `transforms.py`, `train.py`; `backend/adapters/phase1_adapter.py` | BioBERT/ViT fusion and auxiliary image/text classification; adapter into the API |
| Phase 2: retrieval | `phases/phase2/retriever.py`, `reranker.py`; `inference/core.py`, `candidate.py`, `live_query.py`; `backend/services/pubmed_search.py` | Legacy retrieval research and the current BM25 + cross-encoder + optional PubMed route |
| Phase 3: evidence hierarchy | `phases/Phase3-main/phase3_classifier.py`, `evidence_taxonomy_v3.py`; `backend/adapters/phase3_adapter.py` | Evidence classification and authority metadata |
| Phase 4: claim verification | `phases/Phase4-main/src/model.py`, `trainer.py`, `train_and_evaluate_e2e_final.py`; `inference/candidate.py`, `passage_features.py`, `nli_contract.py`, `evidence_guard.py`, `relation_review.py` | Original verifier research and current passage-level NLI, combiner, and evidence checks |
| Phase 5: explanation | `phases/Phase5-main/src/orchestrator.py`, `grounding.py`, `prompting.py`; `backend/adapters/phase5_adapter.py` | Explanation generation and grounding after the verdict is locked |
| API and audit | `backend/api/`, `backend/services/persistence.py`, `inference/dashboard_backend.py` | HTTP endpoint, live experimental integration, saved inference/audit records |
| User interface | `dashboard/index.html`, `app.js`, `styles.css`, `insights.js` | Claim/image form, evidence views, result and failure diagnostics |
| Training additions | `inference/selector/train_selector.py`, `inference/train_combiner.py` | Supervised passage selector and boosted-tree evidence combiner training scripts |
| Tests and evidence | `tests/`, `results/` | Contract checks and compact measured results |

## What to say accurately

- The current dashboard verdict is based on the typed claim and retrieved text. Phase 1 image output is auxiliary; it has **no influence on the Phase 4 verdict**. Do not describe this as diagnostic medical-image classification.
- The current dashboard uses the experimental source under `inference/` in place of parts of the legacy Phase 2/4 adapters. The `phases/` tree documents phase research and training; it is not a second complete application.
- The recorded 67.0% classification result uses 300 repeatedly reused development cases. It is **not** independent held-out or current guarded-service accuracy. The separate 63.3% live-retrieval experiment has a different setup.
- Large corpora, training datasets, frozen checkpoints, Hugging Face caches, and generated run histories are absent by design. Consequently this repository supports explaining and reviewing the implementation, but does not by itself reproduce training or full inference on a clean machine.
- For methodology, include the exact data split, preprocessing, losses, architecture, selected checkpoints, retrieval depth, class mapping, evaluation denominators, and limitations from the phase reports and original experiment logs. Do not infer unrecorded ablation results from source code alone.
