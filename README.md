# BioEvidence Verifier

A research capstone that checks a biomedical claim against retrieved literature and returns **Supported**, **Refuted**, or **Not enough information**, with the passages and model scores visible in a local dashboard.

## What is here

| Directory | Purpose |
|---|---|
| `dashboard/` | Current browser interface and local API bridge |
| `backend/` | Five-stage API contracts, orchestration, and adapters |
| `inference/` | Current 35,183-document retrieval, passage selection, evidence checks, combiner, and startup scripts |
| `phases/` | Selected source files from Phases 2–5 for research context |
| `results/` | Small, auditable evaluation summaries and current failure analysis |
| `assets/` | A synthetic upload test card and a sourced CDC educational graphic |
| `tests/` | Focused source-level regression tests |

## Pipeline

`Claim + optional image → representation → retrieve → rank/select passages → verify → explain → dashboard`

The current verifier uses BM25 over a local 35,183-abstract collection, a MiniLM cross-encoder, a supervised passage selector, PubMedBERT NLI, and a trained boosted-tree combiner. A live PubMed retry may supplement local retrieval. A separate relation review can **withhold** a decisive verdict when selected evidence does not justify it. Phase 5 explains the locked verdict. The image/text branch is experimental and **does not change the claim verdict or diagnose an image**.

## Run locally

This source snapshot is compact; it is **not a model-weight or corpus download**. Inference requires the original Phase 1 checkpoint, 35,183-document metadata, trained selector, Hugging Face model caches, Phase 3/5 dependencies, and Ollama with `qwen2.5:7b`. Set paths with `BIOVERIFY_PHASE1_ROOT`, `BIOVERIFY_PHASE1_CHECKPOINT`, `BIOVERIFY_PHASE2_ROOT`, `BIOVERIFY_PHASE3_ROOT`, `BIOVERIFY_PHASE4_ROOT`, `BIOVERIFY_PHASE5_ROOT`, `BIOVERIFY_CORPUS_METADATA`, and `BIOVERIFY_SELECTOR_MODEL` as needed. See `backend/pipeline_config.py` for adapter defaults. Python 3.9 and the project environment were used for the recorded runs.

On the original development machine, set `BIOVERIFY_PYTHON` to the existing project virtual environment's Python, then run:

```sh
python3 inference/start_project.py
```

Open `http://127.0.0.1:3001/`. The launcher checks Ollama, starts the API on port 8000 and the dashboard on port 3001, and refuses to silently replace unrelated servers.

Run source-level tests without model weights:

```sh
PYTHONPATH=inference python3 -m unittest discover -s tests -p 'test_*.py'
node --check dashboard/app.js
```

## Results and limits

- The passage-aware classifier scored **201/300 (67.0%)** on a **reused SciFact development set**. This is not independent held-out accuracy or the accuracy of the current guarded service.
- A separate 300-case live-retrieval experiment scored **190/300 (63.3%)**. Its setup and scope differ; these percentages should not be presented as a direct before/after improvement.
- A 30-case relation-review audit had 27 valid reviewer outputs. On those 27 cases, reviewer and classifier each matched **17** gold labels. The reviewer remains a veto, not a validated replacement classifier.
- Known failures involve irrelevant passages and reversal of intervention/outcome direction. A matching verdict with unrelated citations is still a failure. See [`results/VALIDATION.md`](results/VALIDATION.md).
- Image uploads are processed by an auxiliary image/text model trained on health-news and out-of-context examples. Image scores are not clinical-image accuracy and do not determine the verdict.

## Example

Claim: **“Antibiotics are effective against the viruses that cause the common cold.”** Expected: **Refuted**. The included [CDC graphic](assets/cdc-common-cold-antibiotics.jpg) is an educational upload example; the verdict is derived from typed text and retrieved research, not OCR of the graphic. [Original CDC image](https://www.cdc.gov/antibiotic-use/media/images/Common-Cold-Do-you-need-antibiotics.jpg).

## Repository scope

The large corpus, model checkpoints, Hugging Face caches, virtual environments, private run artifacts, and raw downloaded datasets are deliberately excluded. Reproducing model inference requires those external artifacts. This repository preserves the current source, small fitted combiner, selected results, and precise limitations for review.
