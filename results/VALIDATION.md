# BiomedicalVerifier validation, 27 October 2026

## Scope
This is a targeted diagnostic, not a new end-to-end accuracy claim. It uses the existing 35,183-document local collection and optional live PubMed retrieval. The historical 67.0% figure is on a reused 300-claim development set before the current evidence veto; it must not be presented as current service accuracy.

## Five dashboard checks
| Claim type | Expected | Observed |
|---|---|---|
| Cardiac rehabilitation and admissions | Supported | Supported |
| MMR and autism | Refuted | Refuted |
| Antibiotics for common cold compared with placebo | Refuted | Not enough information |
| Continued smoking after lung cancer vs quitting | Refuted | Not enough information |
| Precise vitamin D regimen and influenza A effect | Not enough information | Not enough information |

These five examples are diagnostic only. The last result had irrelevant evidence, so label agreement alone does not establish correct evidence-grounded reasoning.

## Failure analysis
- The antibiotics case retrieved a directly relevant PubMed abstract but ranked unrelated cold therapies and sinusitis treatment alongside it. Pairwise NLI gave false support to those mismatched passages, and the combiner proposed Supported. Independent relation review correctly disagreed, so the live decisive result was not accepted.
- The smoking case retrieved cessation studies. Pairwise NLI sometimes treated a benefit from quitting as support for a benefit from continuing. The combiner's live Refuted score was close to Supported, and the current guard lacked a pairwise Refuted signal. The live result was therefore withheld.
- These errors come from claim-evidence relation direction and passage relevance, not only corpus size.

## Reviewer audit
A deterministic stratified sample of 30 saved SciFact development claims was checked using the independent reviewer. It produced valid labels for 27. Both the saved classifier and reviewer matched 17/27 gold labels on those same 27 cases. Three reviewer outputs failed the required quote or format checks. The reviewer is retained as an experimental veto; the audit does not support promoting it to a general verdict override. Full records: `relation-review-audit.json`.

## Image scope
Phase 1 processes an uploaded image with a BioBERT/ViT binary auxiliary model trained on health-news and out-of-context image data. It is not validated as a medical imaging diagnostic model and does not affect the Phase 4 claim verdict. The dashboard now states whether that branch processed an image and explains its limited role. This is a transparency change, not an image accuracy improvement.

## Required evaluation before a classification replacement
Use held-out claims and report retrieval recall, claim-relevant passage precision, per-class precision/recall/F1, abstention coverage, and accepted-verdict accuracy. Keep source overlap separated between training and test. Evaluate direct claim/evidence relation modeling and intervention/outcome mismatch controls before deploying either. A 75% or 80% end-to-end figure must come from that complete service-level evaluation.
