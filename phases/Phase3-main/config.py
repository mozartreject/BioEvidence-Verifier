"""
Phase 3 Configuration
NO TRAINING. Just paths, tiers, and weights.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
HEALTHFC_DIR = os.path.join(DATA_DIR, "healthfc")
PHASE2_DIR = os.path.join(DATA_DIR, "phase2_evidence")
OUTPUT_DIR = os.path.join(BASE_DIR, "phase3_output")

for d in [DATA_DIR, HEALTHFC_DIR, PHASE2_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# HealthFC paths
HEALTHFC_CSV = os.path.join(HEALTHFC_DIR, "healthfc.csv")
HEALTHFC_URL = "https://raw.githubusercontent.com/lingbozhang/healthfc/main/data/healthfc.csv"

# Phase 2 connection (optional — if you have Phase 2 output)
PHASE2_RETRIEVED = os.path.join(PHASE2_DIR, "retrieved_evidence.json")

# Phase 3 outputs
TIER_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "tier_predictions.csv")
WEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_weighted.json")
UNWEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_unweighted.json")
EVALUATION_REPORT_PATH = os.path.join(OUTPUT_DIR, "healthfc_evaluation.txt")

# Evidence Hierarchy Tiers
TIERS = {
    'A': {
        'name': 'High Quality',
        'weight': 1.0,
        'keywords': [
            'randomized controlled trial', 'randomised controlled trial', 'rct',
            'systematic review', 'meta-analysis', 'meta analysis', 'cochrane review',
            'pragmatic trial', 'cluster randomized', 'cluster randomised',
            'double-blind trial', 'double blind', 'single-blind', 'single blind',
            'triple-blind', 'triple blind', 'controlled trial', 'clinical trial',
            'placebo-controlled', 'placebo controlled', 'sham-controlled', 'sham controlled',
            'active-controlled', 'active controlled', 'parallel-group', 'parallel group',
            'multicenter trial', 'multi-center trial', 'mult centre trial',
            'randomly assigned', 'randomly allocated', 'randomization', 'randomisation',
            'phase ii trial', 'phase 2 trial', 'phase iii trial', 'phase 3 trial',
            'phase iv trial', 'phase 4 trial', 'equivalence trial', 'noninferiority trial',
            'non-inferiority trial', 'superiority trial', 'crossover trial', 'cross-over trial',
            'factorial design', 'intervention study', 'therapeutic trial',
            'double-masked', 'single-masked', 'masking', 'allocation concealment'
        ]
    },
    'B': {
        'name': 'Moderate Quality',
        'weight': 0.7,
        'keywords': [
            'cohort study', 'cohort analysis', 'prospective study', 'retrospective study',
            'observational study', 'observational analysis', 'cross-sectional study',
            'cross sectional', 'epidemiological study', 'registry study', 'registry data',
            'follow-up study', 'follow up study', 'panel study', 'longitudinal study',
            'case-control study', 'case control study', 'nested case-control',
            'matched cohort', 'nationwide cohort', 'population-based study',
            'population based', 'population-based cohort', 'multicenter observational',
            'prospective cohort', 'retrospective cohort', 'historical cohort',
            'consecutive patients', 'consecutive sample', 'chart review',
            'medical record review', 'database study', 'administrative data',
            'survey study', 'prevalence study', 'incidence study', 'prospective registry',
            'retrospective registry', 'prospective analysis', 'retrospective analysis',
            'propensity score', 'adjusting for', 'multivariable analysis',
            'prospective observational', 'retrospective observational', 'prospective', 'retrospective', 'observational', 'registry', 'database',
            'population-based', 'population based', 'nationwide', 'multicenter',
            'consecutive patients', 'medical records', 'health records',
            'propensity-matched', 'propensity matched', 'matched cohort',
            'longitudinal data', 'followed up', 'follow-up period',
            'incidence of', 'prevalence of', 'risk factors', 'associated with',
            'odds ratio', 'hazard ratio', 'relative risk', 'confidence interval',
            'adjusted for', 'multivariate', 'multivariable', 'logistic regression',
            'cox regression', 'survival analysis', 'kaplan-meier'
        ]
    },
    'C': {
        'name': 'Low Quality',
        'weight': 0.4,
        'keywords': [
            'case report', 'case series', 'case study', 'case presentation',
            'clinical vignette', 'case of', 'cases of', 'rare case', 'unusual case',
            'unique case', 'first case', 'first report', 'second case',
            'case illustrating', 'patient presented', 'patients presented',
            'we present a case', 'we present the case', 'we describe a case',
            'here we report', 'we report a case', 'case description', 'case history',
            'case account', 'clinical course', 'clinical picture', 'hospital course',
            'year-old patient', 'years old patient', 'year old patient',
            'year-old male', 'year-old female', 'years old male', 'years old female',
            'admitted to', 'presented to', 'presented with', 'diagnosed with',
            'to our knowledge', 'to the best of our knowledge', 'only few cases',
            'rare manifestation', 'unusual manifestation', 'rare presentation',
            'pilot study', 'feasibility study', 'exploratory study',
            'in vitro study', 'in vivo study', 'ex vivo', 'animal study',
            'animal model', 'murine model', 'rat model', 'mouse model',
            'preclinical study', 'bench research', 'cell culture', 'tissue culture',
            'expert opinion', 'narrative review', 'qualitative study',
            'proof of concept', 'proof-of-concept'
        ]
    },
    'D': {
        'name': 'Very Low Quality',
        'weight': 0.2,
        'keywords': [
            'editorial', 'editorial note', 'commentary', 'invited commentary',
            'letter', 'letter to the editor', 'correspondence', 'news',
            'opinion', 'opinion piece', 'perspective', 'viewpoint', 'point of view',
            'special article', 'special communication', 'book review',
            'conference abstract', 'meeting abstract', 'conference proceedings',
            'proceedings of', 'symposium summary', 'workshop summary',
            'preprint', 'medrxiv', 'biorxiv', 'arxiv', 'ssrn', 'chemrxiv',
            'hypothesis', 'hypotheses', 'concept paper', 'conceptual framework',
            'erratum', 'errata', 'correction to', 'corrigendum', 'retraction',
            'retracted', 'expression of concern', 'publisher note',
            'author response', 'reply to', 'rejoinder', 'response to',
            'blog', 'website', 'magazine article', 'news article',
            'policy brief', 'white paper', 'position statement', 'consensus statement'
        ]
    }
}

# LLM settings for optional LLM-based classification
LLM_MODEL = "deepseek-ai/deepseek-llm-7b-base"  # Change to your Phase 2 model if desired

# ADD THESE TWO LINES FOR OLLAMA COMPATIBILITY:
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"  # Or whatever local model you run via Ollama
# PubMed evaluation set
PUBMED_EVAL_DIR = os.path.join(DATA_DIR, "pubmed_evaluation")
os.makedirs(PUBMED_EVAL_DIR, exist_ok=True)

print("Phase 3 Config loaded.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Tiers: A={TIERS['A']['weight']}, B={TIERS['B']['weight']}, C={TIERS['C']['weight']}, D={TIERS['D']['weight']}")
