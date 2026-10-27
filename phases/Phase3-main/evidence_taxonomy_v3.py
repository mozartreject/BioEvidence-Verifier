"""
Phase 3 — Evidence Hierarchy Taxonomy V3
=========================================

Conservative research-grade study-design classifier.

Hierarchy
---------
A = Randomized trials / systematic reviews / meta-analyses
B = Observational human evidence
C = Case reports / case series / preclinical / expert evidence
D = Editorial / commentary / preprint / non-peer-reviewed material

Design principles
-----------------
1. Explicit study design > generic statistical terminology.
2. Title evidence receives modest additional weight.
3. Overlapping/nested signals are suppressed.
4. Supporting signals cannot overwhelm explicit design evidence.
5. No signal causes abstention, not an arbitrary tier assignment.
6. Ambiguous/conflicting evidence causes abstention.
7. Publication type and study design conflicts are explicitly tracked.
8. Generic software/method terminology is NOT automatically Tier D.
9. Output remains compatible with Phase-3 consolidation scripts.
10. Deterministic and dependency-free.

Weights
-------
A = 1.0
B = 0.7
C = 0.4
D = 0.2
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import json
import math
import re


# =============================================================================
# VERSION
# =============================================================================

ENGINE_VERSION = "3.1.0"


# =============================================================================
# HIERARCHY
# =============================================================================

HIERARCHY_WEIGHTS = {
    "A": 1.0,
    "B": 0.7,
    "C": 0.4,
    "D": 0.2,
}

HIERARCHY_NAMES = {
    "A": "High-quality experimental/evidence-synthesis",
    "B": "Observational human evidence",
    "C": "Case/preclinical/expert evidence",
    "D": "Editorial/preprint/non-peer-reviewed evidence",
}

TIER_ORDER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


# =============================================================================
# SIGNAL STRENGTHS
# =============================================================================

DEFINITIVE = 10.0
STRONG = 7.0
SUPPORTING = 3.0
WEAK = 1.0

STRENGTH_RANK = {
    "weak": 0,
    "supporting": 1,
    "strong": 2,
    "definitive": 3,
}


# =============================================================================
# ENGINE PARAMETERS
# =============================================================================

DEFAULT_TITLE_MULTIPLIER = 1.35
DEFAULT_MINIMUM_CONFIDENT_SCORE = 7.0
DEFAULT_AMBIGUITY_MARGIN = 4.0

DEFINITIVE_BONUS = 2.0

MAX_SUPPORTING_MATCHES_PER_TIER = 5
MAX_STRONG_MATCHES_PER_TIER = 5

SUPPORTING_SCORE_CAP = 15.0
WEAK_SCORE_CAP = 3.0


# =============================================================================
# TIER A — RANDOMIZED / EVIDENCE SYNTHESIS
# =============================================================================

A_DEFINITIVE = [
    "randomized controlled trial",
    "randomised controlled trial",
    "randomized clinical trial",
    "randomised clinical trial",
    "randomized trial",
    "randomised trial",
    "cluster randomized trial",
    "cluster randomised trial",
    "cluster randomized controlled trial",
    "cluster randomised controlled trial",
    "pragmatic randomized trial",
    "pragmatic randomised trial",
    "randomized crossover trial",
    "randomised crossover trial",
    "randomized cross-over trial",
    "randomised cross-over trial",
    "factorial randomized trial",
    "factorial randomised trial",
    "controlled clinical trial",
    "placebo-controlled trial",
    "placebo controlled trial",
    "sham-controlled trial",
    "sham controlled trial",
    "active-controlled trial",
    "active controlled trial",
    "parallel-group randomized trial",
    "parallel group randomized trial",
    "parallel-group randomised trial",
    "parallel group randomised trial",
    "double-blind randomized trial",
    "double blind randomized trial",
    "double-blind randomised trial",
    "double blind randomised trial",
    "single-blind randomized trial",
    "single blind randomized trial",
    "triple-blind randomized trial",
    "superiority trial",
    "noninferiority trial",
    "non-inferiority trial",
    "equivalence trial",

    "systematic review",
    "systematic literature review",
    "systematic evidence review",
    "systematic review and meta-analysis",
    "systematic review and meta analysis",
    "meta-analysis",
    "meta analysis",
    "network meta-analysis",
    "network meta analysis",
    "bayesian meta-analysis",
    "bayesian meta analysis",
    "individual participant data meta-analysis",
    "individual patient data meta-analysis",
    "individual participant data meta analysis",
    "individual patient data meta analysis",
    "pairwise meta-analysis",
    "pairwise meta analysis",
    "dose-response meta-analysis",
    "dose response meta-analysis",
    "umbrella review",
    "overview of systematic reviews",
    "meta-review",
    "meta review",
    "cochrane review",
]

A_STRONG = [
    "randomly assigned",
    "randomly allocated",
    "random allocation",
    "random assignment",
    "random sequence generation",
    "allocation concealment",
    "concealed allocation",
    "double-blind",
    "double blind",
    "double-masked",
    "double masked",
    "single-blind",
    "single blind",
    "single-masked",
    "single masked",
    "triple-blind",
    "triple blind",
    "placebo-controlled",
    "placebo controlled",
    "sham-controlled",
    "sham controlled",
    "parallel-group",
    "parallel group",
    "intention-to-treat",
    "intention to treat",
    "per-protocol analysis",
    "per protocol analysis",
    "trial registration",
    "clinical trial registration",
    "registered clinical trial",
    "randomization sequence",
    "randomisation sequence",
    "randomization procedure",
    "randomisation procedure",

    "preferred reporting items for systematic reviews",
    "prisma guidelines",
    "prisma statement",
    "cochrane risk of bias",
    "risk of bias tool",
    "pooled effect estimate",
    "pooled effect size",
    "pooled odds ratio",
    "pooled risk ratio",
    "pooled relative risk",
    "pooled hazard ratio",
    "forest plot",
    "heterogeneity analysis",
    "publication bias",
    "funnel plot",
    "fixed-effect model",
    "fixed effect model",
    "random-effects model",
    "random effects model",
]

A_SUPPORTING = [
    "intervention group",
    "control group",
    "treatment group",
    "placebo group",
    "trial arm",
    "treatment arm",
    "control arm",
    "blinding",
    "masking",
    "allocation ratio",
    "primary endpoint",
    "secondary endpoint",
    "primary outcome",
    "secondary outcome",
    "protocol-specified",
    "prespecified outcome",
    "pre-specified outcome",
    "adverse events",
    "treatment effect",
    "intervention effect",
    "effect estimate",
    "heterogeneity",
    "subgroup analysis",
    "sensitivity analysis",
    "study selection",
    "database search",
    "literature search",
    "eligibility criteria",
    "inclusion criteria",
    "exclusion criteria",
]


# =============================================================================
# TIER B — OBSERVATIONAL HUMAN EVIDENCE
# =============================================================================

B_DEFINITIVE = [
    "prospective cohort study",
    "retrospective cohort study",
    "cohort study",
    "cohort analysis",
    "longitudinal cohort study",
    "population-based cohort study",
    "population based cohort study",
    "nationwide cohort study",
    "birth cohort study",
    "historical cohort study",
    "matched cohort study",
    "observational cohort study",

    "case-control study",
    "case control study",
    "nested case-control study",
    "nested case control study",
    "population-based case-control study",
    "matched case-control study",
    "matched case control study",

    "cross-sectional study",
    "cross sectional study",
    "cross-sectional analysis",
    "cross sectional analysis",

    "observational study",
    "observational analysis",
    "prospective observational study",
    "retrospective observational study",
    "multicenter observational study",
    "multi-center observational study",

    "registry study",
    "registry-based study",
    "registry based study",
    "prospective registry study",
    "retrospective registry study",

    "longitudinal study",
    "longitudinal analysis",
    "follow-up study",
    "follow up study",
    "population-based study",
    "population based study",
    "epidemiological study",
    "epidemiologic study",
    "prevalence study",
    "incidence study",
    "survey study",
    "ecological study",
    "ecologic study",
]

B_STRONG = [
    "prospective cohort",
    "retrospective cohort",
    "historical cohort",
    "matched cohort",
    "birth cohort",
    "study cohort",
    "observational cohort",
    "registry cohort",
    "nationwide cohort",
    "population cohort",
    "case-control",
    "case control",
    "cross-sectional",
    "cross sectional",
    "medical record review",
    "electronic health record study",
    "electronic medical record study",
    "chart review",
    "database study",
    "administrative database",
    "claims database",
    "registry data",
    "registry-based",
    "registry based",
    "prospectively enrolled",
    "retrospectively identified",
    "retrospectively reviewed",
    "consecutive patients",
    "followed prospectively",
    "followed longitudinally",
    "follow-up period",
    "follow up period",
    "person-years",
    "person years",
]

B_SUPPORTING = [
    "hazard ratio",
    "adjusted hazard ratio",
    "odds ratio",
    "adjusted odds ratio",
    "relative risk",
    "risk ratio",
    "incidence rate ratio",
    "prevalence ratio",
    "confidence interval",
    "multivariable regression",
    "multivariate regression",
    "logistic regression",
    "cox regression",
    "cox proportional hazards",
    "survival analysis",
    "kaplan-meier",
    "kaplan meier",
    "propensity score",
    "propensity matching",
    "propensity-matched",
    "propensity matched",
    "inverse probability weighting",
    "adjusted for confounders",
    "confounding variables",
    "covariate adjustment",
    "risk factors",
    "associated with",
    "association between",
    "incidence of",
    "prevalence of",
    "medical records",
    "health records",
    "electronic health records",
    "administrative data",
    "registry",
    "database",
    "survey respondents",
]


# =============================================================================
# TIER C — CASE / PRECLINICAL / EXPERT
# =============================================================================

C_DEFINITIVE = [
    "case report",
    "case series",
    "clinical case report",
    "case presentation",
    "clinical vignette",
    "single case report",
    "multiple case report",
    "case study",

    "animal study",
    "animal model study",
    "preclinical study",
    "pre-clinical study",
    "in vitro study",
    "in-vivo study",
    "in vivo study",
    "ex vivo study",
    "experimental animal study",

    "expert opinion",
    "expert consensus",
    "narrative review",
    "nonsystematic review",
    "non-systematic review",
    "qualitative study",
    "pilot study",
    "feasibility study",
    "proof-of-concept study",
    "proof of concept study",
]

C_STRONG = [
    "we present a case",
    "we present the case",
    "we report a case",
    "we describe a case",
    "here we report",
    "case description",
    "case history",
    "rare case",
    "unusual case",
    "unique case",
    "first case",
    "first reported case",
    "clinical course",
    "hospital course",
    "patient presented with",
    "patients presented with",
    "year-old male",
    "year-old female",
    "year old male",
    "year old female",

    "mouse model",
    "mice model",
    "murine model",
    "rat model",
    "rabbit model",
    "porcine model",
    "canine model",
    "animal model",
    "cell culture",
    "tissue culture",
    "cultured cells",
    "cell line",
    "experimental mice",
    "experimental rats",
    "laboratory animals",
    "murine study",
    "rodent study",

    "proof of concept",
    "proof-of-concept",
    "pilot investigation",
    "feasibility assessment",
]

C_SUPPORTING = [
    "presented with",
    "admitted to",
    "diagnosed with",
    "treated with",
    "clinical presentation",
    "rare manifestation",
    "unusual manifestation",
    "rare presentation",
    "to our knowledge",
    "to the best of our knowledge",
    "few cases have been reported",
    "only a few cases",
    "histopathological examination",
    "immunohistochemistry",
    "western blot",
    "cell viability",
    "cell proliferation",
    "gene expression",
    "protein expression",
    "experimental model",
    "laboratory experiment",
    "mechanistic study",
    "exploratory study",
]


# =============================================================================
# TIER D — EDITORIAL / PREPRINT / NON-PEER-REVIEWED
# =============================================================================

D_DEFINITIVE = [
    "editorial",
    "editorial commentary",
    "invited commentary",
    "commentary",
    "letter to the editor",
    "research letter",
    "correspondence",
    "opinion piece",
    "opinion article",
    "perspective",
    "viewpoint",
    "point of view",
    "news article",
    "news report",
    "book review",

    "conference abstract",
    "meeting abstract",
    "conference proceedings",
    "meeting proceedings",
    "symposium summary",
    "workshop summary",

    "preprint",
    "medrxiv preprint",
    "biorxiv preprint",
    "arxiv preprint",
    "ssrn preprint",
    "chemrxiv preprint",

    "policy brief",
    "white paper",
    "position statement",
    "concept paper",
    "conceptual framework",

    "erratum",
    "corrigendum",
    "retraction notice",
    "expression of concern",
    "publisher note",
    "author response",
    "reply to",
    "response to",
]

D_STRONG = [
    "this editorial",
    "in this editorial",
    "this commentary",
    "in this commentary",
    "this perspective",
    "in this perspective",
    "this viewpoint",
    "in this viewpoint",
    "letter to editor",
    "dear editor",
    "comment on",
    "response to the article",
    "reply to the article",
    "author reply",
    "editor's note",
    "editors note",
    "not peer reviewed",
    "not peer-reviewed",
    "non-peer-reviewed",
    "non peer reviewed",
    "posted as a preprint",
    "preprint server",
    "conference presentation",
    "conference poster",
    "meeting report",
]

D_SUPPORTING = [
    "we argue",
    "we propose",
    "we discuss",
    "we comment",
    "our opinion",
    "our perspective",
    "policy recommendation",
    "policy recommendations",
    "conceptual discussion",
    "hypothesis article",
    "special communication",
    "magazine article",
    "website article",
    "blog post",
]


# =============================================================================
# PHRASE GROUPS
# =============================================================================

PHRASE_GROUPS = {
    "A": {
        "definitive": (A_DEFINITIVE, DEFINITIVE),
        "strong": (A_STRONG, STRONG),
        "supporting": (A_SUPPORTING, SUPPORTING),
    },
    "B": {
        "definitive": (B_DEFINITIVE, DEFINITIVE),
        "strong": (B_STRONG, STRONG),
        "supporting": (B_SUPPORTING, SUPPORTING),
    },
    "C": {
        "definitive": (C_DEFINITIVE, DEFINITIVE),
        "strong": (C_STRONG, STRONG),
        "supporting": (C_SUPPORTING, SUPPORTING),
    },
    "D": {
        "definitive": (D_DEFINITIVE, DEFINITIVE),
        "strong": (D_STRONG, STRONG),
        "supporting": (D_SUPPORTING, SUPPORTING),
    },
}


# =============================================================================
# BASE REGEX PATTERNS
# =============================================================================

REGEX_PATTERNS = {
    "A": [
        (r"\brandomi[sz](ed|ation|ing)\b", STRONG),
        (
            r"\brandomi[sz](ed)?\s+controlled\s+"
            r"(clinical\s+)?trial\b",
            DEFINITIVE,
        ),
        (r"\bcluster[- ]randomi[sz](ed)?\b", STRONG),
        (r"\bdouble[- ]blind(ed)?\b", STRONG),
        (r"\bsingle[- ]blind(ed)?\b", STRONG),
        (r"\btriple[- ]blind(ed)?\b", STRONG),
        (r"\bplacebo[- ]controlled\b", STRONG),
        (
            r"\bsystematic\s+"
            r"(literature\s+|evidence\s+)?review\b",
            DEFINITIVE,
        ),
        (r"\bmeta[- ]analys(is|es)\b", DEFINITIVE),
        (
            r"\bnetwork\s+meta[- ]analys(is|es)\b",
            DEFINITIVE,
        ),
        (
            r"\brandom[- ]effects?\s+model\b",
            SUPPORTING,
        ),
        (
            r"\bfixed[- ]effects?\s+model\b",
            SUPPORTING,
        ),
    ],

    "B": [
        (r"\bprospective\s+cohort\b", DEFINITIVE),
        (r"\bretrospective\s+cohort\b", DEFINITIVE),
        (r"\bcase[- ]control\b", DEFINITIVE),
        (r"\bcross[- ]sectional\b", DEFINITIVE),
        (
            r"\bobservational\s+"
            r"(study|analysis|cohort)\b",
            DEFINITIVE,
        ),
        (r"\bregistry[- ]based\b", STRONG),
        (r"\bhazard\s+ratio\b", SUPPORTING),
        (r"\bodds\s+ratio\b", SUPPORTING),
        (
            r"\bcox\s+"
            r"(proportional\s+hazards?|regression)\b",
            SUPPORTING,
        ),
    ],

    "C": [
        (
            r"\bcase\s+(report|series|presentation)\b",
            DEFINITIVE,
        ),
        (
            r"\bwe\s+(present|report|describe)\s+"
            r"(a|the)\s+case\b",
            STRONG,
        ),
        (
            r"\b\d{1,3}[- ]year[- ]old\s+"
            r"(male|female|man|woman|patient)\b",
            STRONG,
        ),
        (
            r"\b(in[- ]?vitro|in[- ]?vivo|ex[- ]?vivo)\b",
            DEFINITIVE,
        ),
        (
            r"\b(mouse|mice|murine|rat|rodent|rabbit|porcine)"
            r"\s+model\b",
            STRONG,
        ),
        (r"\bcell\s+(culture|line)\b", STRONG),
    ],

    "D": [
        (
            r"\b(editorial|commentary|viewpoint|perspective)\b",
            DEFINITIVE,
        ),
        (
            r"\bletter\s+to\s+the\s+editor\b",
            DEFINITIVE,
        ),
        (
            r"\b(preprint|medrxiv|biorxiv|chemrxiv|arxiv)\b",
            DEFINITIVE,
        ),
        (
            r"\bconference\s+(abstract|proceedings)\b",
            DEFINITIVE,
        ),
        (
            r"\bnot\s+peer[- ]reviewed\b",
            DEFINITIVE,
        ),
        (
            r"\b(retraction|erratum|corrigendum)\b",
            DEFINITIVE,
        ),
    ],
}


# =============================================================================
# ADDITIONAL HIGH-PRECISION DESIGN PATTERNS
# =============================================================================

DESIGN_PATTERNS = {
    "A": [
        (
            r"\bparticipants?\s+(?:were\s+)?"
            r"random(?:ly|ized|ised)\s+"
            r"(?:assigned|allocated)\b",
            8.0,
            "definitive",
        ),
        (
            r"\bpooled\s+"
            r"(?:analysis|estimate|estimates|effect|effects)\b",
            5.0,
            "strong",
        ),
        (
            r"\bprisma\b",
            5.0,
            "strong",
        ),
        (
            r"\bcochrane\b.{0,30}\breview\b",
            8.0,
            "definitive",
        ),
        (
            r"\bforest\s+plot\b",
            3.0,
            "supporting",
        ),
        (
            r"\bpublication\s+bias\b",
            3.0,
            "supporting",
        ),
        (
            r"\brisk\s+of\s+bias\b",
            3.0,
            "supporting",
        ),
        (
            r"\bdatabases?\s+(?:were\s+)?searched\b",
            3.0,
            "supporting",
        ),
        (
            r"\beligibility\s+criteria\b",
            3.0,
            "supporting",
        ),
        (
            r"\binclusion\s+criteria\b.{0,50}"
            r"\b(studies|trials|articles)\b",
            3.0,
            "supporting",
        ),
    ],

    "B": [
        (
            r"\bparticipants?\s+(?:were\s+)?"
            r"(?:followed|observed)\b",
            5.0,
            "strong",
        ),
        (
            r"\bfollowed\s+for\s+\d+",
            5.0,
            "strong",
        ),
        (
            r"\bbaseline\b.{0,80}\bfollow[- ]?up\b",
            3.0,
            "supporting",
        ),
        (
            r"\bprospectively\s+"
            r"(?:collected|enrolled|followed)\b",
            5.0,
            "strong",
        ),
        (
            r"\bretrospectively\s+"
            r"(?:reviewed|identified|collected)\b",
            5.0,
            "strong",
        ),
        (
            r"\bmedical\s+records?\s+"
            r"(?:were\s+)?reviewed\b",
            5.0,
            "strong",
        ),
        (
            r"\bpatients?\s+(?:were\s+)?"
            r"identified\s+from\b",
            3.0,
            "supporting",
        ),
        (
            r"\bnationwide\b.{0,60}"
            r"\b(database|registry|cohort)\b",
            5.0,
            "strong",
        ),
        (
            r"\bpopulation[- ]based\b",
            5.0,
            "strong",
        ),
        (
            r"\bmatched\s+controls?\b",
            3.0,
            "supporting",
        ),
        (
            r"\bcase\s+patients?\b.{0,80}"
            r"\bcontrols?\b",
            5.0,
            "strong",
        ),
    ],

    "C": [
        (
            r"\b(?:cell|cells)\s+(?:were\s+)?"
            r"(?:cultured|grown|treated|transfected)\b",
            5.0,
            "strong",
        ),
        (
            r"\bcell\s+lines?\b",
            5.0,
            "strong",
        ),
        (
            r"\bknockout\s+mice\b",
            5.0,
            "strong",
        ),
        (
            r"\btransgenic\s+mice\b",
            5.0,
            "strong",
        ),
        (
            r"\bwild[- ]type\s+mice\b",
            5.0,
            "strong",
        ),
        (
            r"\bmice\s+were\b",
            3.0,
            "supporting",
        ),
        (
            r"\brats\s+were\b",
            3.0,
            "supporting",
        ),
        (
            r"\banimals\s+were\b",
            3.0,
            "supporting",
        ),
        (
            r"\bxenografts?\b",
            5.0,
            "strong",
        ),
        (
            r"\borganoids?\b",
            5.0,
            "strong",
        ),
        (
            r"\bwestern\s+blot(?:ting)?\b",
            3.0,
            "supporting",
        ),
        (
            r"\bimmunoblot(?:ting)?\b",
            3.0,
            "supporting",
        ),
        (
            r"\bimmunofluorescence\b",
            3.0,
            "supporting",
        ),
        (
            r"\bflow\s+cytometr(?:y|ic)\b",
            3.0,
            "supporting",
        ),
        (
            r"\btransfect(?:ed|ion)\b",
            3.0,
            "supporting",
        ),
        (
            r"\bgene\s+knockdown\b",
            3.0,
            "supporting",
        ),
    ],

    "D": [
        (
            r"\bthis\s+editorial\b",
            8.0,
            "definitive",
        ),
        (
            r"\bthis\s+commentary\b",
            8.0,
            "definitive",
        ),
        (
            r"\bthis\s+viewpoint\b",
            8.0,
            "definitive",
        ),
        (
            r"\bthis\s+perspective\b",
            8.0,
            "definitive",
        ),
        (
            r"\bposted\s+as\s+a\s+preprint\b",
            10.0,
            "definitive",
        ),
        (
            r"\bpreprint\s+server\b",
            8.0,
            "definitive",
        ),
        (
            r"\bnot\s+peer[- ]reviewed\b",
            10.0,
            "definitive",
        ),
    ],
}


# =============================================================================
# CONTEXT FLAGS
# =============================================================================

CONTEXT_FLAGS = {
    "protocol": [
        r"\bstudy\s+protocol\b",
        r"\btrial\s+protocol\b",
        r"\bprotocol\s+for\s+(?:a|an|the)?\s*"
        r"systematic\s+review\b",
        r"\bsystematic\s+review\s+protocol\b",
    ],

    "secondary_analysis": [
        r"\bsecondary\s+analysis\b",
        r"\bpost[- ]hoc\s+analysis\b",
        r"\bposthoc\s+analysis\b",
        r"\bsubstudy\b",
        r"\bsub[- ]study\b",
    ],

    "discussion_of_other_study": [
        r"\bcommentary\s+on\b",
        r"\bcomment\s+on\b",
        r"\bdiscuss(?:es|ing)?\s+the\s+"
        r"(?:trial|study)\b",
        r"\bwe\s+comment\s+on\b",
    ],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Match:
    tier: str
    signal: str
    strength: str
    score: float
    source: str
    location: str
    start: int
    end: int


@dataclass
class RuleResult:
    tier: str
    hierarchy_weight: float
    confidence: float
    margin: float
    scores: Dict[str, float]
    matches: List[Dict]
    context_flags: List[str]
    ambiguous: bool

    no_signal: bool = False
    abstained: bool = False
    status: str = ""

    runner_up: Optional[str] = None
    signal_tiers: Optional[List[str]] = None
    definitive_tiers: Optional[List[str]] = None

    best_score: float = 0.0
    second_score: float = 0.0
    evidence_strength: str = "none"

    engine_version: str = ENGINE_VERSION


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_text(text: str) -> str:

    text = str(
        text or ""
    ).lower()

    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =============================================================================
# ENGINE
# =============================================================================

class EvidenceRuleEngine:

    VERSION = ENGINE_VERSION

    def __init__(
        self,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        minimum_confident_score: float = DEFAULT_MINIMUM_CONFIDENT_SCORE,
        title_multiplier: float = DEFAULT_TITLE_MULTIPLIER,
    ):

        self.ambiguity_margin = float(
            ambiguity_margin
        )

        self.minimum_confident_score = float(
            minimum_confident_score
        )

        self.title_multiplier = float(
            title_multiplier
        )

        self._compiled_phrases = (
            self._compile_phrases()
        )

        self._compiled_regex = (
            self._compile_regex()
        )

        self._compiled_design = (
            self._compile_design()
        )

        self._compiled_context = (
            self._compile_context()
        )


    # =========================================================================
    # COMPILATION
    # =========================================================================

    @staticmethod
    def _phrase_pattern(
        phrase: str,
    ) -> re.Pattern:

        normalized = normalize_text(
            phrase
        )

        escaped = re.escape(
            normalized
        )

        escaped = escaped.replace(
            r"\ ",
            r"\s+",
        )

        return re.compile(
            rf"(?<!\w){escaped}(?!\w)",
            re.I,
        )


    def _compile_phrases(
        self,
    ):

        output = []

        seen = set()

        for tier in "ABCD":

            for strength, (
                phrases,
                score,
            ) in PHRASE_GROUPS[
                tier
            ].items():

                for phrase in phrases:

                    normalized = (
                        normalize_text(
                            phrase
                        )
                    )

                    key = (
                        tier,
                        strength,
                        normalized,
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    output.append(
                        (
                            tier,
                            normalized,
                            strength,
                            float(score),
                            self._phrase_pattern(
                                normalized
                            ),
                        )
                    )

        return tuple(output)


    @staticmethod
    def _compile_regex():

        output = []

        for tier in "ABCD":

            for pattern, score in (
                REGEX_PATTERNS[
                    tier
                ]
            ):

                if score >= DEFINITIVE:
                    strength = "definitive"

                elif score >= STRONG:
                    strength = "strong"

                elif score >= SUPPORTING:
                    strength = "supporting"

                else:
                    strength = "weak"

                output.append(
                    (
                        tier,
                        pattern,
                        strength,
                        float(score),
                        re.compile(
                            pattern,
                            re.I,
                        ),
                    )
                )

        return tuple(output)


    @staticmethod
    def _compile_design():

        output = []

        for tier in "ABCD":

            for (
                pattern,
                score,
                strength,
            ) in DESIGN_PATTERNS[
                tier
            ]:

                output.append(
                    (
                        tier,
                        pattern,
                        strength,
                        float(score),
                        re.compile(
                            pattern,
                            re.I,
                        ),
                    )
                )

        return tuple(output)


    @staticmethod
    def _compile_context():

        return {
            name: tuple(
                re.compile(
                    pattern,
                    re.I,
                )
                for pattern in patterns
            )
            for name, patterns
            in CONTEXT_FLAGS.items()
        }


    # =========================================================================
    # MATCHING
    # =========================================================================

    def _multiplier(
        self,
        location: str,
    ) -> float:

        if location == "title":
            return self.title_multiplier

        return 1.0


    def _match_rules(
        self,
        text: str,
        location: str,
        rules,
        source: str,
    ) -> List[Match]:

        if not text:
            return []

        multiplier = (
            self._multiplier(
                location
            )
        )

        matches = []

        for (
            tier,
            signal,
            strength,
            base_score,
            pattern,
        ) in rules:

            for found in pattern.finditer(
                text
            ):

                matches.append(
                    Match(
                        tier=tier,
                        signal=signal,
                        strength=strength,
                        score=(
                            base_score
                            * multiplier
                        ),
                        source=source,
                        location=location,
                        start=found.start(),
                        end=found.end(),
                    )
                )

        return matches


    def _collect_matches(
        self,
        title: str,
        abstract: str,
    ) -> List[Match]:

        matches = []

        for text, location in (
            (title, "title"),
            (abstract, "abstract"),
        ):

            if not text:
                continue

            matches.extend(
                self._match_rules(
                    text,
                    location,
                    self._compiled_phrases,
                    "phrase",
                )
            )

            matches.extend(
                self._match_rules(
                    text,
                    location,
                    self._compiled_regex,
                    "regex",
                )
            )

            matches.extend(
                self._match_rules(
                    text,
                    location,
                    self._compiled_design,
                    "design",
                )
            )

        return matches


    # =========================================================================
    # DEDUPLICATION
    # =========================================================================

    @staticmethod
    def _deduplicate_exact(
        matches: List[Match],
    ) -> List[Match]:

        best = {}

        for match in matches:

            key = (
                match.tier,
                match.location,
                match.start,
                match.end,
            )

            current = best.get(
                key
            )

            if current is None:

                best[key] = match

                continue

            current_rank = (
                STRENGTH_RANK.get(
                    current.strength,
                    -1,
                )
            )

            new_rank = (
                STRENGTH_RANK.get(
                    match.strength,
                    -1,
                )
            )

            if (
                match.score > current.score
                or (
                    math.isclose(
                        match.score,
                        current.score,
                    )
                    and
                    new_rank > current_rank
                )
            ):

                best[key] = match

        return list(
            best.values()
        )


    @staticmethod
    def _suppress_nested(
        matches: List[Match],
    ) -> List[Match]:

        """
        Suppress weaker same-tier signals fully contained inside
        stronger/longer signals at the same text location.

        Example:
            "prospective cohort study"
        should not independently receive full credit for:
            "cohort study"
            "prospective cohort"
        """

        grouped = defaultdict(
            list
        )

        for match in matches:

            grouped[
                (
                    match.tier,
                    match.location,
                )
            ].append(
                match
            )

        kept = []

        for _, group in (
            grouped.items()
        ):

            ordered = sorted(
                group,
                key=lambda m: (
                    -STRENGTH_RANK.get(
                        m.strength,
                        0,
                    ),
                    -m.score,
                    -(m.end - m.start),
                    m.start,
                    m.signal,
                ),
            )

            accepted = []

            for candidate in ordered:

                nested = False

                for existing in accepted:

                    contains = (
                        existing.start
                        <= candidate.start
                        and
                        existing.end
                        >= candidate.end
                    )

                    stronger = (
                        existing.score
                        >= candidate.score
                    )

                    if (
                        contains
                        and stronger
                    ):
                        nested = True
                        break

                if not nested:
                    accepted.append(
                        candidate
                    )

            kept.extend(
                accepted
            )

        return kept


    @staticmethod
    def _deduplicate_semantic(
        matches: List[Match],
    ) -> List[Match]:

        """
        Regex + phrase + design rules can identify essentially the same
        local signal. Keep the strongest signal for strongly overlapping
        spans of the same tier/location.
        """

        grouped = defaultdict(
            list
        )

        for match in matches:

            grouped[
                (
                    match.tier,
                    match.location,
                )
            ].append(
                match
            )

        output = []

        for _, group in (
            grouped.items()
        ):

            ordered = sorted(
                group,
                key=lambda m: (
                    -m.score,
                    -STRENGTH_RANK.get(
                        m.strength,
                        0,
                    ),
                    -(m.end - m.start),
                    m.start,
                ),
            )

            selected = []

            for candidate in ordered:

                duplicate = False

                for existing in selected:

                    overlap_start = max(
                        candidate.start,
                        existing.start,
                    )

                    overlap_end = min(
                        candidate.end,
                        existing.end,
                    )

                    overlap = max(
                        0,
                        overlap_end
                        - overlap_start,
                    )

                    candidate_len = max(
                        1,
                        candidate.end
                        - candidate.start,
                    )

                    existing_len = max(
                        1,
                        existing.end
                        - existing.start,
                    )

                    overlap_ratio = (
                        overlap
                        / min(
                            candidate_len,
                            existing_len,
                        )
                    )

                    if overlap_ratio >= 0.80:

                        if (
                            candidate.strength
                            == existing.strength
                            or
                            candidate.source
                            != existing.source
                        ):
                            duplicate = True
                            break

                if not duplicate:
                    selected.append(
                        candidate
                    )

            output.extend(
                selected
            )

        return output


    def _clean_matches(
        self,
        matches: List[Match],
    ) -> List[Match]:

        matches = (
            self._deduplicate_exact(
                matches
            )
        )

        matches = (
            self._suppress_nested(
                matches
            )
        )

        matches = (
            self._deduplicate_semantic(
                matches
            )
        )

        return matches


    # =========================================================================
    # CONTEXT
    # =========================================================================

    def _context_flags(
        self,
        text: str,
    ) -> List[str]:

        flags = []

        for name, patterns in (
            self._compiled_context.items()
        ):

            if any(
                pattern.search(
                    text
                )
                for pattern in patterns
            ):

                flags.append(
                    name
                )

        return sorted(
            flags
        )


    # =========================================================================
    # SCORE AGGREGATION
    # =========================================================================

    @staticmethod
    def _score_matches(
        matches: List[Match],
    ) -> Dict[str, float]:

        scores = {
            tier: 0.0
            for tier in "ABCD"
        }

        by_tier_strength = defaultdict(
            list
        )

        for match in matches:

            by_tier_strength[
                (
                    match.tier,
                    match.strength,
                )
            ].append(
                match
            )

        for tier in "ABCD":

            definitive = sorted(
                by_tier_strength[
                    (
                        tier,
                        "definitive",
                    )
                ],
                key=lambda x: -x.score,
            )

            strong = sorted(
                by_tier_strength[
                    (
                        tier,
                        "strong",
                    )
                ],
                key=lambda x: -x.score,
            )

            supporting = sorted(
                by_tier_strength[
                    (
                        tier,
                        "supporting",
                    )
                ],
                key=lambda x: -x.score,
            )

            weak = sorted(
                by_tier_strength[
                    (
                        tier,
                        "weak",
                    )
                ],
                key=lambda x: -x.score,
            )

            definitive_score = sum(
                m.score
                for m in definitive
            )

            strong_score = sum(
                m.score
                for m in strong[
                    :MAX_STRONG_MATCHES_PER_TIER
                ]
            )

            supporting_score = min(
                sum(
                    m.score
                    for m in supporting[
                        :MAX_SUPPORTING_MATCHES_PER_TIER
                    ]
                ),
                SUPPORTING_SCORE_CAP,
            )

            weak_score = min(
                sum(
                    m.score
                    for m in weak
                ),
                WEAK_SCORE_CAP,
            )

            scores[tier] = (
                definitive_score
                + strong_score
                + supporting_score
                + weak_score
            )

            if definitive:
                scores[tier] += (
                    DEFINITIVE_BONUS
                )

        return scores


    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def _confidence(
        best_score: float,
        second_score: float,
        signal_tier_count: int,
    ) -> float:

        if best_score <= 0:
            return 0.0

        separation = (
            best_score
            - second_score
        ) / (
            best_score
            + second_score
            + 1e-9
        )

        separation = max(
            0.0,
            min(
                separation,
                1.0,
            ),
        )

        magnitude = (
            1.0
            - math.exp(
                -best_score / 12.0
            )
        )

        confidence = (
            0.60 * separation
            + 0.40 * magnitude
        )

        if signal_tier_count > 1:

            penalty = (
                1.0
                - 0.08
                * min(
                    signal_tier_count - 1,
                    3,
                )
            )

            confidence *= penalty

        return float(
            max(
                0.0,
                min(
                    confidence,
                    1.0,
                ),
            )
        )


    # =========================================================================
    # EVIDENCE STRENGTH
    # =========================================================================

    @staticmethod
    def _evidence_strength(
        matches: List[Match],
        tier: str,
    ) -> str:

        strengths = [
            STRENGTH_RANK.get(
                match.strength,
                0,
            )
            for match in matches
            if match.tier == tier
        ]

        if not strengths:
            return "none"

        highest = max(
            strengths
        )

        reverse = {
            value: key
            for key, value
            in STRENGTH_RANK.items()
        }

        return reverse.get(
            highest,
            "none",
        )


    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def classify(
        self,
        title: str,
        abstract: str,
    ) -> RuleResult:

        title_n = normalize_text(
            title
        )

        abstract_n = normalize_text(
            abstract
        )

        combined = (
            title_n
            + "\n"
            + abstract_n
        ).strip()

        raw_matches = (
            self._collect_matches(
                title_n,
                abstract_n,
            )
        )

        matches = (
            self._clean_matches(
                raw_matches
            )
        )

        context_flags = (
            self._context_flags(
                combined
            )
        )

        scores = (
            self._score_matches(
                matches
            )
        )

        # ---------------------------------------------------------------------
        # Context corrections
        # ---------------------------------------------------------------------

        if (
            "protocol"
            in context_flags
        ):

            scores["A"] *= 0.50


        if (
            "secondary_analysis"
            in context_flags
        ):

            scores["A"] *= 0.70


        if (
            "discussion_of_other_study"
            in context_flags
        ):

            scores["D"] += 8.0


        # ---------------------------------------------------------------------
        # Signal summaries
        # ---------------------------------------------------------------------

        signal_tiers = sorted(
            {
                match.tier
                for match in matches
                if match.score > 0
            },
            key=lambda tier: (
                TIER_ORDER[tier]
            ),
        )

        definitive_tiers = sorted(
            {
                match.tier
                for match in matches
                if (
                    match.strength
                    == "definitive"
                )
            },
            key=lambda tier: (
                TIER_ORDER[tier]
            ),
        )


        # ---------------------------------------------------------------------
        # Ranking
        # ---------------------------------------------------------------------

        ordered = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                TIER_ORDER[
                    item[0]
                ],
            ),
        )

        best_tier, best_score = (
            ordered[0]
        )

        runner_up, second_score = (
            ordered[1]
        )

        margin = max(
            0.0,
            best_score
            - second_score,
        )

        no_signal = (
            best_score <= 0.0
        )


        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        definitive_conflict = (
            len(
                definitive_tiers
            )
            > 1
        )

        if no_signal:

            status = "NO_SIGNAL"

            abstained = True

            ambiguous = False


        elif definitive_conflict:

            status = (
                "DEFINITIVE_CONFLICT"
            )

            abstained = True

            ambiguous = True


        elif (
            best_score
            < self.minimum_confident_score
        ):

            status = "WEAK_SIGNAL"

            abstained = True

            ambiguous = True


        elif (
            margin
            < self.ambiguity_margin
        ):

            status = "AMBIGUOUS"

            abstained = True

            ambiguous = True


        else:

            status = "CLASSIFIED"

            abstained = False

            ambiguous = False


        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        confidence = (
            self._confidence(
                best_score,
                second_score,
                len(
                    signal_tiers
                ),
            )
        )


        # ---------------------------------------------------------------------
        # Sort matches for diagnostics
        # ---------------------------------------------------------------------

        sorted_matches = sorted(
            matches,
            key=lambda match: (
                -match.score,
                -STRENGTH_RANK.get(
                    match.strength,
                    0,
                ),
                TIER_ORDER[
                    match.tier
                ],
                match.location,
                match.start,
                match.signal,
            ),
        )


        return RuleResult(
            tier=best_tier,

            hierarchy_weight=(
                HIERARCHY_WEIGHTS[
                    best_tier
                ]
            ),

            confidence=confidence,

            margin=float(
                margin
            ),

            scores={
                tier: float(
                    score
                )
                for tier, score
                in scores.items()
            },

            matches=[
                asdict(
                    match
                )
                for match
                in sorted_matches
            ],

            context_flags=(
                context_flags
            ),

            ambiguous=(
                ambiguous
            ),

            no_signal=(
                no_signal
            ),

            abstained=(
                abstained
            ),

            status=status,

            runner_up=(
                runner_up
                if not no_signal
                else None
            ),

            signal_tiers=(
                signal_tiers
            ),

            definitive_tiers=(
                definitive_tiers
            ),

            best_score=float(
                best_score
            ),

            second_score=float(
                second_score
            ),

            evidence_strength=(
                self._evidence_strength(
                    matches,
                    best_tier,
                )
            ),

            engine_version=(
                ENGINE_VERSION
            ),
        )


    # =========================================================================
    # BATCH CLASSIFICATION
    # =========================================================================

    def classify_many(
        self,
        records: List[Dict],
        title_key: str = "title",
        abstract_key: str = "text",
    ) -> List[RuleResult]:

        return [
            self.classify(
                record.get(
                    title_key,
                    "",
                ),
                record.get(
                    abstract_key,
                    "",
                ),
            )
            for record in records
        ]


    # =========================================================================
    # ENGINE FINGERPRINT
    # =========================================================================

    def fingerprint(
        self,
    ) -> str:

        payload = {
            "version": (
                ENGINE_VERSION
            ),
            "weights": (
                HIERARCHY_WEIGHTS
            ),
            "phrase_groups": {
                tier: {
                    strength: phrases
                    for strength, (
                        phrases,
                        _,
                    )
                    in groups.items()
                }
                for tier, groups
                in PHRASE_GROUPS.items()
            },
            "regex": (
                REGEX_PATTERNS
            ),
            "design": (
                DESIGN_PATTERNS
            ),
            "context": (
                CONTEXT_FLAGS
            ),
            "parameters": {
                "title_multiplier": (
                    self.title_multiplier
                ),
                "minimum_confident_score": (
                    self.minimum_confident_score
                ),
                "ambiguity_margin": (
                    self.ambiguity_margin
                ),
            },
        }

        blob = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        return hashlib.sha256(
            blob
        ).hexdigest()


# =============================================================================
# TAXONOMY AUDIT
# =============================================================================

def taxonomy_statistics():

    print(
        "=" * 100
    )

    print(
        "PHASE 3 TAXONOMY V3 AUDIT"
    )

    print(
        "=" * 100
    )

    total_raw = 0
    total_unique = 0

    reverse = defaultdict(
        set
    )

    for tier in "ABCD":

        phrases = []

        print()
        print(
            f"TIER {tier}"
        )

        print(
            "-" * 100
        )

        for strength, (
            items,
            score,
        ) in PHRASE_GROUPS[
            tier
        ].items():

            normalized = [
                normalize_text(
                    item
                )
                for item in items
            ]

            unique = set(
                normalized
            )

            phrases.extend(
                normalized
            )

            for phrase in unique:

                reverse[
                    phrase
                ].add(
                    tier
                )

            print(
                f"{strength:12s} "
                f"raw={len(items):3d} "
                f"unique={len(unique):3d} "
                f"score={score:g}"
            )

        unique_all = set(
            phrases
        )

        duplicates = (
            len(
                phrases
            )
            - len(
                unique_all
            )
        )

        regex_count = len(
            REGEX_PATTERNS[
                tier
            ]
        )

        design_count = len(
            DESIGN_PATTERNS[
                tier
            ]
        )

        total_raw += len(
            phrases
        )

        total_unique += len(
            unique_all
        )

        print(
            f"{'TOTAL':12s} "
            f"raw={len(phrases):3d} "
            f"unique={len(unique_all):3d} "
            f"duplicates={duplicates:3d} "
            f"regex={regex_count:3d} "
            f"design={design_count:3d}"
        )


    collisions = {
        phrase: tiers
        for phrase, tiers
        in reverse.items()
        if len(
            tiers
        ) > 1
    }

    print()
    print(
        "-" * 100
    )

    print(
        "TOTAL RAW PHRASES:",
        total_raw,
    )

    print(
        "TOTAL UNIQUE WITHIN TIERS:",
        total_unique,
    )

    print(
        "TOTAL REGEX PATTERNS:",
        sum(
            len(
                patterns
            )
            for patterns
            in REGEX_PATTERNS.values()
        ),
    )

    print(
        "TOTAL DESIGN PATTERNS:",
        sum(
            len(
                patterns
            )
            for patterns
            in DESIGN_PATTERNS.values()
        ),
    )

    print(
        "CROSS-TIER EXACT COLLISIONS:",
        len(
            collisions
        ),
    )

    for phrase, tiers in sorted(
        collisions.items()
    ):

        print(
            " ",
            phrase,
            "=>",
            "/".join(
                sorted(
                    tiers
                )
            ),
        )


# =============================================================================
# SELF TEST
# =============================================================================

def self_test():

    engine = (
        EvidenceRuleEngine()
    )

    tests = [

        {
            "name": "RCT",
            "title": (
                "Randomized controlled trial of aspirin"
            ),
            "abstract": (
                "Participants were randomly assigned "
                "to placebo or treatment."
            ),
            "expected": "A",
        },

        {
            "name": "Systematic review",
            "title": (
                "Systematic review and meta-analysis "
                "of cardiovascular interventions"
            ),
            "abstract": (
                "Databases were searched and pooled "
                "estimates were calculated."
            ),
            "expected": "A",
        },

        {
            "name": "Prospective cohort",
            "title": (
                "Prospective cohort study of smoking "
                "and cardiovascular disease"
            ),
            "abstract": (
                "Participants were followed for ten "
                "years and hazard ratios were estimated."
            ),
            "expected": "B",
        },

        {
            "name": "Retrospective cohort",
            "title": (
                "Retrospective cohort study of "
                "postoperative outcomes"
            ),
            "abstract": (
                "Medical records were reviewed and "
                "adjusted odds ratios were calculated."
            ),
            "expected": "B",
        },

        {
            "name": "Case report",
            "title": (
                "A rare neurological presentation: "
                "case report"
            ),
            "abstract": (
                "A 45-year-old male presented with "
                "progressive neurological symptoms."
            ),
            "expected": "C",
        },

        {
            "name": "Mouse model",
            "title": (
                "Experimental tumour growth "
                "in a mouse model"
            ),
            "abstract": (
                "Mice were treated and tumour "
                "growth was measured."
            ),
            "expected": "C",
        },

        {
            "name": "In vitro",
            "title": (
                "In vitro analysis of cancer "
                "cell proliferation"
            ),
            "abstract": (
                "Cells were cultured and treated "
                "before western blot analysis."
            ),
            "expected": "C",
        },

        {
            "name": "Editorial discussing RCT",
            "title": (
                "Editorial: implications of a recent "
                "randomized controlled trial"
            ),
            "abstract": (
                "In this editorial we comment on the "
                "recently published trial."
            ),
            "expected": "D",
        },

        {
            "name": "Preprint",
            "title": (
                "Preprint: genomic surveillance "
                "during an outbreak"
            ),
            "abstract": (
                "This manuscript is posted as a preprint "
                "and has not been peer-reviewed."
            ),
            "expected": "D",
        },

        {
            "name": "No signal",
            "title": (
                "Protein localization in "
                "electron micrographs"
            ),
            "abstract": (
                "We investigate molecular localization "
                "using fluorescence nanoscopy."
            ),
            "expected": None,
        },

        {
            "name": "Software should not equal D",
            "title": (
                "STAR: ultrafast universal RNA-seq aligner"
            ),
            "abstract": (
                "We describe an algorithm for rapid "
                "alignment of RNA sequencing reads."
            ),
            "expected": None,
        },
    ]


    print()
    print(
        "=" * 100
    )

    print(
        "TAXONOMY V3 SELF TEST"
    )

    print(
        "=" * 100
    )

    passed = 0

    for test in tests:

        result = (
            engine.classify(
                test[
                    "title"
                ],
                test[
                    "abstract"
                ],
            )
        )

        expected = (
            test[
                "expected"
            ]
        )

        if expected is None:

            ok = (
                result.no_signal
            )

            expected_display = (
                "NO_SIGNAL"
            )

        else:

            ok = (
                result.tier
                == expected
                and
                not result.no_signal
            )

            expected_display = (
                expected
            )

        passed += int(
            ok
        )

        print()

        print(
            f"{'PASS' if ok else 'FAIL'} "
            f"| {test['name']}"
        )

        print(
            f"expected={expected_display} "
            f"| predicted={result.tier} "
            f"| status={result.status}"
        )

        print(
            f"confidence={result.confidence:.3f} "
            f"| margin={result.margin:.3f} "
            f"| best={result.best_score:.3f}"
        )

        print(
            "scores:",
            result.scores,
        )

        if result.context_flags:

            print(
                "flags:",
                result.context_flags,
            )

        if result.matches:

            print(
                "top signals:"
            )

            for match in (
                result.matches[:5]
            ):

                print(
                    " ",
                    match[
                        "tier"
                    ],
                    match[
                        "strength"
                    ],
                    f"{match['score']:.2f}",
                    match[
                        "location"
                    ],
                    "::",
                    match[
                        "signal"
                    ],
                )


    print()
    print(
        "-" * 100
    )

    print(
        f"RESULT: "
        f"{passed}/{len(tests)} PASSED"
    )

    print(
        "ENGINE:",
        ENGINE_VERSION,
    )

    print(
        "FINGERPRINT:",
        engine.fingerprint(),
    )

    print(
        "=" * 100
    )


    if passed != len(
        tests
    ):

        raise RuntimeError(
            "Taxonomy V3 self-test failed."
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    taxonomy_statistics()

    self_test()