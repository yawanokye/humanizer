from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from scholarly_humanizer import analyse_scholarly_style
from services.ai_detector import ai_check_report, sentence_ai_signal
from services.calibration import calibration_status, predict_probability
from services.reference_lm import score_reference_lm

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z\[]|\*\*))")
WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z’'-]*\b")
CITATION_RE = re.compile(r"\([^()\n]{0,260}\b(?:19|20)\d{2}[a-z]?\b[^()\n]{0,260}\)", re.I)
PROTECTED_LINE_RE = re.compile(
    r"^(?:#{1,6}\s+.+|CHAPTER\s+(?:\d+|[A-Z]+)(?:\s+.+)?|"
    r"\d+\.\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{1,150}|"
    r"(?:References|Bibliography|Appendix|Appendices)\b.*)$",
    re.I,
)

GENERIC_PATTERNS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (re.compile(r"\bit is important to note(?: that)?\b", re.I), "Generic metadiscourse", 18),
    (re.compile(r"\bit should be noted(?: that)?\b", re.I), "Generic metadiscourse", 18),
    (re.compile(r"\bin today's world\b", re.I), "Broad stock opening", 18),
    (re.compile(r"\bdelve into\b", re.I), "Inflated stock wording", 14),
    (re.compile(r"\bplays? a crucial role\b", re.I), "Generic evaluative phrase", 14),
    (re.compile(r"\bvarious factors\b", re.I), "Vague noun phrase", 12),
    (re.compile(r"\bthis highlights the importance\b", re.I), "Predictable conclusion frame", 16),
    (re.compile(r"\bthis study aims to contribute\b", re.I), "Formulaic study claim", 16),
    (re.compile(r"\bfrom the foregoing\b", re.I), "Formulaic transition", 15),
    (re.compile(r"\bthe above discussion shows\b", re.I), "Formulaic summary", 15),
    (re.compile(r"\btaken together\b", re.I), "Repeated synthesis frame", 9),
    (re.compile(r"\bnot only\b.+?\bbut also\b", re.I), "Balanced template construction", 10),
)

CONNECTOR_RE = re.compile(
    r"^(Moreover|Furthermore|Additionally|In addition|Besides this|Importantly|"
    r"Consequently|Therefore|Thus|Hence|Taken together|Against this background)\b",
    re.I,
)

TRANSITION_WORD_RE = re.compile(
    r"\b(Moreover|Furthermore|Additionally|In addition|Besides this|Importantly|Consequently|Therefore|Thus|Hence|"
    r"Taken together|Against this background|In summary|In conclusion|Overall|Notably|It is worth noting)\b",
    re.I,
)

REPEATED_FRAMES = (
    "the study",
    "this study",
    "the chapter",
    "this chapter",
    "in this context",
    "within this context",
    "taken together",
    "rather than",
)

COMMON_STYLE_TOKENS = {
    "the", "and", "of", "to", "in", "that", "this", "is", "are", "was", "were", "with", "for", "as", "by",
    "it", "on", "from", "at", "an", "be", "or", "which", "can", "may", "also", "such", "these", "those",
    "therefore", "however", "moreover", "furthermore", "additionally", "overall", "significant", "important",
}

CONTENT_STOPWORDS = COMMON_STYLE_TOKENS | {
    "a", "about", "above", "after", "all", "also", "among", "an", "any", "because", "been", "before", "being",
    "between", "both", "but", "during", "each", "has", "have", "having", "into", "its", "more", "most", "not",
    "only", "other", "over", "same", "should", "than", "their", "them", "then", "there", "through", "under", "using",
    "when", "where", "while", "who", "whose", "will", "within", "without", "would",
}

TEMPORAL_VAGUE_RE = re.compile(
    r"\b(recently|currently|nowadays|today(?:'s)?|in recent years|in modern times|contemporary|latest|emerging)\b",
    re.I,
)
CONFIDENCE_RE = re.compile(r"\b(clearly|obviously|undoubtedly|certainly|proves?|guarantees?|always|never|all|none)\b", re.I)
UNSUPPORTED_CLAIM_RE = re.compile(r"\b(studies show|research shows|evidence shows|it has been proven|scholars agree|experts agree)\b", re.I)
IMPLAUSIBLE_PERCENT_RE = re.compile(r"(?<![\w.])(1[0-9]{2,}|[2-9]\d{2,})(?:\.\d+)?%")
P_VALUE_RE = re.compile(r"\bp\s*[<=>]\s*(-?\d+(?:\.\d+)?)", re.I)
CORRELATION_RE = re.compile(r"\b(?:r|rho|β|beta)\s*=\s*(-?\d+(?:\.\d+)?)", re.I)
TYPO_SIGNAL_RE = re.compile(
    r"(\bteh\b|\brecieve\b|\boccured\b|\bseperate\b|\bdefinately\b|\s{2,}|\s+[,.!?;:]|[,.!?;:]{2,}|[a-z][.!?][A-Z])",
    re.I,
)


@dataclass(slots=True)
class SegmentDiagnostic:
    index: int
    paragraph_index: int
    sentence_index: int
    text: str
    start: int
    end: int
    risk: int
    band: str
    reasons: list[str]
    protected: bool = False
    signal_keys: list[str] | None = None
    primary_signal: str | None = None
    section: str = "other"


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _words(text: str) -> list[str]:
    return [word.lower().replace("’", "'") for word in WORD_RE.findall(text or "")]


def _content_words(text: str) -> list[str]:
    return [word for word in _words(text) if word not in CONTENT_STOPWORDS and len(word) > 2]


def _opening(text: str, count: int = 3) -> str:
    words = WORD_RE.findall(text.lower())
    return " ".join(words[:count])


def _band(risk: int) -> str:
    if risk >= 70:
        return "high"
    if risk >= 45:
        return "moderate"
    if risk >= 25:
        return "low"
    return "natural"


def _percent(value: float) -> int:
    return max(0, min(100, round(value)))


def _std_dev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return _std_dev(values) / mean if mean else 0.0


SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}\.?\s+)?(?:"
    r"abstract|executive summary|introduction|background|literature review|conceptual framework|"
    r"materials? and methods?|methodology|methods?|data and methods?|results?|findings?|"
    r"discussion|integrated discussion|results and discussion|discussion and recommendations?|"
    r"conclusion|conclusions|limitations|recommendations?|references|works cited|bibliography|appendix|appendices)\b",
    re.I,
)


def _normalise_section_heading(value: str) -> str:
    clean = re.sub(r"^\d+(?:\.\d+){0,3}\.?\s+", "", str(value or "").strip()).lower()
    if clean.startswith(("abstract", "executive summary")):
        return "abstract"
    if clean.startswith(("materials", "methodology", "method", "data and method")):
        return "methods"
    if clean.startswith(("results and discussion", "discussion", "integrated discussion")):
        return "discussion"
    if clean.startswith(("result", "finding")):
        return "results"
    if clean.startswith(("references", "works cited", "bibliography")):
        return "references"
    if clean.startswith(("appendix", "appendices")):
        return "appendix"
    if clean.startswith(("conclusion", "recommendation", "limitation")):
        return "conclusion"
    if clean.startswith(("introduction", "background", "literature review", "conceptual framework")):
        return "intro_lit"
    return "other"


def _section_markers(text: str) -> list[tuple[int, str]]:
    raw = str(text or "")
    markers: list[tuple[int, str]] = [(0, "other")]
    cursor = 0
    for line in raw.splitlines(keepends=True):
        core = line.rstrip("\r\n").strip()
        if core and len(WORD_RE.findall(core)) <= 16 and SECTION_HEADING_RE.match(core):
            markers.append((cursor, _normalise_section_heading(core)))
        cursor += len(line)
    return markers


def _section_for_offset(markers: list[tuple[int, str]], offset: int) -> str:
    section = "other"
    for start, label in markers:
        if start > offset:
            break
        section = label
    return section


def _section_profile(segments: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[int]] = {key: [] for key in ("abstract", "intro_lit", "methods", "results", "discussion", "conclusion", "other")}
    for item in segments:
        if item.get("protected"):
            continue
        section = str(item.get("section") or "other")
        buckets.setdefault(section, []).append(int(item.get("risk") or 0))
    profile: dict[str, Any] = {}
    means: list[float] = []
    for section, values in buckets.items():
        mean = round(sum(values) / len(values), 2) if values else 0.0
        profile[section] = {
            "segment_count": len(values),
            "mean_signal": mean,
            "elevated_count": sum(1 for value in values if value >= 45),
            "flagged_count": sum(1 for value in values if value >= 25),
        }
        if values:
            means.append(mean)
    profile["score_spread"] = round(max(means) - min(means), 2) if len(means) >= 2 else 0.0
    return profile


def _iter_sentences_with_offsets(text: str) -> list[tuple[int, int, int, int, str]]:
    """Return exact sentence offsets while respecting extracted document lines.

    DOCX/PDF extraction commonly emits one original paragraph per line. The old
    blank-line-only parser could treat a title, author block and abstract as one
    giant sentence. v2.1 treats each non-empty source line as a paragraph unit,
    then splits sentences inside that line without losing offsets.
    """
    raw = str(text or "")
    items: list[tuple[int, int, int, int, str]] = []
    paragraph_index = 0
    cursor = 0
    for line in raw.splitlines(keepends=True):
        core = line.rstrip("\r\n")
        if not core.strip():
            cursor += len(line)
            continue
        leading = len(core) - len(core.lstrip())
        paragraph = core.strip()
        local_cursor = 0
        sentence_index = 0
        pieces = SENTENCE_RE.split(paragraph)
        for piece in pieces:
            if not piece.strip():
                local_cursor += len(piece)
                continue
            local = paragraph.find(piece, local_cursor)
            if local < 0:
                local = local_cursor
            start = cursor + leading + local
            end = start + len(piece)
            items.append((paragraph_index, sentence_index, start, end, piece))
            local_cursor = local + len(piece)
            sentence_index += 1
        paragraph_index += 1
        cursor += len(line)
    if not items and raw.strip():
        stripped_start = len(raw) - len(raw.lstrip())
        items.append((0, 0, stripped_start, len(raw.rstrip()), raw.strip()))
    return items


def _is_structural_segment(clean: str) -> bool:
    value = str(clean or "").strip()
    if not value:
        return True
    if "\t" in value or " | " in value:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)*\.?", value):
        return True
    if PROTECTED_LINE_RE.match(value) or value.startswith("|") or "$$" in value or "```" in value:
        return True
    if re.match(r"^(?:Table|Figure|Fig\.?|Appendix)\s+\d*[A-Za-z]?[.:]?\b", value, re.I):
        return True
    if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z]", value):
        return True
    if re.match(r"^(?:Abstract|Executive Summary|Introduction|Conclusion|Recommendations?|Limitations?|References|Works Cited|Bibliography)$", value, re.I):
        return True
    if re.match(r"^(?:FULL LEGAL NAME|LOCATION|EMAIL ADDRESS|Team member\s+\d+)\b", value, re.I):
        return True
    if re.search(r"\bEmail\s*:\s*\S+@\S+", value, re.I) or re.fullmatch(r"\S+@\S+", value):
        return True
    if len(WORD_RE.findall(value)) <= 20 and not re.search(r"[.!?][\]\)\"'’”]*$", value):
        if re.search(r"\b(?:University|Department|School|College|Institute|Faculty|Centre|Center)\b", value, re.I):
            return True
        # Most title/headline lines in scholarly documents have no sentence stop.
        if not re.search(r"\b(?:is|are|was|were|has|have|can|may|will|should)\b", value, re.I):
            return True
    return False


def _ngram_repetition_score(words: list[str]) -> tuple[int, list[str]]:
    if len(words) < 20:
        return 0, []
    repeated_units = 0
    examples: list[str] = []
    for n in (2, 3, 4):
        grams = [tuple(words[i:i + n]) for i in range(0, len(words) - n + 1)]
        counts = Counter(gram for gram in grams if any(token not in CONTENT_STOPWORDS for token in gram))
        repeats = [(gram, count) for gram, count in counts.items() if count >= 3]
        repeated_units += sum((count - 2) * n for gram, count in repeats)
        for gram, count in repeats[:3]:
            examples.append(f"{' '.join(gram)} ×{count}")
    density = repeated_units / max(1, len(words))
    return _percent(density * 180), examples[:5]


def _syntactic_signature(sentence: str) -> str:
    stripped = sentence.strip()
    if CONNECTOR_RE.match(stripped):
        return "generic-transition opening"
    if re.match(r"^(this|these|it|the study|the chapter)\b", stripped, re.I):
        return "demonstrative or study-subject opening"
    if re.match(r"^[A-Z][a-z]+(?:\s+[a-z]+){0,3}\s+(?:is|are|was|were|has|have|can|may|should)\b", stripped):
        return "noun phrase plus linking verb"
    if re.search(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", stripped, re.I):
        return "passive construction"
    return "other"


def _semantic_density_score(sentences: list[str], paragraphs: list[str]) -> tuple[int, list[str]]:
    sentence_densities: list[float] = []
    for sentence in sentences:
        total = max(1, _word_count(sentence))
        sentence_densities.append(len(_content_words(sentence)) / total)
    paragraph_densities: list[float] = []
    for paragraph in paragraphs:
        total = max(1, _word_count(paragraph))
        paragraph_densities.append(len(_content_words(paragraph)) / total)
    sentence_cv = _cv(sentence_densities)
    paragraph_cv = _cv(paragraph_densities)
    concern = 0
    evidence: list[str] = []
    if len(sentence_densities) >= 6 and sentence_cv < 0.12:
        concern += 55
        evidence.append("Sentence-level information density is unusually even.")
    else:
        concern += max(0, (0.22 - sentence_cv) * 180)
    if len(paragraph_densities) >= 4 and paragraph_cv < 0.10:
        concern += 30
        evidence.append("Paragraph-level information density varies very little.")
    else:
        concern += max(0, (0.18 - paragraph_cv) * 120)
    return _percent(concern), evidence


def _temporal_awareness_score(text: str) -> tuple[int, list[str]]:
    value = str(text or "")
    vague_hits = TEMPORAL_VAGUE_RE.findall(value)
    years = [int(match[:4]) for match in re.findall(r"\b(?:19|20)\d{2}\b", value)]
    current_year = datetime.now(UTC).year
    concern = 0
    evidence: list[str] = []
    if vague_hits:
        concern += min(55, len(vague_hits) * 12)
        evidence.append(f"Vague time references: {min(len(vague_hits), 5)} found.")
    if vague_hits and not years:
        concern += 35
        evidence.append("Recent/current claims appear without a specific year.")
    if any(year > current_year + 1 for year in years):
        concern += 45
        evidence.append("A future year appears in the text. Confirm whether it is intended.")
    if years and max(years) < current_year - 6 and vague_hits:
        concern += 25
        evidence.append("Current-sounding wording is paired with older dated evidence.")
    return _percent(concern), evidence


def _hallucination_pattern_score(text: str, sentences: list[str]) -> tuple[int, list[str]]:
    concern = 0
    evidence: list[str] = []
    impossible_percentages = IMPLAUSIBLE_PERCENT_RE.findall(text)
    if impossible_percentages:
        concern += min(50, len(impossible_percentages) * 25)
        evidence.append("Percentages above 100% need verification.")
    bad_p_values = [value for value in P_VALUE_RE.findall(text) if float(value) < 0 or float(value) > 1]
    if bad_p_values:
        concern += min(45, len(bad_p_values) * 20)
        evidence.append("One or more p-values fall outside the 0 to 1 range.")
    bad_correlations = [value for value in CORRELATION_RE.findall(text) if abs(float(value)) > 1]
    if bad_correlations:
        concern += min(45, len(bad_correlations) * 20)
        evidence.append("One or more correlation or standardised coefficient values exceed the expected range.")
    unsupported = 0
    for sentence in sentences:
        if UNSUPPORTED_CLAIM_RE.search(sentence) and not CITATION_RE.search(sentence):
            unsupported += 1
    if unsupported:
        concern += min(45, unsupported * 15)
        evidence.append(f"Broad evidence claims without nearby citations: {unsupported}.")
    confidence_hits = len(CONFIDENCE_RE.findall(text))
    if confidence_hits >= 4:
        concern += min(30, (confidence_hits - 3) * 6)
        evidence.append("High-certainty language is repeated and may need evidence checks.")
    return _percent(concern), evidence


def _build_category_concerns(text: str, segments: list[dict[str, Any]], global_report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = str(text or "")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    sentences = [str(item.get("text") or "").strip() for item in segments if not item.get("protected") and str(item.get("text") or "").strip()]
    words = _words(raw)
    word_total = max(1, len(words))

    lexical_diversity = float(global_report.get("lexical_diversity_msttr") or 0.0)
    sentence_cv = float(global_report.get("sentence_length_cv") or 0.0)
    paragraph_cv = float(global_report.get("paragraph_length_cv") or 0.0)
    target_lexical = 0.64 if str(global_report.get("perplexity_level")) == "high" else 0.56
    target_sentence_cv = 0.50 if str(global_report.get("burstiness_level")) == "high" else 0.38
    target_paragraph_cv = 0.42 if str(global_report.get("burstiness_level")) == "high" else 0.30
    generic_hits = int(global_report.get("generic_phrase_hits") or 0)
    repeated_openings = int(global_report.get("repeated_sentence_openings") or 0) + int(global_report.get("repeated_paragraph_openings") or 0)
    frame_density = int(global_report.get("repeated_frame_density") or 0)
    connector_hits = int(global_report.get("generic_connector_hits") or 0)

    low_lexical_gap = max(0.0, target_lexical - lexical_diversity)
    perplexity = _percent(low_lexical_gap * 130 + generic_hits * 5 + repeated_openings * 3 + frame_density * 2)

    sentence_gap = max(0.0, target_sentence_cv - sentence_cv)
    paragraph_gap = max(0.0, target_paragraph_cv - paragraph_cv)
    burstiness = _percent(sentence_gap * 120 + paragraph_gap * 80)
    if global_report.get("uniform_sentence_rhythm"):
        burstiness = _percent(burstiness + 18)
    if global_report.get("uniform_paragraph_rhythm"):
        burstiness = _percent(burstiness + 12)

    common_token_ratio = sum(1 for word in words if word in COMMON_STYLE_TOKENS) / word_total
    top_counts = Counter(words).most_common(12)
    top_token_share = sum(count for _word, count in top_counts) / word_total
    token_distribution = _percent(max(0, common_token_ratio - 0.42) * 170 + max(0, top_token_share - 0.50) * 150 + generic_hits * 2)

    ngram_score, ngram_examples = _ngram_repetition_score(words)
    semantic_score, semantic_evidence = _semantic_density_score(sentences, paragraphs)
    signatures = [_syntactic_signature(sentence) for sentence in sentences]
    signature_counts = Counter(signatures)
    repeated_syntax_share = sum(count for sig, count in signature_counts.items() if sig != "other" and count >= 3) / max(1, len(signatures))
    syntax_score = _percent(repeated_syntax_share * 100 + repeated_openings * 7)

    typo_signals = TYPO_SIGNAL_RE.findall(raw)
    polished_surface = 0
    typo_evidence: list[str] = []
    if word_total >= 250 and not typo_signals:
        polished_surface = 55 if word_total < 800 else 75
        typo_evidence.append("Long passage has no visible typo or punctuation irregularity. Do not add errors, simply review manually.")
    elif typo_signals:
        polished_surface = max(0, 25 - min(25, len(typo_signals) * 4))
        typo_evidence.append(f"Visible typo or punctuation irregularity signals found: {min(len(typo_signals), 5)}.")

    transition_score = _percent((connector_hits * 8) + max(0, len(TRANSITION_WORD_RE.findall(raw)) - 4) * 5)
    vocabulary_score = _percent(low_lexical_gap * 170 + max(0, top_token_share - 0.50) * 120)
    temporal_score, temporal_evidence = _temporal_awareness_score(raw)
    hallucination_score, hallucination_evidence = _hallucination_pattern_score(raw, sentences)

    categories = [
        {
            "group": "Primary Statistical Metrics",
            "description": "Local proxy indicators for predictability, rhythm and over-reliance on high-frequency wording.",
            "metrics": [
                {
                    "key": "perplexity_proxy",
                    "label": "Perplexity proxy",
                    "percentage": perplexity,
                    "evidence": [
                        f"Lexical diversity MSTTR: {lexical_diversity:.3f}.",
                        f"Generic phrase hits: {generic_hits}.",
                    ],
                },
                {
                    "key": "burstiness",
                    "label": "Burstiness concern",
                    "percentage": burstiness,
                    "evidence": [
                        f"Sentence length CV: {sentence_cv:.3f}.",
                        f"Paragraph length CV: {paragraph_cv:.3f}.",
                    ],
                },
                {
                    "key": "token_probability_distribution_proxy",
                    "label": "Token probability distribution proxy",
                    "percentage": token_distribution,
                    "evidence": [
                        f"Common style-token share: {common_token_ratio:.1%}.",
                        f"Top-token share: {top_token_share:.1%}.",
                    ],
                },
            ],
        },
        {
            "group": "Linguistic and N-gram Patterns",
            "description": "Repetition, uniform density and repeated grammatical framing.",
            "metrics": [
                {
                    "key": "ngram_frequency",
                    "label": "N-gram frequency",
                    "percentage": ngram_score,
                    "evidence": ngram_examples or ["No high-frequency repeated 2 to 4 word phrases crossed the local threshold."],
                },
                {
                    "key": "uniform_semantic_density",
                    "label": "Uniform semantic density",
                    "percentage": semantic_score,
                    "evidence": semantic_evidence or ["Information density varies within the normal local threshold."],
                },
                {
                    "key": "repetitive_syntactic_structures",
                    "label": "Repetitive syntactic structures",
                    "percentage": syntax_score,
                    "evidence": [f"Repeated opening/frame count: {repeated_openings}.", f"Dominant structure: {signature_counts.most_common(1)[0][0] if signature_counts else 'none'}."],
                },
            ],
        },
        {
            "group": "Vocabulary and Stylistic Markers",
            "description": "Surface polish, transition habits and vocabulary range.",
            "metrics": [
                {
                    "key": "absence_of_typographical_errors",
                    "label": "Absence of typographical errors",
                    "percentage": polished_surface,
                    "evidence": typo_evidence or ["Text is short or contains enough surface variation that this signal is not active."],
                },
                {
                    "key": "systemic_transitions",
                    "label": "Over-reliance on systemic transitions",
                    "percentage": transition_score,
                    "evidence": [f"Transition hits: {len(TRANSITION_WORD_RE.findall(raw))}.", f"Line-opening generic connectors: {connector_hits}."],
                },
                {
                    "key": "low_vocabulary_diversity",
                    "label": "Low vocabulary diversity",
                    "percentage": vocabulary_score,
                    "evidence": [f"Lexical diversity MSTTR: {lexical_diversity:.3f}.", f"Top-token share: {top_token_share:.1%}."],
                },
            ],
        },
        {
            "group": "Semantic and Logic Constraints",
            "description": "Local checks for vague time framing, impossible values and unsupported certainty. These are not fact checks.",
            "metrics": [
                {
                    "key": "lack_of_temporal_awareness",
                    "label": "Lack of temporal awareness",
                    "percentage": temporal_score,
                    "evidence": temporal_evidence or ["No strong local temporal-awareness signal was found."],
                },
                {
                    "key": "hallucination_patterns",
                    "label": "Hallucination-pattern risk",
                    "percentage": hallucination_score,
                    "evidence": hallucination_evidence or ["No local impossible-statistic or unsupported-certainty pattern crossed the threshold."],
                },
            ],
        },
    ]

    for category in categories:
        values = [int(metric["percentage"]) for metric in category["metrics"]]
        category["percentage"] = _percent(sum(values) / max(1, len(values)))
    return categories


SIGNAL_LABELS = {
    "A": "Perplexity / predictability",
    "B": "Burstiness / rhythm",
    "C": "Hedge density",
    "D": "Structural tells",
    "E": "Specificity",
    "F": "Transitions",
    "G": "Punctuation",
    "H": "Voice / register",
    "I": "Rhetorical scaffolding",
}


def _infer_segment_signal_keys(sentence: str, reasons: list[str]) -> list[str]:
    """Map sentence evidence to the A-I forensic families for coloured diagnostics."""
    joined = " | ".join(reasons).lower()
    keys: list[str] = []
    mapping = (
        ("A", ("predictable", "vocabulary", "generic evaluative", "inflated stock", "generic metadiscourse")),
        ("B", ("rhythm", "sentence-length", "very long sentence", "very short sentence", "cadence")),
        ("C", ("hedge", "softening", "uncertainty")),
        ("D", ("structural", "three-part", "paragraph opening", "sentence opening", "balanced template")),
        ("E", ("specificity", "obscured actor", "unsupported", "broad evidence claim")),
        ("F", ("transition", "connector", "formulaic summary", "synthesis frame")),
        ("G", ("em-dash", "semicolon", "punctuation", "colon pattern", "clause density")),
        ("H", ("voice", "register", "professional voice")),
        ("I", ("rhetorical", "scaffolding", "announcement", "pivot", "repeated frame", "parallel question", "thesis/landing")),
    )
    for key, markers in mapping:
        if any(marker in joined for marker in markers):
            keys.append(key)
    # Pattern-level fallback where the reason text is intentionally concise.
    if not keys:
        lower = sentence.lower()
        if re.search(r"\b(?:furthermore|moreover|additionally|therefore|taken together)\b", lower):
            keys.append("F")
        if "—" in sentence or ";" in sentence:
            keys.append("G")
        if re.search(r"\bnot only\b.+\bbut also\b", lower):
            keys.extend(["D", "I"])
    return list(dict.fromkeys(keys))


def build_signal_coloured_html(text: str, segments: list[dict[str, Any]] | None = None) -> str:
    """Colour sentence spans by their diagnosed A-I signal families.

    The sentence background stays neutral. Category colours appear as an underline/left
    accent plus compact badges, so multi-signal sentences remain readable.
    """
    raw = str(text or "")
    diagnostics = segments or analyse_segments(raw)
    pieces: list[str] = []
    cursor = 0
    for segment in diagnostics:
        start, end = int(segment["start"]), int(segment["end"])
        if start > cursor:
            pieces.append(html.escape(raw[cursor:start]).replace("\n", "<br>"))
        escaped = html.escape(raw[start:end]).replace("\n", "<br>")
        keys = list(segment.get("signal_keys") or [])
        if segment.get("protected"):
            pieces.append(f'<span class="signal-text protected-text">{escaped}</span>')
        elif keys:
            primary = str(segment.get("primary_signal") or keys[0])
            badges = "".join(
                f'<span class="signal-badge signal-{html.escape(key)}" title="{html.escape(SIGNAL_LABELS.get(key, key))}">{html.escape(key)}</span>'
                for key in keys
            )
            reasons = html.escape(" • ".join(segment.get("reasons") or []), quote=True)
            pieces.append(
                f'<span class="signal-text signal-text-{html.escape(primary)}" data-reasons="{reasons}" '
                f'data-signals="{html.escape(",".join(keys))}" tabindex="0">{escaped}{badges}</span>'
            )
        else:
            pieces.append(escaped)
        cursor = end
    if cursor < len(raw):
        pieces.append(html.escape(raw[cursor:]).replace("\n", "<br>"))
    return "".join(pieces)


def _statistical_fingerprint(categories: list[dict[str, Any]]) -> tuple[int, dict[str, int]]:
    """Build the v2.1 statistical fingerprint from continuous local metrics.

    This remains a black-box-free statistical layer. The perplexity and token-probability
    fields are explicitly proxies unless a reference LM is configured in a future build.
    """
    values: dict[str, int] = {}
    for group in categories:
        for metric in group.get("metrics", []):
            values[str(metric.get("key"))] = int(metric.get("percentage") or 0)
    weights = {
        "perplexity_proxy": 0.18,
        "burstiness": 0.20,
        "token_probability_distribution_proxy": 0.17,
        "ngram_frequency": 0.15,
        "uniform_semantic_density": 0.10,
        "repetitive_syntactic_structures": 0.10,
        "systemic_transitions": 0.05,
        "low_vocabulary_diversity": 0.05,
    }
    score = round(sum(values.get(key, 0) * weight for key, weight in weights.items()))
    return max(0, min(100, score)), {key: values.get(key, 0) for key in weights}


def _signal_level_from_percentage(score: int) -> str:
    if score < 20:
        return "Minimal AI-style signal"
    if score < 40:
        return "Low AI-style signal"
    if score < 60:
        return "Moderate AI-style signal"
    if score < 80:
        return "Elevated AI-style signal"
    return "Strong AI-style signal"


def analyse_segments(text: str) -> list[dict[str, Any]]:
    raw = str(text or "")
    positions = _iter_sentences_with_offsets(raw)
    section_markers = _section_markers(raw)
    openings = Counter(_opening(item[4]) for item in positions if _opening(item[4]))
    paragraph_openings: Counter[str] = Counter()
    for paragraph_index, sentence_index, *_rest, sentence in positions:
        if sentence_index == 0:
            paragraph_openings[_opening(sentence)] += 1

    frame_counts = {frame: len(re.findall(rf"\b{re.escape(frame)}\b", raw, re.I)) for frame in REPEATED_FRAMES}
    connector_counts = Counter()
    for *_pos, sentence in positions:
        connector = CONNECTOR_RE.match(sentence.strip())
        if connector:
            connector_counts[connector.group(1).casefold()] += 1

    results: list[SegmentDiagnostic] = []
    for global_index, (paragraph_index, sentence_index, start, end, sentence) in enumerate(positions):
        clean = sentence.strip()
        section = _section_for_offset(section_markers, start)
        protected = _is_structural_segment(clean) or section in {"references", "appendix"}
        reasons: list[str] = []
        risk = 0
        word_count = _word_count(clean)

        if protected:
            results.append(SegmentDiagnostic(
                index=global_index,
                paragraph_index=paragraph_index,
                sentence_index=sentence_index,
                text=sentence,
                start=start,
                end=end,
                risk=0,
                band="protected",
                reasons=["Protected academic structure"],
                protected=True,
                signal_keys=[],
                primary_signal=None,
                section=section,
            ))
            continue

        ai_risk, ai_reasons = sentence_ai_signal(clean, academic=True)
        if ai_risk:
            risk += ai_risk
            reasons.extend(ai_reasons)

        # Retain a smaller portion of the earlier scholarly-style diagnostics as supporting evidence.
        for pattern, reason, weight in GENERIC_PATTERNS:
            hits = len(pattern.findall(clean))
            if hits:
                risk += min(max(4, weight // 3) + (hits - 1) * 2, 12)
                reasons.append(reason)

        opening = _opening(clean)
        if opening and openings[opening] > 2:
            risk += min(18, 6 * (openings[opening] - 2))
            reasons.append(f'Repeated sentence opening: “{opening}”')
        if sentence_index == 0 and opening and paragraph_openings[opening] > 1:
            risk += min(18, 8 * (paragraph_openings[opening] - 1))
            reasons.append(f'Repeated paragraph opening: “{opening}”')

        connector_match = CONNECTOR_RE.match(clean)
        if connector_match and connector_counts[connector_match.group(1).casefold()] > 2:
            risk += 12
            reasons.append("Repeated generic connector")

        if word_count > 65:
            risk += 8
            reasons.append("Very long sentence adds to structural regularity review")
        elif word_count < 5 and word_count > 0:
            risk += 4
            reasons.append("Very short sentence reviewed for inverted burstiness")

        if clean.count(";") >= 2:
            risk += 12
            reasons.append("Heavy semicolon chaining")
        if clean.count(",") >= 7:
            risk += 10
            reasons.append("High clause density")
        if len(re.findall(r"\b(?:this|these|it)\b", clean, re.I)) >= 5:
            risk += 10
            reasons.append("Dense pronoun framing")
        if len(CONFIDENCE_RE.findall(clean)) >= 2:
            risk += 10
            reasons.append("Inflated or over-certain wording")
        if UNSUPPORTED_CLAIM_RE.search(clean) and not CITATION_RE.search(clean):
            risk += 14
            reasons.append("Broad evidence claim without nearby citation")
        if TEMPORAL_VAGUE_RE.search(clean) and not re.search(r"\b(?:19|20)\d{2}\b", clean):
            risk += 10
            reasons.append("Vague temporal framing without a specific date")
        if IMPLAUSIBLE_PERCENT_RE.search(clean):
            risk += 18
            reasons.append("Implausible percentage needs verification")

        for frame, count in frame_counts.items():
            if count > max(2, _word_count(raw) // 500) and re.search(rf"\b{re.escape(frame)}\b", clean, re.I):
                risk += min(12, 3 + count)
                reasons.append(f'Repeated frame: “{frame}”')

        citations = CITATION_RE.findall(clean)
        if len(citations) >= 3:
            risk += 8
            reasons.append("Dense citation cluster")

        risk = max(0, min(100, risk))
        if not reasons and risk == 0:
            reasons.append("No major formulaic-style issue detected")
        signal_keys = _infer_segment_signal_keys(clean, reasons)
        results.append(SegmentDiagnostic(
            index=global_index,
            paragraph_index=paragraph_index,
            sentence_index=sentence_index,
            text=sentence,
            start=start,
            end=end,
            risk=risk,
            band=_band(risk),
            reasons=reasons,
            signal_keys=signal_keys,
            primary_signal=signal_keys[0] if signal_keys else None,
            section=section,
        ))
    return [asdict(item) for item in results]


def build_highlighted_html(text: str, segments: list[dict[str, Any]] | None = None) -> str:
    raw = str(text or "")
    diagnostics = segments or analyse_segments(raw)
    pieces: list[str] = []
    cursor = 0
    for segment in diagnostics:
        start, end = int(segment["start"]), int(segment["end"])
        if start > cursor:
            pieces.append(html.escape(raw[cursor:start]).replace("\n", "<br>"))
        escaped = html.escape(raw[start:end]).replace("\n", "<br>")
        reasons = html.escape(" • ".join(segment.get("reasons") or []), quote=True)
        pieces.append(
            f'<span class="risk-segment risk-{segment["band"]}" '
            f'data-risk="{segment["risk"]}" data-reasons="{reasons}" '
            f'tabindex="0">{escaped}</span>'
        )
        cursor = end
    if cursor < len(raw):
        pieces.append(html.escape(raw[cursor:]).replace("\n", "<br>"))
    return "".join(pieces)


def _calibration_feature_vector(
    detector: dict[str, Any],
    statistical_fingerprint: int,
    statistical_components: dict[str, int],
    reference_lm: dict[str, Any],
    section_profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    segment_count = max(1, int(detector.get("segment_count") or 0))
    flagged_ratio = int(detector.get("flagged_segment_count") or 0) / segment_count
    features = {
        "forensic_pct": float(detector.get("category_signal_percentage") or 0),
        "statistical_pct": float(statistical_fingerprint),
        "paragraph_pct": float(detector.get("segment_signal_percentage") or 0),
        "regularity_pct": float(detector.get("consistency_signal_percentage") or 0),
        "perplexity_proxy": float(statistical_components.get("perplexity_proxy") or 0),
        "burstiness": float(statistical_components.get("burstiness") or 0),
        "token_distribution_proxy": float(statistical_components.get("token_probability_distribution_proxy") or 0),
        "ngram_frequency": float(statistical_components.get("ngram_frequency") or 0),
        "uniform_semantic_density": float(statistical_components.get("uniform_semantic_density") or 0),
        "repetitive_syntactic_structures": float(statistical_components.get("repetitive_syntactic_structures") or 0),
        "systemic_transitions": float(statistical_components.get("systemic_transitions") or 0),
        "low_vocabulary_diversity": float(statistical_components.get("low_vocabulary_diversity") or 0),
        "segment_p90": float(detector.get("segment_p90") or 0),
        "flagged_segment_ratio": round(flagged_ratio * 100, 4),
        "reference_perplexity": float(reference_lm.get("perplexity") or 0),
        "reference_surprisal_mean": float(reference_lm.get("surprisal_mean") or 0),
        "reference_surprisal_std": float(reference_lm.get("surprisal_std") or 0),
        "reference_low_surprisal_share": float(reference_lm.get("low_surprisal_share") or 0) * 100,
        "reference_curvature_abs_mean": float(reference_lm.get("curvature_abs_mean") or 0),
        "reference_curvature_std": float(reference_lm.get("curvature_std") or 0),
        "reference_curvature_regular_share": float(reference_lm.get("curvature_regular_share") or 0) * 100,
    }
    sections = section_profile or {}
    for key in ("abstract", "intro_lit", "methods", "results", "discussion", "conclusion", "other"):
        features[f"section_{key}_mean"] = float((sections.get(key) or {}).get("mean_signal") or 0)
    features["section_score_spread"] = float(sections.get("score_spread") or 0)
    return features


def _decision_status(ai_signal: int, layers: list[int], *, calibrated: bool) -> tuple[str, str]:
    """Return a public review status without exposing the detector operating mode."""
    spread = max(layers, default=0) - min(layers, default=0)
    if 40 <= ai_signal <= 60:
        return "Indeterminate", "The score falls inside the abstention zone; review the highlighted evidence rather than forcing a human/AI conclusion."
    if spread >= 50 and 20 <= ai_signal <= 80:
        return "Indeterminate", "Independent signal families disagree strongly; review the highlighted sections individually."
    return "Result available", "Review the highlighted sentence and signal evidence alongside the headline score."


def dashboard_report(text: str, *, use_calibrator: bool = True, include_private: bool = False) -> dict[str, Any]:
    global_report = analyse_scholarly_style(text)
    segments = analyse_segments(text)

    # Naturalness remains an internal rewrite-quality metric. It is deliberately
    # not presented as the inverse of AI likelihood. The public complementary
    # score is Human-like Style = 100 - AI Signal.
    naturalness = max(0, min(100, int(global_report.get("naturalness_score", 0))))
    style_concern = 100 - naturalness  # legacy API field only

    category_concerns = _build_category_concerns(text, segments, global_report)
    statistical_fingerprint, statistical_components = _statistical_fingerprint(category_concerns)
    detector = ai_check_report(text, global_report=global_report, academic=True)
    reference_lm = score_reference_lm(text)
    section_profile = _section_profile(segments)

    forensic_pct = int(detector.get("category_signal_percentage", 0))
    paragraph_pct = int(detector.get("segment_signal_percentage", 0))
    regularity_pct = int(detector.get("consistency_signal_percentage", 0))

    # v2.4 adds a probability-curvature family when a true local reference LM is
    # available. The raw curvature proxy is not treated as DetectGPT/Fast-DetectGPT.
    # It becomes useful primarily after calibration against labelled benchmark data.
    curvature_available = bool(reference_lm.get("scored")) and "curvature_regular_share" in reference_lm
    curvature_pct = _percent(float(reference_lm.get("curvature_regular_share") or 0.0) * 100) if curvature_available else 0
    layers = [forensic_pct, statistical_fingerprint, paragraph_pct, regularity_pct] + ([curvature_pct] if curvature_available else [])

    if curvature_available:
        base_ensemble = round(
            forensic_pct * 0.20
            + statistical_fingerprint * 0.30
            + paragraph_pct * 0.25
            + regularity_pct * 0.10
            + curvature_pct * 0.15
        )
        fallback_weights = {"forensic": 0.20, "statistical": 0.30, "segments": 0.25, "document_consistency": 0.10, "probability_curvature": 0.15}
    else:
        base_ensemble = round(
            forensic_pct * 0.25
            + statistical_fingerprint * 0.35
            + paragraph_pct * 0.30
            + regularity_pct * 0.10
        )
        fallback_weights = {"forensic": 0.25, "statistical": 0.35, "segments": 0.30, "document_consistency": 0.10}
    corroborating_layers = sum(1 for value in layers if value >= 35)
    corroboration_bonus = 8 if corroborating_layers >= 3 else 4 if corroborating_layers >= 2 else 0
    fallback_signal = max(0, min(100, base_ensemble + corroboration_bonus))

    calibration = calibration_status()
    features = _calibration_feature_vector(detector, statistical_fingerprint, statistical_components, reference_lm, section_profile)
    meta_prediction = predict_probability(features) if use_calibrator and calibration.get("trained") else None
    if meta_prediction:
        ai_signal = int(meta_prediction["percentage"])
        score_source = "calibrated_meta_classifier"
    else:
        ai_signal = fallback_signal
        score_source = "transparent_four_layer_ensemble"

    decision_status, decision_reason = _decision_status(ai_signal, layers, calibrated=bool(meta_prediction))
    if decision_status == "Indeterminate" and detector.get("confidence") == "High":
        detector["confidence"] = "Medium"

    detector["ai_detection_percentage"] = ai_signal
    detector["verdict"] = _signal_level_from_percentage(ai_signal)
    detector["signal_level"] = detector["verdict"]
    detector["statistical_fingerprint_percentage"] = statistical_fingerprint
    detector["statistical_components"] = statistical_components
    detector["section_profile"] = section_profile
    detector["probability_curvature_percentage"] = curvature_pct if curvature_available else None
    detector["decision_status"] = decision_status
    detector["decision_reason"] = decision_reason
    if include_private:
        detector["reference_lm"] = reference_lm
        detector["calibration"] = calibration
        detector["calibration_prediction"] = meta_prediction
        detector["calibration_features"] = features
        detector["score_source"] = score_source
        detector["fallback_ensemble_percentage"] = fallback_signal
        detector["corroboration_bonus"] = corroboration_bonus
        detector["corroborating_layers"] = corroborating_layers
        detector["composite_weights"] = fallback_weights
    if not include_private:
        # Keep detector operating mode, ensemble weights and calibration mechanics private.
        detector.pop("composite_weights", None)
        detector.pop("score_source", None)
        detector.pop("fallback_ensemble_percentage", None)
        detector.pop("corroboration_bonus", None)
        detector.pop("corroborating_layers", None)
        detector.pop("calibration_prediction", None)
        detector.pop("calibration_features", None)
        detector.pop("reference_lm", None)
        detector.pop("calibration", None)

    detector["calibration_notes"] = [
        "AI Signal is a style-screening index, not the percentage of words written by AI.",
        "Human-context evidence changes confidence only and does not erase AI-style evidence.",
        "Tables, form rows, headings and table punctuation are excluded from prose-style scoring where possible.",
        "Different AI-writing detectors can disagree substantially on the same scholarly passage; use this as a diagnostic aid, not proof of authorship.",
    ]
    human_like_style = 100 - ai_signal
    high_count = sum(1 for item in segments if item["band"] == "high")
    moderate_count = sum(1 for item in segments if item["band"] == "moderate")
    active_signal_categories = sum(1 for signal in detector.get("signals", []) if int(signal.get("score", 0)) > 0)
    elevated_signal_categories = sum(1 for signal in detector.get("signals", []) if int(signal.get("score", 0)) >= 2)
    evidence_items = sum(len(signal.get("evidence", []) or []) for signal in detector.get("signals", []))
    flagged_sentence_count = int(detector.get("flagged_sentence_count", 0))

    result = {
        "ai_detection_percentage": ai_signal,
        "human_like_style_percentage": human_like_style,
        "ai_detector": detector,
        "ai_verdict": detector["verdict"],
        "ai_signal_level": detector["verdict"],
        "ai_confidence": detector["confidence"],
        "ai_score": detector["overall_score"],
        "ai_score_max": detector["max_score"],
        "ai_signal_breakdown": detector["signals"],
        "decision_status": decision_status,
        "decision_reason": decision_reason,
        "section_profile": section_profile,
        # Internal/backward-compatible rewrite-quality field. The dashboard does not
        # use this as the complement of AI Signal.
        "naturalness_percentage": naturalness,
        "style_concern_percentage": style_concern,
        "style_concern_categories": category_concerns,
        "statistical_fingerprint_percentage": statistical_fingerprint,
        "statistical_fingerprint_components": statistical_components,
        "high_risk_segments": high_count,
        "moderate_risk_segments": moderate_count,
        "active_signal_categories": active_signal_categories,
        "elevated_signal_categories": elevated_signal_categories,
        "signal_evidence_items": evidence_items,
        "flagged_sentence_count": flagged_sentence_count,
        "prose_segment_count": int(detector.get("segment_count", 0)),
        "flagged_prose_segments": int(detector.get("flagged_segment_count", 0)),
        "elevated_prose_segments": int(detector.get("elevated_segment_count", 0)),
        "strong_prose_segments": int(detector.get("strong_segment_count", 0)),
        "segments": segments,
        "highlighted_html": build_highlighted_html(text, segments),
        "signal_coloured_html": build_signal_coloured_html(text, segments),
        "signal_colour_legend": SIGNAL_LABELS,
        "metrics": global_report,
        "disclaimer": (
            "AI Signal and Human-like Style are complementary style indicators: Human-like Style = 100 - AI Signal. "
            "AI Signal does not prove authorship. Different detectors may disagree substantially on the same text. "
            "The detector abstains when evidence is ambiguous or internally inconsistent."
        ),
        "detector_variability_notice": detector.get("detector_variability_notice", ""),
    }
    if include_private:
        result.update({
            "score_source": score_source,
            "calibration": calibration,
            "calibration_features": features,
            "reference_lm": reference_lm,
            "probability_curvature_percentage": curvature_pct if curvature_available else None,
            "detector_layers": {
                "forensic": forensic_pct,
                "statistical": statistical_fingerprint,
                "paragraph": paragraph_pct,
                "document_regularity": regularity_pct,
                "probability_curvature": curvature_pct if curvature_available else None,
            },
            "fallback_weights": fallback_weights,
        })
    return result

