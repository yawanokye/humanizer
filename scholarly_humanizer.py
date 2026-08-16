from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any


# This module improves scholarly naturalness without introducing deliberate
# mistakes, changing evidence, or attempting to evade detection systems.
# It is deterministic so the same text receives the same protected local edits.

_LEGACY_ARTIFACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s+That is, it matters\.\s*", re.I), " "),
    (re.compile(r"\s+That matters\.\s*", re.I), " "),
    (
        re.compile(
            r"\s+This qualification matters (?:because|insofar as) it keeps the argument tied to the evidence rather than to an unsupported general claim\.\s*",
            re.I,
        ),
        " ",
    ),
)

# Only low-risk replacements are made locally. Substantive restructuring is left
# to the preservation-gated model pass because local synonym rotation can damage
# disciplinary meaning and the author's voice.
_SAFE_PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\binsofar as of\b", re.I), "because of"),
    (re.compile(r"\binsofar as\b", re.I), "because"),
    (re.compile(r"\bdue to the fact that\b", re.I), "because"),
    (re.compile(r"\bin order to\b", re.I), "to"),
    (re.compile(r"\bfor the purpose of\b", re.I), "to"),
    (re.compile(r"\bwith regard to\b", re.I), "regarding"),
    (re.compile(r"\bwith respect to\b", re.I), "regarding"),
    (re.compile(r"\bthe present investigation\b", re.I), "the study"),
    (re.compile(r"\bthe present study\b", re.I), "the study"),
    (re.compile(r"\bthe current study\b", re.I), "the study"),
    (re.compile(r"\bthe results obtained\b", re.I), "the results"),
    (re.compile(r"\bas an illustration\b", re.I), "for example"),
    (re.compile(r"\bexemplifies how\b", re.I), "shows how"),
    (re.compile(r"\bnon[-\s]trivial function\b", re.I), "important role"),
    (re.compile(r"\bit is important to note that\b", re.I), ""),
    (re.compile(r"\bit should be noted that\b", re.I), ""),
    (re.compile(r"\bin today's world\b", re.I), "in the present context"),
    (re.compile(r"\bdelve into\b", re.I), "examine"),
    (re.compile(r"\bplays a crucial role\b", re.I), "is important"),
    (re.compile(r"\bhas the ability to\b", re.I), "can"),
    (re.compile(r"\bis able to\b", re.I), "can"),
    (re.compile(r"\ba large number of\b", re.I), "many"),
    (re.compile(r"\bit is against this background that\b", re.I), "against this background,"),
    (re.compile(r"\bthe reason for this is because\b", re.I), "this is because"),
    (re.compile(r"\bof particular importance is the fact that\b", re.I), "importantly,"),
    (re.compile(r"\bthis study aims to contribute to\b", re.I), "the study contributes to"),
    (re.compile(r"\bthe study is important in\b", re.I), "the study is useful for"),
    (re.compile(r"\bit can therefore be said that\b", re.I), "therefore,"),
    (re.compile(r"\bthe above discussion shows that\b", re.I), "the discussion shows that"),
    (re.compile(r"\bthe foregoing discussion shows that\b", re.I), "the discussion shows that"),
    (re.compile(r"\bfrom the foregoing,?\s*\b", re.I), ""),
    (re.compile(r"\bthis highlights the importance of\b", re.I), "this points to the importance of"),
    (re.compile(r"\bthis means that\b", re.I), "this suggests that"),
    (re.compile(r"\bit is clear that\b", re.I), ""),
    (re.compile(r"\bit is evident that\b", re.I), ""),
    (re.compile(r"\bit is worth noting that\b", re.I), ""),
    (re.compile(r"\bit is worth mentioning that\b", re.I), ""),
    (re.compile(r"\bin the realm of\b", re.I), "in"),
    (re.compile(r"\ba myriad of\b", re.I), "many"),
    (re.compile(r"\ba plethora of\b", re.I), "many"),
    (re.compile(r"\butili[sz]e\b", re.I), "use"),
    (re.compile(r"\bpivotal role\b", re.I), "central role"),
    (re.compile(r"\bmultifaceted\b", re.I), "complex"),
    (re.compile(r"\bthis (?:highlights|underscores|demonstrates) the importance of\b", re.I), "this shows the importance of"),
)

_GENERIC_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bin today's world\b", re.I),
    re.compile(r"\bit is important to note\b", re.I),
    re.compile(r"\bit should be noted\b", re.I),
    re.compile(r"\bdelve into\b", re.I),
    re.compile(r"\bplays a crucial role\b", re.I),
    re.compile(r"\bvarious factors\b", re.I),
    re.compile(r"\bthis highlights the importance\b", re.I),
    re.compile(r"\bthis study aims to contribute\b", re.I),
    re.compile(r"\bthe research problem is that\b", re.I),
    re.compile(r"\bthat matters\b", re.I),
    re.compile(r"\bthis qualification matters\b", re.I),
    re.compile(r"\bit is against this background that\b", re.I),
    re.compile(r"\bit can therefore be said that\b", re.I),
    re.compile(r"\bfrom the foregoing\b", re.I),
    re.compile(r"\bthe above discussion shows\b", re.I),
    re.compile(r"\bthe foregoing discussion\b", re.I),
    re.compile(r"\bneedless to say\b", re.I),
    re.compile(r"\bthe study is important in\b", re.I),
    re.compile(r"\bthe study can show how\b", re.I),
    re.compile(r"\bit is clear that\b", re.I),
    re.compile(r"\bit is evident that\b", re.I),
    re.compile(r"\bit is worth noting\b", re.I),
    re.compile(r"\bin the realm of\b", re.I),
    re.compile(r"\ba myriad of\b", re.I),
    re.compile(r"\ba plethora of\b", re.I),
    re.compile(r"\butili[sz]e\b", re.I),
    re.compile(r"\bmultifaceted\b", re.I),
    re.compile(r"\bthis (?:highlights|underscores|demonstrates) the importance\b", re.I),
)

_GENERIC_CONNECTOR_RE = re.compile(
    r"^(?P<connector>Moreover|Furthermore|Additionally|In addition|Besides this|It is also worth noting that|"
    r"Importantly|Consequently|Therefore|Thus|Hence|Taken together|Against this background)\s*,?\s+",
    re.I,
)

_REPEATED_FRAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthe study\b", re.I),
    re.compile(r"\bthis study\b", re.I),
    re.compile(r"\bthe chapter\b", re.I),
    re.compile(r"\bthis chapter\b", re.I),
    re.compile(r"\bin this context\b", re.I),
    re.compile(r"\bwithin this context\b", re.I),
    re.compile(r"\btaken together\b", re.I),
    re.compile(r"\bnot only\b", re.I),
    re.compile(r"\brather than\b", re.I),
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z\[]|\*\*))")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:%|\b)")
_PLACEHOLDER_RE = re.compile(r"\[[^\]\n]+\]")
_URL_RE = re.compile(r"https?://\S+|\bdoi:\s*\S+|\b10\.\d{4,9}/\S+", re.I)
_CITATION_BLOCK_RE = re.compile(r"\([^()\n]{0,260}\b(?:19|20)\d{2}[a-z]?\b[^()\n]{0,260}\)", re.I)
_HEADING_LINE_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s+.+|CHAPTER\s+(?:\d+|[A-Z]+)(?:\s+.+)?|\d+\.\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{1,150})\s*$", re.I)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\d+[.)]\s+[^\n]+$")
_DISPLAY_EQUATION_RE = re.compile(r"\$\$.*?\$\$", re.S)
_TABLE_LINE_RE = re.compile(r"(?m)^\s*\|[^\n]+\|\s*$")
_SECTION_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|CHAPTER\s+(?:\d+|[A-Z]+)(?:\s+.+)?|\d+\.\d+(?:\.\d+){0,3}\s+[A-Z][^\n]{1,150})$",
    re.I,
)
_REFERENCE_HEADING_RE = re.compile(r"^(?:#{1,6}\s*)?(?:References|Bibliography|Source Use Audit|Appendix|Appendices)\b", re.I)


def _normalise_variation_level(value: str, *, default: str = "high") -> str:
    level = str(value or default).strip().lower()
    return level if level in {"moderate", "high"} else default


def humanizer_variation_profile() -> dict[str, Any]:
    """Return the controlled variation targets used by both drafting workflows.

    ``Perplexity`` is treated here as context-sensitive lexical and syntactic
    variety, not random or obscure wording. ``Burstiness`` is measured through
    purposeful variation in sentence and paragraph rhythm.
    """
    perplexity = _normalise_variation_level(
        os.getenv("PROJECTREADY_HUMANIZER_PERPLEXITY_LEVEL", "high")
    )
    burstiness = _normalise_variation_level(
        os.getenv("PROJECTREADY_HUMANIZER_BURSTINESS_LEVEL", "high")
    )
    high = perplexity == "high" or burstiness == "high"
    return {
        "perplexity_level": perplexity,
        "burstiness_level": burstiness,
        "lexical_diversity_target": 0.64 if perplexity == "high" else 0.56,
        "sentence_length_cv_target": 0.50 if burstiness == "high" else 0.38,
        "paragraph_length_cv_target": 0.42 if burstiness == "high" else 0.30,
        "short_sentence_ratio_target": 0.10 if burstiness == "high" else 0.06,
        "long_sentence_ratio_target": 0.14 if burstiness == "high" else 0.09,
        "model_word_change_limit": float(
            os.getenv("PROJECTREADY_HUMANIZER_MAX_WORD_CHANGE_RATIO", "0.18" if high else "0.10")
            or (0.18 if high else 0.10)
        ),
    }


def variation_targets_met(report: dict[str, Any], profile: dict[str, Any] | None = None) -> bool:
    targets = profile or humanizer_variation_profile()
    return (
        float(report.get("lexical_diversity_msttr") or 0.0) >= float(targets["lexical_diversity_target"])
        and float(report.get("sentence_length_cv") or 0.0) >= float(targets["sentence_length_cv_target"])
        and float(report.get("paragraph_length_cv") or 0.0) >= float(targets["paragraph_length_cv_target"])
        and float(report.get("short_sentence_ratio") or 0.0) >= float(targets["short_sentence_ratio_target"])
        and float(report.get("long_sentence_ratio") or 0.0) >= float(targets["long_sentence_ratio_target"])
    )


def scholarly_humanizer_prompt_rules() -> list[str]:
    """Prompt rules shared by chapter generation and chapter strengthening."""
    return [
        "Write in a natural, disciplined scholarly voice rather than a promotional, formulaic or template-like voice.",
        "Preserve the author's substantive voice. Improve clarity and flow without making every paragraph sound as though it was written by the same generic editor.",
        "Use high controlled perplexity: vary vocabulary, clause structure and rhetorical framing through precise context-specific choices. Do not create variety through rare synonyms, technical-term substitution or ornamental wording.",
        "Use high controlled burstiness: mix concise emphasis sentences, medium analytical sentences and occasional longer synthesis sentences where the argument calls for them. Avoid a uniform cadence, but do not manufacture fragments or overlong sentences.",
        "Vary paragraph length and internal movement according to function. A definition, comparison, qualification, empirical synthesis and transition should not all have the same shape.",
        "Avoid repeating distinctive content words, sentence openings or grammatical frames within a short span when an equally precise natural construction is available.",
        "Use direct subjects and active verbs where they improve clarity, but retain passive constructions when the disciplinary convention or focus on process makes them appropriate.",
        "Vary sentence length and paragraph density according to argumentative function. Do not force every paragraph into the same number of sentences or the same claim-evidence-conclusion template.",
        "Let paragraph movement follow the evidence. Some paragraphs may define, compare, qualify, critique, interpret or connect; do not append a generic concluding sentence merely to make a paragraph appear complete.",
        "Use transitions only when they express the actual logical relationship, such as contrast, cause, condition, sequence, implication or limitation. Avoid mechanically rotating 'moreover', 'furthermore', 'additionally' and similar connectors.",
        "Avoid repeated paragraph openings, repeated restatement of the study title, excessive 'the study', excessive 'this chapter', inflated vocabulary and predictable sentence frames.",
        "Reduce unnecessary nominalisation when a clear verb is more natural, but preserve technical terms and discipline-specific concepts.",
        "Avoid overusing balanced triples, 'not only ... but also', 'rather than', 'this means that', 'this suggests that', 'taken together', and repeated contrast formulas such as 'on one hand ... on the other hand'.",
        "Do not over-explain obvious links. State the analytical point once, support it, and move the argument forward.",
        "Synthesis should organise sources around a claim, tension, pattern or gap. Do not produce an author-by-author catalogue unless chronology or study comparison genuinely requires it.",
        "Place citations where they naturally support the relevant claim. Avoid citation dumping at the end of long paragraphs and avoid attaching the same cluster mechanically to several consecutive sentences.",
        "Preserve the strength of claims. Do not replace cautious terms such as 'suggests', 'indicates', 'may' or 'is associated with' with stronger causal language unless the evidence warrants it.",
        "Use formal British English, clear discipline-specific wording and moderate lexical variety. Prefer familiar precise words over rare synonyms.",
        "Preserve all verified facts, statistics, dates, citations, references, equations, tables, headings, objectives, questions, hypotheses and bracketed action placeholders.",
        "Keep academic prose free from drafting commentary. Place every unresolved confirmation, missing source, missing evidence or student instruction on its own [ACTION REQUIRED: ...] line immediately after the affected paragraph or sentence.",
        "Do not add deliberate errors, typographical mistakes, spelling variation, false hesitations, casual fillers or artificial drafting artefacts. When the surface is too polished, improve rhythm and specificity instead of inserting mistakes.",
        "Do not discuss AI detection or claim that the text is human-authored. The purpose of the pass is natural scholarly quality, evidential integrity and alignment with the researcher's supplied voice.",
    ]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text or ""))


def _std_dev(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _sentence_opening(sentence: str) -> str:
    cleaned = re.sub(r"^[\s\"'“”‘’([{]+", "", sentence or "")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned.lower())
    return " ".join(words[:3])


def _paragraph_opening(paragraph: str) -> str:
    cleaned = re.sub(r"^[\s\"'“”‘’([{]+", "", paragraph or "")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", cleaned.lower())
    return " ".join(words[:3])


def _moving_standardised_type_token_ratio(text: str, *, window: int = 50) -> float:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", str(text or "").lower())
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    scores: list[float] = []
    step = max(10, window // 2)
    for start in range(0, len(tokens) - window + 1, step):
        sample = tokens[start:start + window]
        scores.append(len(set(sample)) / window)
    return sum(scores) / len(scores) if scores else 0.0


def _coefficient_of_variation(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (_std_dev(values) / mean) if mean else 0.0


def analyse_scholarly_style(text: str) -> dict[str, Any]:
    """Return an explainable diagnostic for natural scholarly prose."""
    value = str(text or "")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip() and not _is_protected_block(part)]
    sentences: list[str] = []
    for paragraph in paragraphs:
        sentences.extend([item.strip() for item in _SENTENCE_RE.split(paragraph) if item.strip()])

    sentence_lengths = [_word_count(sentence) for sentence in sentences if _word_count(sentence)]
    paragraph_lengths = [_word_count(paragraph) for paragraph in paragraphs if _word_count(paragraph)]
    lexical_diversity = _moving_standardised_type_token_ratio(value)
    sentence_length_cv = _coefficient_of_variation(sentence_lengths)
    paragraph_length_cv = _coefficient_of_variation(paragraph_lengths)
    short_sentence_ratio = (sum(1 for length in sentence_lengths if 5 <= length <= 11) / len(sentence_lengths)) if sentence_lengths else 0.0
    long_sentence_ratio = (sum(1 for length in sentence_lengths if 30 <= length <= 52) / len(sentence_lengths)) if sentence_lengths else 0.0
    variation_profile = humanizer_variation_profile()
    sentence_openings = [_sentence_opening(sentence) for sentence in sentences]
    paragraph_openings = [_paragraph_opening(paragraph) for paragraph in paragraphs]
    sentence_opening_counts = Counter(opening for opening in sentence_openings if opening)
    paragraph_opening_counts = Counter(opening for opening in paragraph_openings if opening)
    repeated_sentence_openings = sum(max(0, count - 2) for count in sentence_opening_counts.values())
    repeated_paragraph_openings = sum(max(0, count - 1) for count in paragraph_opening_counts.values())
    generic_hits = sum(len(pattern.findall(value)) for pattern in _GENERIC_PHRASES)
    connector_hits = len(re.findall(r"(?im)^\s*(?:Moreover|Furthermore|Additionally|In addition|Taken together)\s*,", value))
    long_sentences = sum(1 for length in sentence_lengths if length > 45)
    overloaded_sentences = sum(1 for length in sentence_lengths if length > 65)
    very_short_sentences = sum(1 for length in sentence_lengths if 0 < length < 5)
    uniform_sentence_rhythm = len(sentence_lengths) >= 6 and _std_dev(sentence_lengths) < 5
    uniform_paragraph_rhythm = len(paragraph_lengths) >= 4 and _std_dev(paragraph_lengths) < 18

    repeated_frames: dict[str, int] = {}
    word_total = max(1, _word_count(value))
    for pattern in _REPEATED_FRAME_PATTERNS:
        label = pattern.pattern.replace("\\b", "")
        count = len(pattern.findall(value))
        if count:
            repeated_frames[label] = count
    frame_density = sum(max(0, count - max(2, word_total // 500)) for count in repeated_frames.values())

    score = 100
    score -= min(24, generic_hits * 4)
    score -= min(14, repeated_sentence_openings * 2)
    score -= min(14, repeated_paragraph_openings * 3)
    score -= min(10, max(0, connector_hits - 2) * 2)
    score -= min(14, long_sentences)
    score -= min(12, overloaded_sentences * 2)
    score -= min(8, very_short_sentences * 2)
    score -= min(12, frame_density)
    if uniform_sentence_rhythm:
        score -= 7
    if uniform_paragraph_rhythm:
        score -= 5
    if lexical_diversity < float(variation_profile["lexical_diversity_target"]):
        score -= min(10, round((float(variation_profile["lexical_diversity_target"]) - lexical_diversity) * 50))
    if sentence_length_cv < float(variation_profile["sentence_length_cv_target"]):
        score -= min(10, round((float(variation_profile["sentence_length_cv_target"]) - sentence_length_cv) * 20))
    if paragraph_length_cv < float(variation_profile["paragraph_length_cv_target"]):
        score -= min(7, round((float(variation_profile["paragraph_length_cv_target"]) - paragraph_length_cv) * 15))
    if short_sentence_ratio < float(variation_profile["short_sentence_ratio_target"]):
        score -= 4
    if long_sentence_ratio < float(variation_profile["long_sentence_ratio_target"]):
        score -= 4

    return {
        "score": max(0, min(100, score)),
        "naturalness_score": max(0, min(100, score)),
        "word_count": _word_count(value),
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "perplexity_level": variation_profile["perplexity_level"],
        "burstiness_level": variation_profile["burstiness_level"],
        "lexical_diversity_msttr": round(lexical_diversity, 3),
        "sentence_length_cv": round(sentence_length_cv, 3),
        "paragraph_length_cv": round(paragraph_length_cv, 3),
        "short_sentence_ratio": round(short_sentence_ratio, 3),
        "long_sentence_ratio": round(long_sentence_ratio, 3),
        "variation_targets_met": variation_targets_met({
            "lexical_diversity_msttr": lexical_diversity,
            "sentence_length_cv": sentence_length_cv,
            "paragraph_length_cv": paragraph_length_cv,
            "short_sentence_ratio": short_sentence_ratio,
            "long_sentence_ratio": long_sentence_ratio,
        }, variation_profile),
        "sentence_length_std_dev": round(_std_dev(sentence_lengths), 2),
        "paragraph_length_std_dev": round(_std_dev(paragraph_lengths), 2),
        "generic_phrase_hits": generic_hits,
        "repeated_sentence_openings": repeated_sentence_openings,
        "repeated_paragraph_openings": repeated_paragraph_openings,
        "generic_connector_hits": connector_hits,
        "long_sentence_count": long_sentences,
        "overloaded_sentence_count": overloaded_sentences,
        "very_short_sentence_count": very_short_sentences,
        "repeated_frame_density": frame_density,
        "repeated_frames": repeated_frames,
        "uniform_sentence_rhythm": uniform_sentence_rhythm,
        "uniform_paragraph_rhythm": uniform_paragraph_rhythm,
    }


def _is_protected_block(block: str) -> bool:
    value = str(block or "").strip()
    if not value:
        return True
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return True
    if lines[0].startswith("#") or re.fullmatch(r"CHAPTER\s+(?:\d+|[A-Z]+)", lines[0], re.I):
        return True
    if "```" in value or "$$" in value:
        return True
    if any(line.startswith("|") for line in lines) or any(re.match(r"^\|?\s*:?-{3,}", line) for line in lines):
        return True
    if all(re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", line) for line in lines):
        return True
    if len(lines) == 1 and len(lines[0].split()) <= 14 and lines[0].isupper():
        return True
    return False


def _apply_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _replace_preserving_case(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    return pattern.sub(lambda match: _apply_case(match.group(0), replacement), text)




def _repair_sentence_initial_caps(text: str) -> str:
    def fix(match: re.Match[str]) -> str:
        prefix, letter = match.group(1), match.group(2)
        return prefix + letter.upper()
    return re.sub(r"(^|[.!?]\s+)([a-z])", fix, text.strip())

def _split_long_semicolon_sentences(paragraph: str) -> str:
    sentences = _SENTENCE_RE.split(paragraph)
    revised: list[str] = []
    for sentence in sentences:
        if _word_count(sentence) <= 45 or ";" not in sentence:
            revised.append(sentence)
            continue
        parts = [part.strip() for part in sentence.split(";") if part.strip()]
        if len(parts) < 2 or any(_word_count(part) < 7 for part in parts):
            revised.append(sentence)
            continue
        for part in parts:
            clean = part.rstrip(".!?")
            if clean and clean[:1].islower():
                clean = clean[:1].upper() + clean[1:]
            revised.append(clean + ".")
    return " ".join(item.strip() for item in revised if item.strip())


def _soften_repeated_study_openings(paragraph: str) -> str:
    """Reduce repeated local openings without guessing new evidence."""
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(paragraph) if sentence.strip()]
    if len(sentences) < 3:
        return paragraph
    revised: list[str] = []
    study_openings = 0
    chapter_openings = 0
    for sentence in sentences:
        if re.match(r"^(This study|The study)\b", sentence):
            study_openings += 1
            if study_openings == 2:
                sentence = re.sub(r"^(This study|The study)\b", "The analysis", sentence, count=1)
            elif study_openings >= 3:
                sentence = re.sub(r"^(This study|The study)\b", "It", sentence, count=1)
        elif re.match(r"^(This chapter|The chapter)\b", sentence):
            chapter_openings += 1
            if chapter_openings >= 2:
                sentence = re.sub(r"^(This chapter|The chapter)\b", "The section", sentence, count=1)
        revised.append(sentence)
    return " ".join(revised)


def _remove_repeated_sentence_connectors(paragraph: str, connector_seen: dict[str, int], *, max_uses: int = 2) -> str:
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(paragraph) if sentence.strip()]
    revised: list[str] = []
    for sentence in sentences:
        match = _GENERIC_CONNECTOR_RE.match(sentence)
        if match:
            key = match.group("connector").casefold()
            connector_seen[key] = connector_seen.get(key, 0) + 1
            # Keep the first two uses in a chapter. Later uses are usually clearer
            # without a generic connector than with a mechanically substituted one.
            if connector_seen[key] > max_uses:
                sentence = sentence[match.end():].lstrip()
                if sentence[:1].islower():
                    sentence = sentence[:1].upper() + sentence[1:]
        revised.append(sentence)
    return " ".join(revised)


def _split_long_compound_sentences(paragraph: str) -> str:
    """Split only long independent-clause joins where the second clause has an explicit subject."""
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(paragraph) if sentence.strip()]
    revised: list[str] = []
    subject = r"(?:the|this|these|those|it|they|we|researchers|students|institutions|organisations|organizations|authors|results|findings|evidence|analysis|study)"
    splitter = re.compile(rf",\s+(and|but)\s+(?={subject}\b)", re.I)
    for sentence in sentences:
        if _word_count(sentence) < 38:
            revised.append(sentence)
            continue
        match = splitter.search(sentence)
        if not match:
            revised.append(sentence)
            continue
        left = sentence[:match.start()].rstrip(" ,")
        right = sentence[match.end():].strip()
        conjunction = match.group(1).lower()
        if _word_count(left) < 12 or _word_count(right) < 9:
            revised.append(sentence)
            continue
        if right[:1].islower():
            right = right[:1].upper() + right[1:]
        # Preserve contrast when the original used 'but'; plain coordination can
        # stand as a separate sentence without a mechanical connector.
        if conjunction == "but":
            right = "But " + right
        revised.extend([left.rstrip(".!?") + ".", right])
    return " ".join(revised)


def _refine_paragraph(paragraph: str, connector_seen: dict[str, int], *, level: str = "balanced") -> str:
    """Refine one paragraph with progressively stronger, meaning-safe edits."""
    value = paragraph.strip()
    for pattern, replacement in _LEGACY_ARTIFACT_PATTERNS:
        value = pattern.sub(replacement, value)
    for pattern, replacement in _SAFE_PHRASE_REPLACEMENTS:
        value = _replace_preserving_case(value, pattern, replacement)

    if level in {"balanced", "deep"}:
        value = _remove_repeated_sentence_connectors(
            value, connector_seen, max_uses=0 if level == "deep" else 1
        )
        value = _soften_repeated_study_openings(value)
        value = _split_long_semicolon_sentences(value)
        if level == "deep":
            value = _split_long_compound_sentences(value)

    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", value)
    value = re.sub(r",\s*,", ",", value)
    value = _repair_sentence_initial_caps(value)
    return value.strip()


def _signature(text: str) -> dict[str, list[str]]:
    value = str(text or "")
    return {
        "headings": _HEADING_LINE_RE.findall(value),
        "years": _YEAR_RE.findall(value),
        "numbers": _NUMBER_RE.findall(value),
        "placeholders": _PLACEHOLDER_RE.findall(value),
        "urls": _URL_RE.findall(value),
        "citation_blocks": _CITATION_BLOCK_RE.findall(value),
        "numbered_items": _NUMBERED_ITEM_RE.findall(value),
        "display_equations": _DISPLAY_EQUATION_RE.findall(value),
        "table_lines": _TABLE_LINE_RE.findall(value),
    }


def validate_humanizer_preservation(original: str, candidate: str, *, max_word_change_ratio: float = 0.06) -> tuple[bool, list[str]]:
    """Check that a style-only pass preserved core academic content."""
    reasons: list[str] = []
    before = _signature(original)
    after = _signature(candidate)
    for key in ("headings", "years", "numbers", "placeholders", "urls", "citation_blocks", "numbered_items", "display_equations", "table_lines"):
        if before[key] != after[key]:
            reasons.append(f"{key} changed")

    original_words = max(1, _word_count(original))
    candidate_words = _word_count(candidate)
    ratio = abs(candidate_words - original_words) / original_words
    if ratio > max_word_change_ratio:
        reasons.append(f"word count changed by {ratio:.1%}")
    return not reasons, reasons


def split_scholarly_sections(text: str) -> list[dict[str, Any]]:
    """Split a chapter into heading-led sections without changing content order.

    The result supports preservation-gated, section-batched model refinement. Numbered
    objectives such as ``1. Examine...`` are not treated as headings because the pattern
    requires a chapter-style number such as ``1.2``.
    """
    value = str(text or "")
    if not value.strip():
        return []
    lines = value.splitlines(keepends=True)
    sections: list[dict[str, Any]] = []
    current: list[str] = []
    heading = ""

    def flush() -> None:
        nonlocal current, heading
        if not current:
            return
        section_text = "".join(current).strip()
        if section_text:
            sections.append({
                "heading": heading,
                "text": section_text,
                "protected": bool(_REFERENCE_HEADING_RE.match(heading.strip())) if heading else False,
                "word_count": _word_count(section_text),
            })
        current = []

    for line in lines:
        stripped = line.strip()
        if stripped and _SECTION_HEADING_RE.match(stripped):
            flush()
            heading = stripped
            current = [line]
        else:
            current.append(line)
    flush()
    return sections or [{"heading": "", "text": value.strip(), "protected": False, "word_count": _word_count(value)}]


def build_humanizer_batches(text: str, *, max_words: int = 2600) -> list[dict[str, Any]]:
    """Build manageable section batches for model refinement without compressing long chapters."""
    max_words = max(700, int(max_words or 2600))
    sections = split_scholarly_sections(text)
    batches: list[dict[str, Any]] = []
    current: list[str] = []
    current_words = 0
    current_protected = False

    def flush() -> None:
        nonlocal current, current_words, current_protected
        if not current:
            return
        batch_text = "\n\n".join(part.strip() for part in current if part.strip()).strip()
        if batch_text:
            batches.append({
                "text": batch_text,
                "protected": current_protected,
                "word_count": _word_count(batch_text),
                "diagnostic": analyse_scholarly_style(batch_text),
            })
        current = []
        current_words = 0
        current_protected = False

    for section in sections:
        section_text = str(section.get("text") or "").strip()
        words = int(section.get("word_count") or _word_count(section_text))
        protected = bool(section.get("protected"))
        if protected:
            flush()
            current = [section_text]
            current_words = words
            current_protected = True
            flush()
            continue

        if current and current_words + words > max_words:
            flush()
        current.append(section_text)
        current_words += words
        current_protected = False
    flush()
    return batches


def humanize_scholarly_text(text: str, mode: str = "balanced") -> tuple[str, dict[str, Any]]:
    """Improve scholarly naturalness with preservation-gated local editing.

    Engine 1 is deliberately conservative. It creates progressively stronger
    rule-based candidates, verifies academic evidence preservation, scores each
    candidate for naturalness, and keeps only the best non-degrading version.
    """
    original = str(text or "")
    normalised_mode = str(mode or "balanced").strip().lower()
    before_report = analyse_scholarly_style(original)
    before_score = int(before_report.get("naturalness_score", 0))

    if normalised_mode in {"off", "none", "disabled", "0", "false"} or not original.strip():
        report = dict(before_report)
        report.update({
            "mode": "off", "engine": "engine1", "label": "Engine 1, Local rewrite",
            "applied": False, "preservation_passed": True, "preservation_issues": [],
            "score_before": before_score, "score_after": before_score, "naturalness_gain": 0,
        })
        return original, report

    allowed_levels = {
        "light": ["light"],
        "balanced": ["light", "balanced"],
        "deep": ["light", "balanced", "deep"],
    }.get(normalised_mode, ["light", "balanced"])

    original_words = max(1, _word_count(original))
    local_change_limit = 0.60 if original_words < 120 else max(0.12, min(0.45, 70 / original_words))
    candidates: list[tuple[int, str, str, dict[str, Any]]] = []
    preservation_failures: list[str] = []

    for level in allowed_levels:
        parts = re.split(r"(\n\s*\n)", original)
        connector_seen: dict[str, int] = {}
        output: list[str] = []
        reference_tail = False
        for part in parts:
            if not part or re.fullmatch(r"\n\s*\n", part):
                output.append(part)
                continue
            stripped = part.strip()
            if re.match(r"^#{1,6}\s*(?:References|Bibliography|Source Use Audit|Appendix|Appendices)\b", stripped, re.I):
                reference_tail = True
            if reference_tail or _is_protected_block(part):
                output.append(part)
                continue
            output.append(_refine_paragraph(part, connector_seen, level=level))

        candidate = "".join(output)
        candidate = re.sub(r"[ \t]+\n", "\n", candidate)
        candidate = re.sub(r"\n{3,}", "\n\n", candidate).strip()
        valid, issues = validate_humanizer_preservation(
            original, candidate, max_word_change_ratio=local_change_limit
        )
        if not valid:
            preservation_failures.extend(issues)
            continue
        candidate_report = analyse_scholarly_style(candidate)
        score = int(candidate_report.get("naturalness_score", 0))
        candidates.append((score, level, candidate, candidate_report))

    if not candidates:
        report = dict(before_report)
        report.update({
            "mode": normalised_mode, "engine": "engine1", "label": "Engine 1, Local rewrite",
            "applied": False, "preservation_passed": False,
            "preservation_issues": sorted(set(preservation_failures)),
            "score_before": before_score, "score_after": before_score, "naturalness_gain": 0,
            "reason": "No local candidate passed the preservation checks.",
        })
        return original, report

    # Prefer the highest naturalness score. For a tie, prefer the lighter pass.
    level_rank = {"light": 0, "balanced": 1, "deep": 2}
    candidates.sort(key=lambda item: (-item[0], level_rank.get(item[1], 9)))
    best_score, best_level, best_text, best_report = candidates[0]

    if best_score < before_score:
        report = dict(before_report)
        report.update({
            "mode": normalised_mode, "engine": "engine1", "label": "Engine 1, Local rewrite",
            "applied": False, "preservation_passed": True, "preservation_issues": [],
            "score_before": before_score, "score_after": before_score, "naturalness_gain": 0,
            "reason": "Candidate was rejected because it reduced the naturalness score.",
        })
        return original, report

    report = dict(best_report)
    report.update({
        "mode": normalised_mode, "applied_level": best_level, "engine": "engine1",
        "label": "Engine 1, Local rewrite", "applied": best_text != original,
        "preservation_passed": True, "preservation_issues": [],
        "score_before": before_score, "score_after": best_score,
        "naturalness_gain": best_score - before_score,
    })
    return best_text, report
