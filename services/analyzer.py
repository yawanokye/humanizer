from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from scholarly_humanizer import analyse_scholarly_style

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


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


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


def _iter_sentences_with_offsets(text: str) -> list[tuple[int, int, int, int, str]]:
    """Return paragraph/sentence positions without losing exact source offsets."""
    items: list[tuple[int, int, int, int, str]] = []
    cursor = 0
    paragraph_index = 0
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        paragraph = match.group(0)
        paragraph_start = match.start()
        sentence_cursor = 0
        sentence_index = 0
        pieces = SENTENCE_RE.split(paragraph)
        for piece in pieces:
            if not piece.strip():
                sentence_cursor += len(piece)
                continue
            local = paragraph.find(piece, sentence_cursor)
            if local < 0:
                local = sentence_cursor
            start = paragraph_start + local
            end = start + len(piece)
            items.append((paragraph_index, sentence_index, start, end, piece))
            sentence_cursor = local + len(piece)
            sentence_index += 1
        paragraph_index += 1
        cursor = match.end()
    if not items and text.strip():
        stripped_start = len(text) - len(text.lstrip())
        items.append((0, 0, stripped_start, len(text.rstrip()), text.strip()))
    return items


def analyse_segments(text: str) -> list[dict[str, Any]]:
    raw = str(text or "")
    positions = _iter_sentences_with_offsets(raw)
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
        protected = bool(PROTECTED_LINE_RE.match(clean)) or clean.startswith("|") or "$$" in clean or "```" in clean
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
            ))
            continue

        for pattern, reason, weight in GENERIC_PATTERNS:
            hits = len(pattern.findall(clean))
            if hits:
                risk += min(weight + (hits - 1) * 4, weight + 8)
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
            risk += 30
            reasons.append("Overloaded sentence above 65 words")
        elif word_count > 45:
            risk += 18
            reasons.append("Long sentence above 45 words")
        elif word_count < 5 and word_count > 0:
            risk += 12
            reasons.append("Very short sentence may be abrupt")

        if clean.count(";") >= 2:
            risk += 12
            reasons.append("Heavy semicolon chaining")
        if clean.count(",") >= 7:
            risk += 10
            reasons.append("High clause density")
        if len(re.findall(r"\b(?:this|these|it)\b", clean, re.I)) >= 5:
            risk += 10
            reasons.append("Dense pronoun framing")
        if len(re.findall(r"\b(?:clearly|obviously|undoubtedly|significantly|remarkably)\b", clean, re.I)) >= 2:
            risk += 10
            reasons.append("Inflated evaluative wording")

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


def dashboard_report(text: str) -> dict[str, Any]:
    global_report = analyse_scholarly_style(text)
    segments = analyse_segments(text)
    weighted_total = sum(max(1, _word_count(item["text"])) * item["risk"] for item in segments if not item["protected"])
    weighted_words = sum(max(1, _word_count(item["text"])) for item in segments if not item["protected"])
    segment_concern = round(weighted_total / weighted_words) if weighted_words else 0
    global_concern = 100 - int(global_report.get("naturalness_score", 0))
    concern_percentage = max(0, min(100, round(global_concern * 0.55 + segment_concern * 0.45)))
    high_count = sum(1 for item in segments if item["band"] == "high")
    moderate_count = sum(1 for item in segments if item["band"] == "moderate")
    return {
        "naturalness_percentage": 100 - concern_percentage,
        "style_concern_percentage": concern_percentage,
        "high_risk_segments": high_count,
        "moderate_risk_segments": moderate_count,
        "segments": segments,
        "highlighted_html": build_highlighted_html(text, segments),
        "metrics": global_report,
        "disclaimer": (
            "This is an explainable scholarly-style diagnostic. It does not determine authorship "
            "and it is not an AI-detection probability."
        ),
    }
