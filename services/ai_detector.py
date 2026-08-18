from __future__ import annotations

import re
from collections import Counter
from typing import Any

WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z’'-]*\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z\[]|\*\*))")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
CITATION_RE = re.compile(r"\([^()\n]{0,260}\b(?:19|20)\d{2}[a-z]?\b[^()\n]{0,260}\)", re.I)

AI_VOCAB = (
    "delve", "leverage", "utilize", "robust", "comprehensive", "streamline", "foster", "facilitate",
    "pivotal", "nuanced", "notable", "notably", "enduring", "garner", "multifaceted",
    "in the realm of", "the landscape of", "a myriad of", "a plethora of", "it is worth noting",
    "it is important to note",
)
HEDGE_PATTERNS = (
    r"\boften\b", r"\bgenerally\b", r"\btypically\b", r"\bin many cases\b", r"\bit can be argued\b",
    r"\bit is important to note(?: that)?\b", r"\bit is worth mentioning\b", r"\bone might consider\b",
    r"\bmay result in\b", r"\bcan often lead to\b", r"\btends? to\b",
)
TRANSITION_STRONG = (
    r"^\s*Furthermore,", r"^\s*Moreover,", r"^\s*Additionally,", r"\bIt is clear that\b",
    r"\bThis (?:highlights|underscores|demonstrates) the importance of\b", r"\bAs previously mentioned\b",
    r"\bIn addition to the above\b", r"\bIt goes without saying\b", r"\bNeedless to say\b",
)
TRANSITION_MODERATE = (
    r"^\s*Therefore,", r"\bturns out\b", r"\bit turns out that\b", r"\bThe standard fix is\b",
    r"\bThe common approach is\b", r"\bSimple enough on paper\b", r"\bThe rule I use\b",
    r"\bThe key insight\b", r"\bThe approach here\b", r"\bThe other thing I(?:'|’)d say\b",
    r"\bThe pattern is almost always the same\b",
)
SETUP_RE = re.compile(
    r"\b(?:What I didn(?:'|’)t expect was|What surprised me was|The thing I realized was|What it didn(?:'|’)t have was|"
    r"What ended up working was|What changed everything was|What finally clicked was|What made the difference was)\b",
    re.I,
)
DIMINISHMENT_RE = re.compile(r"\b(?:not just\b.{0,80}\bbut\b|not\b.{1,80},\s*it(?:'|’)s\b|not\b.{1,80}\bbut\b)", re.I)
MORE_THAN_RE = re.compile(r"\bmore\s+[A-Za-z'-]+\s+than\s+[A-Za-z'-]+\b", re.I)
BINARY_RE = re.compile(r"\beither\b.{1,100}\bor\b", re.I)
BINARY_CHOICE_RE = re.compile(r"\b(?:choice|choose|choosing|option|options|trade-?off)\b.{0,45}\bbetween\b.{1,90}\band\b", re.I)
TURNS_OUT_RE = re.compile(r"\b(?:turns out|it turns out that)\b", re.I)
ACTUAL_WORK_RE = re.compile(r"\bis the (?:actual|real) work\b", re.I)
THESIS_FIRST_RE = re.compile(r"^(?:.{0,70}\bis the (?:easy|hard) part\b|.{0,80}\bhas become increasingly important\b)", re.I)
PASSIVE_RE = re.compile(r"\b(?:it has been found that|research suggests|it is believed that|it is widely accepted that)\b", re.I)
UNIVERSAL_RE = re.compile(r"\b(?:teams|developers|organizations|institutions|researchers|students) (?:often|frequently|generally|typically)\b", re.I)
GENERIC_EXAMPLE_RE = re.compile(r"\b(?:Netflix|Amazon|Stripe)\b")
INFORMAL_MARKER_RE = re.compile(r"\b(?:lmk|btw|imo|fyi)\b|~\d+%|<\d+\s*(?:min|mins|minutes)", re.I)
SELF_CORRECTION_RE = re.compile(r"\b(?:actually|wait|that(?:'|’)s not quite right|I mean|rather,)\b", re.I)
FIRST_PERSON_RE = re.compile(r"\b(?:I|me|my|mine|we|our|ours)\b", re.I)
SECOND_PERSON_RE = re.compile(r"\b(?:you|your|yours)\b", re.I)
RHETORICAL_Q_RE = re.compile(r"\?")
TEMPORAL_VAGUE_RE = re.compile(r"\b(?:recently|currently|nowadays|today(?:'s)?|in recent years|latest|emerging)\b", re.I)

PATTERN_ANNOUNCEMENT_RE = re.compile(
    r"\b(?:the pattern is (?:almost always|always|usually|typically|clear|simple|the same)|"
    r"the rule is|the key point is|the key insight is|what (?:I|we|it)[^.!?]{0,60} was)\b",
    re.I,
)
PARTICIPIAL_REFRAME_RE = re.compile(r"^(?:Laid out|Arranged|Seen|Viewed|Framed|Read|Presented)\s+(?:this way|that way|in this way|in these terms|in a [^,]{1,35})?,?", re.I)
COMPOSED_PARENT_RE = re.compile(r"\((?:which|something)\s+I\s+(?:choose to read as|take as|am choosing to interpret as|interpret as)[^)]{0,90}\)", re.I)
BALANCED_PAREN_RE = re.compile(r"\([^()]{1,70},\s*but\s+[^()]{1,70}\)\s+(?:or|and)\s+\([^()]{1,70},\s*but\s+[^()]{1,70}\)", re.I)
APOSTROPHE_CLOSER_RE = re.compile(r"^(?:That(?:'|’)s|This is|It is|The work|The point|The lesson)\b", re.I)
REASON_CLAUSE_RE = re.compile(r"\b(?:because|when)\b", re.I)
PARALLEL_QUESTION_LIST_RE = re.compile(r"\b(?:what|why|how)\b[^,.]{1,60},\s*\b(?:what|why|how)\b[^,.]{1,60},\s*\b(?:what|why|how)\b[^,.]{1,60}(?:,\s*\b(?:what|why|how)\b[^,.]{1,60})?", re.I)

SEVERITY_WEIGHT = {"weak": 1, "moderate": 2, "strong": 3}

# Reliability weights derived from the supplied AI-check evaluation rubric.
# Countable lexical/punctuation signals carry more weight than subjective
# rhetoric/register judgments. The weighted maximum is 25.05 before rescaling.
CATEGORY_WEIGHTS = {
    "A": 1.0, "B": 1.0, "C": 0.75, "D": 0.85, "E": 0.85,
    "F": 1.0, "G": 1.0, "H": 0.75, "I": 0.60,
}
WEIGHTED_MAX = sum(3 * weight for weight in CATEGORY_WEIGHTS.values())


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(text or "")]


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    return parts if parts else ([text.strip()] if text and text.strip() else [])


def _is_prose_sentence(sentence: str) -> bool:
    """Exclude headings, table rows and form-like fragments from prose statistics.

    Document-level burstiness and rhetorical checks become misleading when table
    rows, references, headings and short form labels are treated as sentences.
    """
    s = re.sub(r"\s+", " ", sentence or "").strip()
    words = WORD_RE.findall(s)
    if len(words) < 6:
        return False
    if re.match(r"^(?:Table|Figure|Appendix)\s+\d+[A-Za-z]?[.:]?\s", s, re.I):
        return False
    if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z][A-Za-z]", s):
        return False
    if re.match(r"^(?:FULL LEGAL NAME|LOCATION|EMAIL ADDRESS|Team member\s+\d+|Works Cited|References|Bibliography)\b", s, re.I):
        return False
    # Rows dominated by numbers/tickers are data, not prose rhythm evidence.
    tokens = re.findall(r"\S+", s)
    numeric_like = sum(bool(re.fullmatch(r"[-+<>]?\d+(?:\.\d+)?%?|[A-Z]{2,6}|\.?\d{3,}", token.strip("(),;:"))) for token in tokens)
    if tokens and numeric_like / len(tokens) >= 0.45:
        return False
    return True


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _score(evidence: list[dict[str, str]]) -> int:
    weak = sum(1 for e in evidence if e["severity"] == "weak")
    moderate = sum(1 for e in evidence if e["severity"] == "moderate")
    strong = sum(1 for e in evidence if e["severity"] == "strong")
    if strong >= 1 or moderate >= 2 or weak >= 4:
        return 3
    if moderate >= 1 or weak >= 2:
        return 2
    if weak >= 1:
        return 1
    return 0


def _add(evidence: list[dict[str, str]], description: str, severity: str) -> None:
    if description and not any(item["description"] == description for item in evidence):
        evidence.append({"description": description, "severity": severity})


def _quote(text: str, limit: int = 115) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def _sentence_lengths(sentences: list[str]) -> list[int]:
    return [len(WORD_RE.findall(s)) for s in sentences]


def _cv(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if not mean:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean


def _tricolon_like(sentence: str) -> bool:
    # Structural proxy. A single academic list is weak evidence, repeated symmetry is more useful.
    bits = [b.strip() for b in sentence.split(",")]
    if len(bits) != 3:
        return False
    lens = [len(_words(b)) for b in bits]
    if min(lens, default=0) < 3 or max(lens) - min(lens) > 4:
        return False
    starts = [(_words(bit) or [""])[0] for bit in bits]
    # Parallel starts or near-identical item lengths make the construction more diagnostic.
    return len(set(starts)) <= 2 or max(lens) - min(lens) <= 2


def _parallel_reason_run(sentences: list[str]) -> int:
    longest = 0
    run = 0
    for sentence in sentences:
        if REASON_CLAUSE_RE.search(sentence):
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _paragraph_ai_profile(paragraphs: list[str], academic: bool) -> tuple[int, list[int]]:
    if not paragraphs:
        return 0, []
    risks: list[int] = []
    word_counts: list[int] = []
    for paragraph in paragraphs:
        sentences = _sentences(paragraph)
        local = [sentence_ai_signal(sentence, academic=academic)[0] for sentence in sentences]
        # Use the strongest local sentence plus a smaller contribution from repeated lower-level signals.
        peak = max(local, default=0)
        mean = sum(local) / max(1, len(local))
        risks.append(min(100, round(peak * 0.7 + mean * 0.3)))
        word_counts.append(max(1, len(_words(paragraph))))
    weighted = sum(r * w for r, w in zip(risks, word_counts)) / max(1, sum(word_counts))
    return round(weighted), risks


def _anaphora(sentences: list[str]) -> list[tuple[str, int]]:
    starts = []
    for s in sentences:
        words = WORD_RE.findall(s.lower())[:2]
        starts.append(" ".join(words))
    hits: list[tuple[str, int]] = []
    i = 0
    while i < len(starts):
        j = i + 1
        while j < len(starts) and starts[j] == starts[i] and starts[i]:
            j += 1
        if j - i >= 2:
            hits.append((starts[i], j - i))
        i = j
    return hits


def _verdict(score: int) -> str:
    if score <= 4:
        return "Human"
    if score <= 8:
        return "Likely Human"
    if score <= 13:
        return "Uncertain"
    if score <= 19:
        return "Likely AI"
    return "AI"


def _verdict_from_index(ai_pct: int) -> str:
    """Return a descriptive signal band, not an authorship claim.

    Commercial detectors can disagree sharply on polished scholarly prose, so the
    public label describes the strength of patterns observed by this application
    rather than asserting who or what wrote the text.
    """
    if ai_pct < 20:
        return "Minimal AI-style signal"
    if ai_pct < 40:
        return "Low AI-style signal"
    if ai_pct < 60:
        return "Moderate AI-style signal"
    if ai_pct < 80:
        return "Elevated AI-style signal"
    return "Strong AI-style signal"


def _confidence(text: str, score: int, scores: list[int]) -> str:
    wc = len(_words(text))
    active = sum(1 for x in scores if x > 0)
    corroborating = sum(1 for x in scores if x >= 2)
    if wc < 100:
        return "Medium" if corroborating >= 3 else "Low"
    if wc < 180:
        return "Medium"
    if active >= 6 and corroborating >= 4 and (score >= 14 or score <= 4):
        return "High"
    return "Medium"


def sentence_ai_signal(sentence: str, academic: bool = True) -> tuple[int, list[str]]:
    """Return a local sentence-level AI-style signal score for colour highlighting."""
    s = sentence.strip()
    if not s:
        return 0, []
    risk = 0
    reasons: list[str] = []
    lower = s.lower()

    vocab_hits = [item for item in AI_VOCAB if item in lower and not (academic and item == "robust")]
    if vocab_hits:
        risk += min(30, 10 + (len(vocab_hits) - 1) * 6)
        reasons.append("Predictable AI-associated vocabulary: " + ", ".join(vocab_hits[:3]))
    hedge_hits = [p for p in HEDGE_PATTERNS if re.search(p, s, re.I)]
    if hedge_hits:
        risk += min(24 if academic else 32, 8 + len(hedge_hits) * (5 if academic else 7))
        reasons.append("Institutional hedge pattern")
    if any(re.search(p, s, re.I) for p in TRANSITION_STRONG):
        risk += 28
        reasons.append("Strong transition-word fingerprint")
    elif any(re.search(p, s, re.I) for p in TRANSITION_MODERATE):
        risk += 18
        reasons.append("Formulaic transition or announcement")
    if SETUP_RE.search(s) or DIMINISHMENT_RE.search(s) or MORE_THAN_RE.search(s) or TURNS_OUT_RE.search(s):
        risk += 22
        reasons.append("Rhetorical scaffolding pattern")
    if PATTERN_ANNOUNCEMENT_RE.search(s):
        risk += 14
        reasons.append("Pattern/insight announcement frame")
    if PARTICIPIAL_REFRAME_RE.search(s):
        risk += 16
        reasons.append("Participial reframe pivot")
    if COMPOSED_PARENT_RE.search(s):
        risk += 16
        reasons.append("Composed self-aware parenthetical")
    if BALANCED_PAREN_RE.search(s):
        risk += 22
        reasons.append("Balanced parenthetical trade-off pair")
    if PARALLEL_QUESTION_LIST_RE.search(s):
        risk += 24
        reasons.append("Within-sentence parallel question list")
    if THESIS_FIRST_RE.search(s) or ACTUAL_WORK_RE.search(s):
        risk += 20
        reasons.append("Formulaic thesis/landing construction")
    if PASSIVE_RE.search(s) or UNIVERSAL_RE.search(s):
        risk += 16
        reasons.append("Specificity deficit or obscured actor")
    if _tricolon_like(s):
        risk += 12 if academic else 24
        reasons.append("Symmetrical three-part construction")
    if s.count("—") >= 2:
        risk += 28
        reasons.append("Double em-dash construction")
    elif s.count("—") == 1:
        risk += 10
        reasons.append("Em-dash pivot/aside")
    if not academic and ";" in s:
        risk += 12
        reasons.append("Semicolon fingerprint in non-academic prose")
    if re.search(r"\b(?:The problem|The answer|The rule|The key insight|The approach):", s, re.I):
        risk += 18
        reasons.append("Announcement-colon pattern")
    if TEMPORAL_VAGUE_RE.search(s) and not YEAR_RE.search(s):
        risk += 8
        reasons.append("Vague current-time framing")

    return min(100, risk), reasons


def _prose_sentences_from_text(text: str) -> list[str]:
    """Extract prose without letting headings, tables or form rows pollute signals."""
    results: list[str] = []
    for line_no, line in enumerate((text or "").splitlines()):
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue
        if "\t" in line or " | " in line:
            continue
        if re.match(r"^(?:Table|Figure|Fig\.?|Appendix)\s+\d*[A-Za-z]?[.:]?\b", stripped, re.I):
            continue
        if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z]", stripped):
            continue
        if re.match(r"^(?:Abstract|Executive Summary|Introduction|Conclusion|Recommendations?|Limitations?|References|Works Cited|Bibliography)$", stripped, re.I):
            continue
        if re.match(r"^(?:FULL LEGAL NAME|LOCATION|EMAIL ADDRESS|Team member\s+\d+)\b", stripped, re.I):
            continue
        # Titles and affiliation lines often have no sentence-ending punctuation.
        if line_no < 15 and not re.search(r"[.!?]$", stripped) and len(_words(stripped)) <= 30:
            continue
        for sentence in _sentences(stripped):
            if _is_prose_sentence(sentence):
                results.append(sentence)
    return results


def _prose_paragraphs_from_text(text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text or ""):
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or "\t" in line or " | " in line:
                continue
            if re.match(r"^(?:Table|Figure|Fig\.?|Appendix)\s+\d*[A-Za-z]?[.:]?\b", stripped, re.I):
                continue
            if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z]", stripped):
                continue
            if re.match(r"^(?:Abstract|Executive Summary|Introduction|Conclusion|Recommendations?|Limitations?|References|Works Cited|Bibliography)$", stripped, re.I):
                continue
            if len(_words(stripped)) >= 6:
                lines.append(stripped)
        if lines:
            paragraphs.append(" ".join(lines))
    return paragraphs


def _humanness_counter_evidence(text: str, prose_sentences: list[str], academic: bool) -> tuple[int, list[dict[str, str]]]:
    """Score counter-evidence before interpreting AI-style cues.

    This is intentionally conservative. It never rewards deliberately inserted
    mistakes. Existing idiosyncrasies, self-corrections and domain-specific
    technical execution can offset weak style-only signals.
    """
    ev: list[dict[str, str]] = []
    raw = text or ""
    explicit_self_correction = re.search(r"\b(?:wait[,—-]|I mean[,—-]|that(?:'|’)s not quite right|no,? that(?:'|’)s not)\b", raw, re.I)
    if explicit_self_correction:
        _add(ev, "Natural self-correction or mid-thought revision appears in the text.", "moderate")
    if re.search(r"\b(?:I|we)\b.{0,90}\b(?:on|in|during)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}\s+[A-Z][a-z]+)\b", raw, re.I):
        _add(ev, "Specific personal/time-anchored detail appears rather than generic illustration.", "moderate")
    # Academic counter-evidence: exact methods + statistics + named datasets or software.
    if academic:
        technical_hits = 0
        if re.search(r"\b(?:Friedman|Wilcoxon|Kendall(?:'s)?\s+W|latent class analysis|LCA|Cram[eé]r(?:'s)?\s+V|chi-square|p\s*[<=>]|AIC|BIC|entropy|StepMix|SciPy|NumPy)\b", raw, re.I):
            technical_hits += 1
        if len(re.findall(r"\b\d+(?:\.\d+)?%\b", raw)) >= 5 or len(re.findall(r"\bp\s*[<=>]\s*\d", raw, re.I)) >= 2:
            technical_hits += 1
        if re.search(r"\b(?:World Bank|Global Public Procurement Database|GPPD|OECD|Yahoo Finance|Kenneth French)\b", raw, re.I):
            technical_hits += 1
        if technical_hits >= 3:
            _add(ev, "Dense domain-specific methods, exact statistics and named data sources provide counter-evidence to style-only cues.", "weak")
        elif technical_hits == 2:
            _add(ev, "Domain-specific technical detail provides some counter-evidence to generic-style cues.", "weak")
    return _score(ev), ev


def ai_check_report(text: str, global_report: dict[str, Any] | None = None, academic: bool = True) -> dict[str, Any]:
    raw = str(text or "")
    prose_sentences = _prose_sentences_from_text(raw)
    if not prose_sentences:
        prose_sentences = [s for s in _sentences(raw) if _is_prose_sentence(s)]
    paragraphs = _prose_paragraphs_from_text(raw)
    if not paragraphs:
        paragraphs = _paragraphs(raw)
    prose_text = "\n".join(prose_sentences)
    words = _words(prose_text)
    wc = len(words)
    lengths = _sentence_lengths(prose_sentences)
    sent_cv = _cv(lengths)
    paragraph_lengths = [len(_words(p)) for p in paragraphs]
    para_cv = _cv(paragraph_lengths)
    global_report = global_report or {}

    # J is deliberately scored before A-I to counter one-sided anchoring.
    j_score, j_evidence = _humanness_counter_evidence(raw, prose_sentences, academic)
    signals: list[dict[str, Any]] = []

    # A. Perplexity / predictability proxy
    ev: list[dict[str, str]] = []
    low = prose_text.lower()
    vocab_hits: list[str] = []
    for item in AI_VOCAB:
        if academic and item in {"robust", "comprehensive"}:
            continue
        count = low.count(item)
        vocab_hits.extend([item] * count)
    vocab_density = len(vocab_hits) / max(1, wc) * 1000
    if vocab_hits and (len(set(vocab_hits)) >= 2 or vocab_density >= 2.5):
        unique = list(dict.fromkeys(vocab_hits))
        if vocab_density >= 6 or len(vocab_hits) >= 8:
            sev = "strong"
        elif vocab_density >= 3 or len(vocab_hits) >= 4:
            sev = "moderate"
        else:
            sev = "weak"
        _add(ev, f"AI-associated safe vocabulary occurs at {vocab_density:.1f} per 1,000 words: {', '.join(unique[:7])}", sev)
    generic_adj = re.findall(r"\b(?:significant|notable|key|important)\s+(?:improvements?|progress|challenges?|issues?|factors?)\b", prose_text, re.I)
    if len(generic_adj) >= (4 if academic else 2):
        _add(ev, f"Generic evaluative wording appears {len(generic_adj)} time(s), e.g. “{_quote(generic_adj[0])}”.", "moderate" if len(generic_adj) >= 6 else "weak")
    score = _score(ev)
    signals.append({"key": "A", "name": "Perplexity", "score": score, "evidence": ev, "summary": "Safe/predictable vocabulary and generic phrasing."})

    # B. Burstiness deficit. In a long paper, one coincidental 3-sentence cluster is not enough.
    ev = []
    cluster_count = 0
    for i in range(max(0, len(lengths) - 2)):
        window = lengths[i:i+3]
        if len(window) == 3 and max(window) - min(window) <= 5:
            cluster_count += 1
    cluster_share = cluster_count / max(1, len(lengths) - 2)
    if len(lengths) < 8 and cluster_count:
        _add(ev, f"Three consecutive sentence lengths are tightly clustered: {lengths[:3] if len(lengths)>=3 else lengths}.", "moderate")
    elif cluster_share >= 0.28:
        _add(ev, f"Tightly clustered three-sentence windows occur in {cluster_share:.0%} of the passage.", "moderate")
    elif cluster_share >= 0.14 and sent_cv < 0.45:
        _add(ev, f"Repeated sentence-length clustering occurs in {cluster_share:.0%} of the passage.", "weak")
    if len(lengths) >= 8 and sent_cv < 0.22:
        _add(ev, f"Sentence-length variation is low (CV {sent_cv:.2f}).", "moderate")
    elif len(lengths) >= 8 and sent_cv < 0.30:
        _add(ev, f"Sentence rhythm is fairly uniform (CV {sent_cv:.2f}).", "weak")
    if not academic and wc >= 150 and lengths and min(lengths) >= 8:
        _add(ev, "No sentence shorter than 8 words appears in this long passage.", "weak")
    score = _score(ev)
    signals.append({"key": "B", "name": "Burstiness", "score": score, "evidence": ev, "summary": f"Sentence-length CV {sent_cv:.2f}; rhythm variation check."})

    # C. Hedge density, calibrated to density rather than raw count.
    ev = []
    hedge_matches: list[str] = []
    for pattern in HEDGE_PATTERNS:
        hedge_matches.extend(m.group(0) for m in re.finditer(pattern, prose_text, re.I))
    hedge_density = len(hedge_matches) / max(1, wc) * 1000
    if hedge_matches and (hedge_density >= (3.0 if academic else 2.0) or len(hedge_matches) >= (7 if academic else 5)):
        sev = "moderate" if hedge_density >= (7 if academic else 5) else "weak"
        _add(ev, f"Hedge language appears {len(hedge_matches)} time(s), {hedge_density:.1f} per 1,000 words, including “{_quote(hedge_matches[0])}”.", sev)
    score = _score(ev)
    signals.append({"key": "C", "name": "Hedge density", "score": score, "evidence": ev, "summary": "Institutional softening and reflexive uncertainty."})

    # D. Structural tells. Numerical result summaries are not treated as rhetorical tricolons.
    ev = []
    bullet_lines = [line for line in raw.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)]
    bullet_words = sum(len(_words(line)) for line in bullet_lines)
    bullet_share = bullet_words / max(1, wc)
    if (not academic and len(bullet_lines) >= 5) or (academic and len(bullet_lines) >= 12 and bullet_share >= 0.45):
        _add(ev, f"Structured list contains {len(bullet_lines)} items and occupies {bullet_share:.0%} of prose.", "weak")
    if re.search(r"(?:^|\n)\s*(?:In conclusion|To summarize|In summary)\b", prose_text, re.I):
        _add(ev, "Formulaic conclusion opener detected.", "moderate")
    tricolons = []
    for sentence in prose_sentences:
        numeric_count = len(NUMBER_RE.findall(sentence))
        if numeric_count >= 3:
            continue
        if _tricolon_like(sentence):
            tricolons.append(sentence)
    if len(tricolons) >= 2:
        _add(ev, f"Repeated symmetrical three-part construction, e.g. “{_quote(tricolons[0])}”.", "moderate" if academic else "strong")
    elif tricolons and not academic:
        _add(ev, f"Symmetrical three-part construction: “{_quote(tricolons[0])}”.", "moderate")
    straw = next((s for s in prose_sentences if DIMINISHMENT_RE.search(s)), None)
    if straw:
        _add(ev, f"Not-only/not-X rhetorical pivot: “{_quote(straw)}”.", "weak" if academic else "moderate")
    if len(paragraph_lengths) >= 5 and para_cv < 0.18:
        _add(ev, f"Paragraph lengths are unusually even (CV {para_cv:.2f}).", "weak" if academic else "moderate")
    repeated_frames = int(global_report.get("repeated_frame_density") or 0)
    if repeated_frames >= (7 if academic else 4):
        _add(ev, f"Repeated framing phrases occur at elevated density ({repeated_frames}).", "weak" if academic else "moderate")
    score = _score(ev)
    signals.append({"key": "D", "name": "Structural tells", "score": score, "evidence": ev, "summary": "Template-like document architecture and symmetry."})

    # E. Specificity deficit
    ev = []
    passive_hits = [s for s in prose_sentences if PASSIVE_RE.search(s)]
    if len(passive_hits) >= 2:
        _add(ev, f"Actor-obscuring evidence frame: “{_quote(passive_hits[0])}”.", "moderate")
    universal_hits = [s for s in prose_sentences if UNIVERSAL_RE.search(s)]
    if len(universal_hits) >= 2:
        _add(ev, f"Universalist framing: “{_quote(universal_hits[0])}”.", "moderate")
    score = _score(ev)
    signals.append({"key": "E", "name": "Specificity", "score": score, "evidence": ev, "summary": "Concrete names, actors, numbers, dates and examples."})

    # F. Transition fingerprint
    ev = []
    strong_hits: list[str] = []
    moderate_hits: list[str] = []
    for pattern in TRANSITION_STRONG:
        strong_hits.extend(m.group(0) for m in re.finditer(pattern, prose_text, re.I | re.M))
    for pattern in TRANSITION_MODERATE:
        moderate_hits.extend(m.group(0) for m in re.finditer(pattern, prose_text, re.I | re.M))
    if strong_hits:
        _add(ev, f"Strong formulaic transition(s): {', '.join('“'+_quote(x,45)+'”' for x in strong_hits[:4])}.", "strong" if len(strong_hits) >= 2 else "moderate")
    if len(moderate_hits) >= 2:
        _add(ev, f"Repeated formulaic pivot/announcement(s): {', '.join('“'+_quote(x,45)+'”' for x in moderate_hits[:4])}.", "moderate")
    score = _score(ev)
    signals.append({"key": "F", "name": "Transitions", "score": score, "evidence": ev, "summary": "Mechanical connective tissue and reveal/announcement phrasing."})

    # G. Punctuation fingerprint. Count punctuation only in prose, never table cells.
    ev = []
    em_count = prose_text.count("—")
    if em_count > max(1, wc // 300):
        _add(ev, f"Em dashes appear {em_count} time(s), above the prose-length threshold.", "moderate")
    if re.search(r"—[^—\n]{1,120}—", prose_text):
        _add(ev, "Double em-dash wrapping appears as a dramatic aside.", "weak" if academic else "strong")
    punctuation_text = CITATION_RE.sub("", prose_text)
    semicolons = punctuation_text.count(";")
    if not academic and semicolons:
        _add(ev, f"Semicolons linking clauses appear {semicolons} time(s).", "weak" if semicolons == 1 else "moderate")
    elif academic and semicolons >= max(8, wc // 100):
        _add(ev, f"Semicolon use is unusually dense even for academic prose ({semicolons} instances).", "weak")
    colon_hits = re.findall(r"\b(?:The problem|The answer|The rule|The key insight|The approach):", prose_text, re.I)
    if colon_hits:
        _add(ev, f"Announcement-colon construction: “{colon_hits[0]}”.", "moderate")
    score = _score(ev)
    signals.append({"key": "G", "name": "Punctuation", "score": score, "evidence": ev, "summary": "Em-dash, semicolon and announcement-colon patterns."})

    # H. Voice/register, heavily down-weighted for academic prose.
    ev = []
    informal = bool(INFORMAL_MARKER_RE.search(prose_text))
    complete_ratio = sum(1 for s in prose_sentences if re.search(r"[.!?]$", s)) / max(1, len(prose_sentences))
    if informal and len(prose_sentences) >= 4 and complete_ratio > 0.9:
        _add(ev, "Informal markers sit on top of unusually polished complete-sentence prose.", "strong")
    traces = sum(bool(rx.search(prose_text)) for rx in (FIRST_PERSON_RE, SECOND_PERSON_RE, RHETORICAL_Q_RE, SELF_CORRECTION_RE))
    if wc >= 300 and traces == 0 and not academic:
        _add(ev, "Long passage contains no first/second person, rhetorical question, or self-correction trace.", "moderate")
    score = _score(ev)
    signals.append({"key": "H", "name": "Voice / register", "score": score, "evidence": ev, "summary": "Human traces, register variation and templated professional voice."})

    # I. Rhetorical scaffolding: corroborating evidence only in scholarly prose.
    ev = []
    patterns = [
        (SETUP_RE, "Setup/revelation announcement sentence"),
        (MORE_THAN_RE, "'More X than Y' comparative framing"),
        (DIMINISHMENT_RE, "'Not X but Y' diminishment framing"),
        (TURNS_OUT_RE, "'Turns out' reveal pivot"),
        (BINARY_RE, "Clean binary 'either/or' framing"),
        (BINARY_CHOICE_RE, "Choice/trade-off framed as a clean binary"),
        (ACTUAL_WORK_RE, "'is the actual/real work' landing phrase"),
        (PATTERN_ANNOUNCEMENT_RE, "Pattern/insight announcement frame"),
    ]
    for regex, label in patterns:
        matches = [s for s in prose_sentences if regex.search(s)]
        if matches and (not academic or len(matches) >= 2):
            _add(ev, f"{label}: “{_quote(matches[0])}”.", "weak" if academic else "moderate")
    for opener, count in _anaphora(prose_sentences):
        if count >= 3:
            _add(ev, f"Repeated sentence starter “{opener}” occurs {count} times consecutively.", "moderate" if not academic else "weak")
            break
    question_matches = [s for s in prose_sentences if PARALLEL_QUESTION_LIST_RE.search(s)]
    if len(question_matches) >= 2:
        _add(ev, f"Repeated within-sentence parallel question lists, e.g. “{_quote(question_matches[0])}”.", "moderate" if academic else "strong")
    elif question_matches and not academic:
        _add(ev, f"Within-sentence parallel question list: “{_quote(question_matches[0])}”.", "moderate")
    if len(prose_sentences) >= 8 and sent_cv < 0.25 and len(paragraphs) >= 3 and para_cv < 0.25:
        _add(ev, "Local coherence/rhythm is unusually smooth across sentence and paragraph scales.", "weak" if academic else "moderate")
    score = _score(ev)
    signals.append({"key": "I", "name": "Rhetorical scaffolding", "score": score, "evidence": ev, "summary": "Over-composed rhetorical devices and smoothness patterns."})

    raw_total = sum(int(signal["score"]) for signal in signals)
    weighted_parts = []
    for signal in signals:
        weight = CATEGORY_WEIGHTS[signal["key"]]
        weighted = round(int(signal["score"]) * weight, 2)
        signal["weight"] = weight
        signal["weighted_score"] = weighted
        weighted_parts.append(weighted)
    weighted_raw_total = round(sum(weighted_parts), 2)
    scaled_score = weighted_raw_total * (27 / WEIGHTED_MAX) if WEIGHTED_MAX else 0.0
    net_score = max(0.0, min(27.0, scaled_score - j_score))
    displayed_net_score = round(net_score, 1)
    ai_pct = max(0, min(100, round(displayed_net_score / 27 * 100)))
    verdict = _verdict_from_index(ai_pct)

    conditional_share = sum(signal["weighted_score"] for signal in signals if signal["key"] in {"C", "H", "I"}) / max(0.001, weighted_raw_total)
    confidence = _confidence(prose_text, round(net_score), [int(s["score"]) for s in signals])
    reliability_flag = None
    if conditional_share > 0.60 and weighted_raw_total > 0:
        reliability_flag = ">60% of weighted score comes from register-dependent C/H/I signals; confidence capped at Medium."
        if confidence == "High":
            confidence = "Medium"

    paragraph_ai_pct, paragraph_risks = _paragraph_ai_profile(paragraphs, academic=academic)
    sentence_risks = [sentence_ai_signal(sentence, academic=academic)[0] for sentence in prose_sentences]
    flagged_sentence_count = sum(1 for risk in sentence_risks if risk >= 25)
    moderate_sentence_count = sum(1 for risk in sentence_risks if risk >= 45)
    high_sentence_count = sum(1 for risk in sentence_risks if risk >= 70)

    # Paragraph/section profiling surfaces mixed-text hotspots without contaminating
    # the document score with headings and tables.
    segment_profile: list[dict[str, Any]] = []
    if len(paragraphs) >= 3:
        for index, paragraph in enumerate(paragraphs[:80]):
            local_risks = [sentence_ai_signal(s, academic=academic)[0] for s in _sentences(paragraph) if _is_prose_sentence(s)]
            peak = max(local_risks, default=0)
            mean = sum(local_risks) / max(1, len(local_risks))
            local_pct = min(100, round(peak * 0.65 + mean * 0.35))
            segment_profile.append({"segment": index + 1, "ai_signal": local_pct, "label": _verdict_from_index(local_pct), "excerpt": _quote(paragraph, 120)})

    ranked = sorted(signals, key=lambda x: (x["weighted_score"], len(x["evidence"])), reverse=True)
    strongest = [s for s in ranked if s["score"] > 0][:3]
    if strongest:
        gave_away = "The strongest remaining signals were " + ", ".join(f"{s['name']} ({s['score']}/3; weighted {s['weighted_score']:.2f})" for s in strongest) + "."
    else:
        gave_away = "No category produced a meaningful AI-style signal."

    calibration_notes = [
        "AI Signal is derived directly from the weighted forensic net score, so the headline percentage and category statistics reconcile.",
        "The nine visible categories are weighted by reliability; rhetorical and register-based signals carry less weight in scholarly prose.",
        "Humanness counter-evidence is scored separately and subtracted before the final 0–27 net score.",
        "Tables, form rows, headings and table punctuation are excluded from prose-style scoring.",
        "Different AI-writing detectors can disagree substantially on the same scholarly passage; use this as a style-screening indicator, not proof of authorship.",
    ]
    if wc < 100:
        calibration_notes.append("Text under 100 prose words has limited evidence; confidence is capped at Medium.")
        if confidence == "High":
            confidence = "Medium"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "overall_score": displayed_net_score,
        "net_score_exact": round(net_score, 3),
        "max_score": 27,
        "raw_category_score": raw_total,
        "weighted_raw_score": weighted_raw_total,
        "scaled_score_before_humanness": round(scaled_score, 2),
        "humanness_counter_score": j_score,
        "humanness_counter_evidence": j_evidence,
        "forensic_verdict": _verdict(round(net_score)),
        "ai_detection_percentage": ai_pct,
        "signal_level": verdict,
        "signals": signals,
        "what_gave_it_away": gave_away,
        "calibration_notes": calibration_notes,
        "word_count": wc,
        "sentence_lengths": lengths,
        "category_signal_percentage": ai_pct,
        "evidence_signal_percentage": ai_pct,
        "sentence_signal_percentage": round(sum(sentence_risks) / max(1, len(sentence_risks))) if sentence_risks else 0,
        "flagged_sentence_count": flagged_sentence_count,
        "moderate_sentence_count": moderate_sentence_count,
        "high_sentence_count": high_sentence_count,
        "paragraph_ai_signal_percentage": paragraph_ai_pct,
        "paragraph_signal_profile": paragraph_risks,
        "segment_profile": segment_profile,
        "reliability_flag": reliability_flag,
        "detector_variability_notice": (
            "AI-writing detectors can disagree substantially, especially on formal academic prose. "
            "Use this score as a style-screening indicator, not proof of authorship."
        ),
    }
