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
BINARY_RE = re.compile(r"\b(?:either\b.{1,100}\bor\b|between\b.{1,100}\band\b)", re.I)
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

PATTERN_ANNOUNCEMENT_RE = re.compile(r"\b(?:the pattern is|the rule is|the key point is|the key insight is|what (?:I|we|it)[^.!?]{0,60} was)\b", re.I)
PARTICIPIAL_REFRAME_RE = re.compile(r"^(?:Laid out|Arranged|Seen|Viewed|Framed|Read|Presented)\s+(?:this way|that way|in this way|in these terms|in a [^,]{1,35})?,?", re.I)
COMPOSED_PARENT_RE = re.compile(r"\((?:which|something)\s+I\s+(?:choose to read as|take as|am choosing to interpret as|interpret as)[^)]{0,90}\)", re.I)
BALANCED_PAREN_RE = re.compile(r"\([^()]{1,70},\s*but\s+[^()]{1,70}\)\s+(?:or|and)\s+\([^()]{1,70},\s*but\s+[^()]{1,70}\)", re.I)
APOSTROPHE_CLOSER_RE = re.compile(r"^(?:That(?:'|’)s|This is|It is|The work|The point|The lesson)\b", re.I)
REASON_CLAUSE_RE = re.compile(r"\b(?:because|when)\b", re.I)
PARALLEL_QUESTION_LIST_RE = re.compile(r"\b(?:what|why|how)\b[^,.]{1,60},\s*\b(?:what|why|how)\b[^,.]{1,60},\s*\b(?:what|why|how)\b[^,.]{1,60}(?:,\s*\b(?:what|why|how)\b[^,.]{1,60})?", re.I)

SEVERITY_WEIGHT = {"weak": 1, "moderate": 2, "strong": 3}


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in WORD_RE.findall(text or "")]


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
    return parts if parts else ([text.strip()] if text and text.strip() else [])


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


def _fraction(score: int, scores: list[int]) -> str:
    active = sum(1 for x in scores if x > 0)
    strong = sum(1 for x in scores if x == 3)
    if score <= 4:
        return "Pure human (~0%)"
    if score <= 8:
        return "Lightly AI-assisted (~10–30%)"
    if score <= 13:
        return "Mixed authorship (~30–60%)"
    if score <= 19 or active < 8 or strong < 5:
        return "Heavily AI-edited (~60–90%)"
    return "Pure AI (~100%)"


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

    vocab_hits = [item for item in AI_VOCAB if item in lower]
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


def ai_check_report(text: str, global_report: dict[str, Any] | None = None, academic: bool = True) -> dict[str, Any]:
    raw = str(text or "")
    sentences = _sentences(raw)
    paragraphs = _paragraphs(raw)
    words = _words(raw)
    wc = len(words)
    lengths = _sentence_lengths(sentences)
    sent_cv = _cv(lengths)
    paragraph_lengths = [len(_words(p)) for p in paragraphs]
    para_cv = _cv(paragraph_lengths)
    global_report = global_report or {}

    signals: list[dict[str, Any]] = []

    # A. Perplexity / predictability proxy
    ev: list[dict[str, str]] = []
    vocab_hits = []
    low = raw.lower()
    for item in AI_VOCAB:
        count = low.count(item)
        if count:
            vocab_hits.extend([item] * count)
    if vocab_hits:
        unique = list(dict.fromkeys(vocab_hits))
        sev = "strong" if len(vocab_hits) >= 4 else "moderate" if len(vocab_hits) >= 2 else "weak"
        _add(ev, f"AI-associated safe vocabulary: {', '.join(unique[:7])}", sev)
    generic_adj = re.findall(r"\b(?:significant|notable|key|important)\s+(?:improvements?|progress|challenges?|issues?|factors?)\b", raw, re.I)
    if generic_adj:
        _add(ev, f"Generic evaluative wording appears {len(generic_adj)} time(s), e.g. “{_quote(generic_adj[0])}”.", "moderate" if len(generic_adj) >= 2 else "weak")
    safe_hedges = re.findall(r"\b(?:can often lead to|may result in|tends? to)\b", raw, re.I)
    if safe_hedges:
        _add(ev, f"Predictability-increasing safe hedge: “{safe_hedges[0]}”.", "weak")
    score = _score(ev)
    signals.append({"key": "A", "name": "Perplexity", "score": score, "evidence": ev, "summary": "Safe/predictable vocabulary and generic phrasing."})

    # B. Burstiness deficit
    ev = []
    for i in range(max(0, len(lengths) - 2)):
        window = lengths[i:i+3]
        if len(window) == 3 and max(window) - min(window) <= 5:
            _add(ev, f"Three consecutive sentence lengths are tightly clustered: {window[0]}, {window[1]}, {window[2]} words.", "moderate")
            break
    if len(lengths) >= 6 and sent_cv < 0.22:
        _add(ev, f"Sentence-length variation is low (CV {sent_cv:.2f}).", "moderate")
    elif len(lengths) >= 5 and sent_cv < 0.32:
        _add(ev, f"Sentence rhythm is fairly uniform (CV {sent_cv:.2f}).", "weak")
    if wc >= 150 and lengths and min(lengths) >= 8:
        _add(ev, "No sentence shorter than 8 words appears in this long passage.", "weak")
    if wc >= 220 and not any(0 < n < 6 for n in lengths):
        _add(ev, "No short fragment-like sentence appears for emphasis.", "weak")
    short_run = 0
    for n in lengths:
        short_run = short_run + 1 if n < 7 else 0
        if short_run >= 3:
            _add(ev, "Three or more very short sentences occur consecutively without a longer counterweight.", "weak")
            break
    score = _score(ev)
    signals.append({"key": "B", "name": "Burstiness", "score": score, "evidence": ev, "summary": f"Sentence-length CV {sent_cv:.2f}; rhythm variation check."})

    # C. Hedge density, down-weighted for scholarly register
    ev = []
    hedge_matches: list[str] = []
    for pattern in HEDGE_PATTERNS:
        hedge_matches.extend(m.group(0) for m in re.finditer(pattern, raw, re.I))
    if hedge_matches:
        density = len(hedge_matches) / max(1, wc) * 1000
        sev = "moderate" if density >= (12 if academic else 8) or len(hedge_matches) >= 5 else "weak"
        _add(ev, f"Hedge language appears {len(hedge_matches)} time(s), including “{_quote(hedge_matches[0])}”.", sev)
    diplomatic = re.findall(r"\bwhile\b.{0,120}\b(?:benefits?|advantages?)\b.{0,120}\b(?:challenges?|limitations?|drawbacks?)\b", raw, re.I | re.S)
    if diplomatic:
        _add(ev, f"Balanced diplomatic framing: “{_quote(diplomatic[0])}”.", "weak" if academic else "moderate")
    score = _score(ev)
    signals.append({"key": "C", "name": "Hedge density", "score": score, "evidence": ev, "summary": "Institutional softening and reflexive uncertainty."})

    # D. Structural tells
    ev = []
    bullet_lines = [line for line in raw.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line)]
    if len(bullet_lines) >= 5:
        _add(ev, f"Structured list contains {len(bullet_lines)} items; verify that the content is genuinely sequential/list-like.", "weak")
    if re.search(r"(?:^|\n)\s*(?:In conclusion|To summarize|In summary)\b", raw, re.I):
        _add(ev, "Formulaic conclusion opener detected.", "moderate")
    if re.search(r"(?:^|\n)\s*In this (?:post|article|section|chapter) I will\b", raw, re.I):
        _add(ev, "Formulaic 'In this … I will' opener detected.", "moderate")
    tricolons = [sentence for sentence in sentences if _tricolon_like(sentence)]
    if tricolons:
        tri_severity = "strong" if len(tricolons) >= 2 else ("weak" if academic else "moderate")
        _add(ev, f"Symmetrical three-part construction: “{_quote(tricolons[0])}”.", tri_severity)
    straw = next((s for s in sentences if DIMINISHMENT_RE.search(s)), None)
    if straw:
        _add(ev, f"Strawman/diminishment pivot: “{_quote(straw)}”.", "moderate")
    if len(paragraph_lengths) >= 5 and para_cv < 0.20:
        _add(ev, f"Paragraph lengths are unusually even (CV {para_cv:.2f}), consistent with a templated paragraph-per-idea arc.", "moderate")
    repeated_openings = int(global_report.get("repeated_sentence_openings") or 0) + int(global_report.get("repeated_paragraph_openings") or 0)
    repeated_frames = int(global_report.get("repeated_frame_density") or 0)
    if repeated_openings >= 3:
        _add(ev, f"Repeated sentence/paragraph openings occur {repeated_openings} times across the passage.", "moderate")
    if repeated_frames >= 4:
        _add(ev, f"Repeated framing phrases occur at elevated density ({repeated_frames}).", "moderate")
    score = _score(ev)
    signals.append({"key": "D", "name": "Structural tells", "score": score, "evidence": ev, "summary": "Template-like document architecture and symmetry."})

    # E. Specificity deficit
    ev = []
    passive_hits = [s for s in sentences if PASSIVE_RE.search(s)]
    if passive_hits:
        _add(ev, f"Actor-obscuring evidence frame: “{_quote(passive_hits[0])}”.", "moderate" if len(passive_hits) >= 2 else "weak")
    universal_hits = [s for s in sentences if UNIVERSAL_RE.search(s)]
    if universal_hits:
        _add(ev, f"Universalist framing: “{_quote(universal_hits[0])}”.", "moderate" if len(universal_hits) >= 2 else "weak")
    abstract_candidates = []
    for s in sentences:
        if re.search(r"\b(?:many|various|several|numerous) (?:organizations|institutions|studies|researchers|factors|challenges)\b", s, re.I):
            if not NUMBER_RE.search(s) and not YEAR_RE.search(s) and not CITATION_RE.search(s):
                abstract_candidates.append(s)
    if abstract_candidates:
        _add(ev, f"Unanchored abstract claim: “{_quote(abstract_candidates[0])}”.", "moderate")
    generic_examples = GENERIC_EXAMPLE_RE.findall(raw)
    if generic_examples:
        _add(ev, f"Canonical generic example(s) used without clear context: {', '.join(sorted(set(generic_examples)))}.", "weak")
    score = _score(ev)
    signals.append({"key": "E", "name": "Specificity", "score": score, "evidence": ev, "summary": "Concrete names, actors, numbers, dates and examples."})

    # F. Transition fingerprint
    ev = []
    strong_hits: list[str] = []
    moderate_hits: list[str] = []
    for pattern in TRANSITION_STRONG:
        strong_hits.extend(m.group(0) for m in re.finditer(pattern, raw, re.I | re.M))
    for pattern in TRANSITION_MODERATE:
        moderate_hits.extend(m.group(0) for m in re.finditer(pattern, raw, re.I | re.M))
    if strong_hits:
        _add(ev, f"Strong formulaic transition(s): {', '.join('“'+_quote(x,45)+'”' for x in strong_hits[:4])}.", "strong" if len(strong_hits) >= 2 else "moderate")
    if moderate_hits:
        _add(ev, f"Formulaic pivot/announcement(s): {', '.join('“'+_quote(x,45)+'”' for x in moderate_hits[:4])}.", "moderate" if len(moderate_hits) >= 2 else "weak")
    however_count = len(re.findall(r"(?:^|\n)\s*However,", raw, re.I))
    if wc and however_count > max(1, wc // 200):
        _add(ev, f"'However,' opens {however_count} sentence/paragraph(s), above the repeated-transition threshold.", "moderate")
    score = _score(ev)
    signals.append({"key": "F", "name": "Transitions", "score": score, "evidence": ev, "summary": "Mechanical connective tissue and reveal/announcement phrasing."})

    # G. Punctuation fingerprint, academic calibration
    ev = []
    em_count = raw.count("—")
    if em_count > max(1, wc // 300):
        _add(ev, f"Em dashes appear {em_count} time(s), above the length-adjusted threshold.", "moderate")
    if re.search(r"—[^—\n]{1,120}—", raw):
        _add(ev, "Double em-dash wrapping appears as a dramatic aside.", "strong")
    semicolons = raw.count(";")
    if not academic and semicolons:
        _add(ev, f"Semicolons linking clauses appear {semicolons} time(s) in non-academic prose.", "weak" if semicolons == 1 else "moderate")
    elif academic and semicolons >= max(4, wc // 180):
        _add(ev, f"Semicolon use is unusually dense even for academic prose ({semicolons} instances).", "weak")
    colon_hits = re.findall(r"\b(?:The problem|The answer|The rule|The key insight|The approach):", raw, re.I)
    if colon_hits:
        _add(ev, f"Announcement-colon construction: “{colon_hits[0]}”.", "moderate")
    score = _score(ev)
    signals.append({"key": "G", "name": "Punctuation", "score": score, "evidence": ev, "summary": "Em-dash, semicolon and announcement-colon patterns."})

    # H. Voice / register, heavily calibrated for scholarly prose
    ev = []
    informal = bool(INFORMAL_MARKER_RE.search(raw))
    complete_ratio = sum(1 for s in sentences if re.search(r"[.!?]$", s)) / max(1, len(sentences))
    if informal and len(sentences) >= 4 and complete_ratio > 0.9:
        _add(ev, "Informal markers are layered over consistently polished complete sentences (register collapse).", "strong")
    if re.search(r"\b(?:Happy to jump on a call if that(?:'|’)s easier|Let me know if you have any questions|Feel free to reach out)\b", raw, re.I):
        _add(ev, "Templated professional closer detected.", "weak")
    # In academic prose, absence of first/second person is normal, so only flag a cluster of missing human traces weakly.
    traces = sum(bool(rx.search(raw)) for rx in (FIRST_PERSON_RE, SECOND_PERSON_RE, RHETORICAL_Q_RE, SELF_CORRECTION_RE))
    if wc >= 300 and traces == 0 and not academic:
        _add(ev, "Long passage contains no first/second person, rhetorical question, or self-correction trace.", "moderate")
    elif wc >= 450 and traces == 0 and academic:
        _add(ev, "Long passage maintains a perfectly neutral register with no local voice variation; this is weak evidence in scholarly prose.", "weak")
    score = _score(ev)
    signals.append({"key": "H", "name": "Voice / register", "score": score, "evidence": ev, "summary": "Human traces, register variation and templated professional voice."})

    # I. Rhetorical scaffolding
    ev = []
    patterns = [
        (SETUP_RE, "Setup/revelation announcement sentence", "moderate"),
        (MORE_THAN_RE, "'More X than Y' comparative framing", "moderate"),
        (DIMINISHMENT_RE, "'Not X but Y' diminishment framing", "moderate"),
        (TURNS_OUT_RE, "'Turns out' reveal pivot", "moderate"),
        (BINARY_RE, "Clean binary 'either/or' or 'between/and' framing", "moderate"),
        (ACTUAL_WORK_RE, "'is the actual/real work' landing phrase", "moderate"),
        (PATTERN_ANNOUNCEMENT_RE, "Pattern/insight announcement frame", "moderate"),
        (PARTICIPIAL_REFRAME_RE, "Participial reframe pivot", "moderate"),
        (COMPOSED_PARENT_RE, "Composed self-aware parenthetical", "moderate"),
        (BALANCED_PAREN_RE, "Balanced parenthetical pair", "moderate"),
    ]
    for regex, label, severity in patterns:
        match = regex.search(raw)
        if match:
            containing = next((s for s in sentences if match.group(0).lower() in s.lower()), match.group(0))
            _add(ev, f"{label}: “{_quote(containing)}”.", severity)
    if sentences and THESIS_FIRST_RE.search(sentences[0]):
        _add(ev, f"Thesis-first opener: “{_quote(sentences[0])}”.", "moderate")
    for opener, count in _anaphora(sentences):
        _add(ev, f"Repeated sentence starter “{opener}” occurs {count} times consecutively.", "moderate")
        break
    question_list = PARALLEL_QUESTION_LIST_RE.search(raw)
    if question_list:
        _add(ev, f"Within-sentence anaphoric parallel list: “{_quote(question_list.group(0))}”.", "strong" if not academic else "moderate")
    reason_run = _parallel_reason_run(sentences)
    if reason_run >= 3:
        _add(ev, f"Parallel reason-chain structure runs across {reason_run} consecutive sentences.", "moderate")
    if len(sentences) >= 8 and sent_cv < 0.28 and len(paragraphs) >= 3 and para_cv < 0.28:
        _add(ev, "Local coherence/rhythm is unusually smooth across both sentence and paragraph scales.", "moderate")
    short_closers = [sentence for sentence in sentences if 4 <= len(_words(sentence)) <= 7 and APOSTROPHE_CLOSER_RE.match(sentence)]
    if short_closers:
        _add(ev, f"Mini-aphorism closer pattern: “{_quote(short_closers[-1])}”.", "weak" if academic else "moderate")
    score = _score(ev)
    signals.append({"key": "I", "name": "Rhetorical scaffolding", "score": score, "evidence": ev, "summary": "Over-composed rhetorical devices and smoothness patterns."})

    scores = [int(signal["score"]) for signal in signals]
    total = sum(scores)
    paragraph_ai_pct, paragraph_risks = _paragraph_ai_profile(paragraphs, academic=academic)

    # Keep the 0–27 category score for the verdict, but use a finer-grained
    # signal index for the percentage shown in the dashboard. This avoids a
    # staircase effect where several edits can remove evidence without moving
    # a category from 2/3 to 1/3. Evidence density makes those improvements
    # visible while corroboration across categories remains the main signal.
    category_pct = round(total / 27 * 100)
    evidence_load = 0
    for signal in signals:
        points = sum(SEVERITY_WEIGHT.get(item.get("severity", "weak"), 1) for item in signal.get("evidence", []))
        evidence_load += min(6, points)
    evidence_pct = round(evidence_load / (9 * 6) * 100)
    ai_pct = round(category_pct * 0.72 + evidence_pct * 0.18 + paragraph_ai_pct * 0.10)
    if total >= 14:
        ai_pct = max(50, ai_pct)
    elif total <= 4:
        ai_pct = min(24, ai_pct)
    ai_pct = max(0, min(100, ai_pct))
    verdict = _verdict(total)
    confidence = _confidence(raw, total, scores)
    fraction = _fraction(total, scores)
    active_paragraphs = sum(1 for risk in paragraph_risks if risk >= 25)
    if paragraph_risks:
        share = active_paragraphs / len(paragraph_risks)
        if share >= 0.85 and total >= 14:
            fraction = "Heavily AI-edited (~60–90%)" if total < 20 else "Pure AI (~100%)"
        elif share >= 0.45 and total >= 9:
            fraction = "Mixed authorship (~30–60%)"
        elif 0 < share < 0.45 and total >= 5:
            fraction = "Lightly AI-assisted (~10–30%)"

    calibration_notes = [
        "The percentage is an AI-style signal score, not a calibrated probability that a machine wrote the text.",
        "This is a stylistic AI-signal detector, not proof of authorship. Strong verdicts require corroboration across categories.",
        "Academic writing is calibrated to reduce penalties for legitimate hedging, neutral register, lists, tricolons and semicolon use.",
    ]
    if wc < 100:
        calibration_notes.append("Short text under 100 words has fewer detectable signals; confidence is capped at Medium.")
    if wc < 40:
        calibration_notes.append("Very short text is especially unreliable for authorship inference.")

    # Strongest signals in plain language.
    ranked = sorted(signals, key=lambda x: (x["score"], len(x["evidence"])), reverse=True)
    strongest = [s for s in ranked if s["score"] > 0][:3]
    if strongest:
        gave_away = " The strongest signals were " + ", ".join(f"{s['name']} ({s['score']}/3)" for s in strongest) + "."
    else:
        gave_away = " No category produced a meaningful AI-style signal."

    return {
        "verdict": verdict,
        "confidence": confidence,
        "overall_score": total,
        "max_score": 27,
        "ai_detection_percentage": ai_pct,
        "ai_edited_fraction": fraction,
        "signals": signals,
        "what_gave_it_away": gave_away.strip(),
        "calibration_notes": calibration_notes,
        "word_count": wc,
        "sentence_lengths": lengths,
        "category_signal_percentage": category_pct,
        "evidence_signal_percentage": evidence_pct,
        "paragraph_ai_signal_percentage": paragraph_ai_pct,
        "paragraph_signal_profile": paragraph_risks,
    }
