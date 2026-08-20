from __future__ import annotations

import hashlib
import io
import re
import zipfile
from typing import Any

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS, "m": M_NS, "a": A_NS}


def _q(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


W_P = _q(W_NS, "p")
W_TBL = _q(W_NS, "tbl")
W_T = _q(W_NS, "t")
W_TAB = _q(W_NS, "tab")
W_BR = _q(W_NS, "br")
W_R = _q(W_NS, "r")
W_RPR = _q(W_NS, "rPr")
W_PPR = _q(W_NS, "pPr")
W_PSTYLE = _q(W_NS, "pStyle")
W_VAL = _q(W_NS, "val")

_REFERENCE_HEADING = re.compile(r"^(?:references|works\s+cited|bibliography)\s*$", re.I)
_APPENDIX_HEADING = re.compile(r"^appendix(?:\s+[A-Z0-9.-]+)?(?:\s*[-:–—].*)?$", re.I)
_CAPTION = re.compile(r"^(?:table|figure)\s+(?:\d+|[A-Z])(?:[.:\s]|$)", re.I)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+){0,4}\.?\s+\S+")

# These elements carry document semantics or non-trivial inline formatting that
# must never be reconstructed from plain text. Paragraphs containing them are
# copied byte-for-byte at the OOXML level by the format-preserving workflow.
_SPECIAL_PARAGRAPH_XPATHS = (
    ".//w:fldChar",
    ".//w:instrText",
    ".//w:hyperlink",
    ".//w:drawing",
    ".//w:object",
    ".//m:oMath",
    ".//m:oMathPara",
    ".//w:footnoteReference",
    ".//w:endnoteReference",
    ".//w:commentReference",
    ".//w:commentRangeStart",
    ".//w:commentRangeEnd",
    ".//w:bookmarkStart",
    ".//w:bookmarkEnd",
    ".//w:ins",
    ".//w:del",
    ".//w:moveFrom",
    ".//w:moveTo",
    ".//w:sdt",
    ".//w:tab",
    ".//w:br",
)


def _paragraph_text(paragraph: etree._Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == W_T and node.text:
            parts.append(node.text)
        elif node.tag == W_TAB:
            parts.append("\t")
        elif node.tag == W_BR:
            parts.append("\n")
    return "".join(parts)


def _style_id(paragraph: etree._Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", namespaces=NS)
    if style is None:
        return ""
    return str(style.get(W_VAL) or "")


def _run_style_signature(run: etree._Element) -> bytes:
    rpr = run.find("./w:rPr", namespaces=NS)
    if rpr is None:
        return b""
    return etree.tostring(rpr, method="c14n", exclusive=False, with_comments=False)


def _lock_reason(paragraph: etree._Element, text: str, *, in_reference_tail: bool = False, in_appendix: bool = False) -> str | None:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return "blank paragraph"
    style = _style_id(paragraph)
    style_lower = style.lower()
    if style_lower.startswith(("heading", "title", "subtitle", "caption", "toc", "bibliography")):
        return f"Word style {style or 'structural'}"
    if _REFERENCE_HEADING.match(clean) or in_reference_tail:
        return "reference list"
    if _APPENDIX_HEADING.match(clean) or in_appendix:
        return "appendix structure"
    if _CAPTION.match(clean):
        return "table/figure caption"
    if len(clean.split()) <= 16 and _NUMBERED_HEADING.match(clean) and not re.search(r"[.!?]$", clean):
        return "numbered heading"
    for xpath in _SPECIAL_PARAGRAPH_XPATHS:
        if paragraph.xpath(xpath, namespaces=NS):
            return "field, link, equation, object, note or manual break"

    # Preserve paragraphs with mixed run formatting exactly. Rewriting those
    # safely would require a span-level semantic alignment between the revised
    # prose and each bold/italic/superscript run. Simple/uniform paragraphs can
    # be patched without changing their visible formatting.
    signatures: set[bytes] = set()
    for run in paragraph.findall("./w:r", namespaces=NS):
        if any((node.text or "") for node in run.findall(".//w:t", namespaces=NS)):
            signatures.add(_run_style_signature(run))
    if len(signatures) > 1:
        return "mixed inline formatting"
    return None


def _table_display(table: etree._Element) -> str:
    rows: list[str] = []
    for row in table.findall("./w:tr", namespaces=NS):
        cells: list[str] = []
        for cell in row.findall("./w:tc", namespaces=NS):
            paragraphs = [re.sub(r"\s+", " ", _paragraph_text(p)).strip() for p in cell.findall(".//w:p", namespaces=NS)]
            cells.append(" ".join(p for p in paragraphs if p))
        rows.append("\t".join(cells))
    return "\n".join(rows).strip()


def inspect_docx(content: bytes) -> dict[str, Any]:
    """Read a DOCX without flattening away its document object structure.

    Only top-level body paragraphs are candidates for rewriting. Tables and all
    non-textual OOXML objects remain locked in the original package.
    """
    with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
        try:
            xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("The DOCX does not contain a valid Word document.xml part.") from exc

    root = etree.fromstring(xml)
    body = root.find("./w:body", namespaces=NS)
    if body is None:
        raise ValueError("The DOCX body could not be read.")

    paragraphs: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    paragraph_index = 0
    table_index = 0
    in_reference_tail = False
    in_appendix = False

    for child in body:
        if child.tag == W_P:
            text = _paragraph_text(child)
            clean = re.sub(r"\s+", " ", text).strip()
            if _APPENDIX_HEADING.match(clean):
                in_reference_tail = False
                in_appendix = True
            if _REFERENCE_HEADING.match(clean):
                in_reference_tail = True
                in_appendix = False
            reason = _lock_reason(child, text, in_reference_tail=in_reference_tail, in_appendix=in_appendix)
            entry = {
                "paragraph_index": paragraph_index,
                "text": text,
                "editable": reason is None,
                "lock_reason": reason or "",
                "style_id": _style_id(child),
                "word_count": len(text.split()),
            }
            paragraphs.append(entry)
            blocks.append({"type": "paragraph", "paragraph_index": paragraph_index})
            paragraph_index += 1
        elif child.tag == W_TBL:
            table_text = _table_display(child)
            blocks.append({"type": "table", "table_index": table_index, "text": table_text})
            table_index += 1
        else:
            # sectPr and other structural body children are preserved in the
            # package and need no plain-text representation.
            continue

    text = render_structured_text({"blocks": blocks, "paragraphs": paragraphs}, {})
    locked = [p for p in paragraphs if not p["editable"]]
    editable = [p for p in paragraphs if p["editable"]]
    return {
        "text": text,
        "paragraphs": paragraphs,
        "blocks": blocks,
        "paragraph_count": len(paragraphs),
        "table_count": table_index,
        "editable_paragraphs": len(editable),
        "locked_paragraphs": len(locked),
        "locked_reason_counts": _count_reasons(locked),
        "source_digest": text_digest(text),
        "signature": docx_structure_signature(content),
    }


def _count_reasons(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        reason = str(item.get("lock_reason") or "locked")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def render_structured_text(structure: dict[str, Any], replacements: dict[int, str] | dict[str, str]) -> str:
    paragraphs = {int(p["paragraph_index"]): p for p in structure.get("paragraphs", [])}
    blocks: list[str] = []
    for block in structure.get("blocks", []):
        if block.get("type") == "paragraph":
            idx = int(block["paragraph_index"])
            value = replacements.get(idx, replacements.get(str(idx), paragraphs.get(idx, {}).get("text", "")))
            if str(value or "").strip():
                blocks.append(str(value))
        elif block.get("type") == "table":
            value = str(block.get("text") or "")
            if value.strip():
                blocks.append(value)
    return "\n\n".join(blocks).strip()


def text_digest(text: str) -> str:
    normalised = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _set_text_node(node: etree._Element, value: str) -> None:
    node.text = value
    if value[:1].isspace() or value[-1:].isspace():
        node.set(_q(XML_NS, "space"), "preserve")
    else:
        node.attrib.pop(_q(XML_NS, "space"), None)


def _replace_simple_paragraph_text(paragraph: etree._Element, new_text: str) -> None:
    text_nodes = paragraph.findall(".//w:t", namespaces=NS)
    if not text_nodes:
        # Editable paragraphs should normally have text nodes. This fallback
        # retains paragraph properties and creates one ordinary text run.
        run = etree.SubElement(paragraph, W_R)
        node = etree.SubElement(run, W_T)
        _set_text_node(node, new_text)
        return
    _set_text_node(text_nodes[0], new_text)
    for node in text_nodes[1:]:
        _set_text_node(node, "")


def patch_docx(content: bytes, structure: dict[str, Any], replacements: dict[int, str]) -> bytes:
    editable = {int(p["paragraph_index"]): p for p in structure.get("paragraphs", []) if p.get("editable")}
    illegal = sorted(set(int(k) for k in replacements) - set(editable))
    if illegal:
        raise ValueError(f"Attempted to modify locked Word paragraph(s): {illegal[:8]}")

    source = io.BytesIO(content)
    output = io.BytesIO()
    with zipfile.ZipFile(source, "r") as src:
        xml = src.read("word/document.xml")
        root = etree.fromstring(xml)
        body = root.find("./w:body", namespaces=NS)
        if body is None:
            raise ValueError("The DOCX body could not be patched.")
        paragraph_index = 0
        for child in body:
            if child.tag != W_P:
                continue
            if paragraph_index in replacements:
                _replace_simple_paragraph_text(child, str(replacements[paragraph_index]))
            paragraph_index += 1
        patched_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

        with zipfile.ZipFile(output, "w") as dst:
            for info in src.infolist():
                data = patched_xml if info.filename == "word/document.xml" else src.read(info.filename)
                dst.writestr(info, data)
    return output.getvalue()


def _part_hashes(archive: zipfile.ZipFile, names: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in names:
        try:
            hashes[name] = hashlib.sha256(archive.read(name)).hexdigest()
        except KeyError:
            continue
    return hashes


def docx_structure_signature(content: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(content), "r") as archive:
        xml = archive.read("word/document.xml")
        root = etree.fromstring(xml)
        body = root.find("./w:body", namespaces=NS)
        tables = root.findall(".//w:tbl", namespaces=NS)
        table_hashes = [hashlib.sha256(etree.tostring(table, method="c14n", exclusive=False)).hexdigest() for table in tables]
        names = archive.namelist()
        protected_parts = [
            name for name in names
            if name.startswith(("word/header", "word/footer", "word/media/"))
            or name in {
                "word/styles.xml", "word/numbering.xml", "word/footnotes.xml", "word/endnotes.xml",
                "word/comments.xml", "word/settings.xml", "word/fontTable.xml", "word/theme/theme1.xml",
            }
        ]
        return {
            "body_paragraphs": sum(1 for child in body if child.tag == W_P) if body is not None else 0,
            "tables": len(tables),
            "table_rows": len(root.findall(".//w:tr", namespaces=NS)),
            "table_cells": len(root.findall(".//w:tc", namespaces=NS)),
            "sections": len(root.findall(".//w:sectPr", namespaces=NS)),
            "drawings": len(root.findall(".//w:drawing", namespaces=NS)),
            "equations": len(root.findall(".//m:oMath", namespaces=NS)) + len(root.findall(".//m:oMathPara", namespaces=NS)),
            "hyperlinks": len(root.findall(".//w:hyperlink", namespaces=NS)),
            "fields": len(root.findall(".//w:fldChar", namespaces=NS)),
            "page_breaks": len(root.xpath(".//w:br[@w:type='page']", namespaces=NS)),
            "table_hashes": table_hashes,
            "protected_parts": _part_hashes(archive, protected_parts),
            "media_parts": sorted(name for name in names if name.startswith("word/media/")),
            "header_parts": sorted(name for name in names if name.startswith("word/header")),
            "footer_parts": sorted(name for name in names if name.startswith("word/footer")),
        }


def format_preservation_certificate(original: bytes, patched: bytes, *, changed_paragraphs: int = 0, locked_paragraphs: int = 0) -> dict[str, Any]:
    before = docx_structure_signature(original)
    after = docx_structure_signature(patched)
    checks = {
        "word_paragraph_structure": before["body_paragraphs"] == after["body_paragraphs"],
        "word_tables": before["tables"] == after["tables"] and before["table_rows"] == after["table_rows"] and before["table_cells"] == after["table_cells"] and before["table_hashes"] == after["table_hashes"],
        "word_sections": before["sections"] == after["sections"],
        "word_figures_drawings": before["drawings"] == after["drawings"] and before["media_parts"] == after["media_parts"],
        "word_equations": before["equations"] == after["equations"],
        "word_hyperlinks_fields": before["hyperlinks"] == after["hyperlinks"] and before["fields"] == after["fields"],
        "word_headers_footers": before["header_parts"] == after["header_parts"] and before["footer_parts"] == after["footer_parts"],
        "word_page_breaks": before["page_breaks"] == after["page_breaks"],
        "word_styles_numbering_notes": before["protected_parts"] == after["protected_parts"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "changed_paragraphs": int(changed_paragraphs),
        "locked_paragraphs": int(locked_paragraphs),
        "note": "The humanized DOCX was patched into the original Word package. Tables and document structure were not rebuilt from extracted text.",
    }
