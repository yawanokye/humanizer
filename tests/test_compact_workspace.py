from __future__ import annotations

import asyncio
import io
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from docx import Document
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app import HumanizeRequest, ExportRequest, export_docx, humanize, upload
from scholarly_humanizer import humanize_scholarly_text, humanize_signal_guided, validate_humanizer_preservation
from services.analyzer import dashboard_report
from services.model_refiner import RefinerConfig, refine_paragraphs_with_model


SOURCE = "We conducted an analysis of daily returns from 2024 (Adam, 2024). The association may depend on the sample."


class LayoutParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.nodes = {}
        self.ids = []

    def handle_starttag(self, tag, attrs):
        node = dict(attrs)
        if node.get("id"):
            self.ids.append(node["id"])
            self.nodes[node["id"]] = (node, list(self.stack))
        if tag not in {"input", "link", "meta", "br", "hr", "img"}:
            self.stack.append((tag, node))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                return


class CompactWorkspaceTests(unittest.TestCase):
    def test_both_editors_are_visible_outside_collapsed_diagnostics(self):
        parser = LayoutParser()
        parser.feed((Path(__file__).resolve().parents[1] / "templates/index.html").read_text())
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "DOM IDs must remain unique")
        for editor in ("sourceText", "revisedText"):
            node, parents = parser.nodes[editor]
            self.assertIn("aria-label", node)
            self.assertTrue(any("editor-grid" in parent.get("class", "").split() for _, parent in parents))
            self.assertFalse(any(parent.get("id") == "analysisDetails" for _, parent in parents))
            self.assertFalse(any("panel" in parent.get("class", "").split() for _, parent in parents))
        details, _ = parser.nodes["analysisDetails"]
        self.assertNotIn("open", details)
        for control in ("humanizeBtn", "analyseBtn", "copyBtn", "docxBtn", "preserveTerms", "analyseResultBtn"):
            self.assertIn(control, parser.nodes)

    def test_frozen_phrase_survives_local_editing_while_other_prose_improves(self):
        source = "It is important to note that the present study made use of daily returns to examine the sample. The present study considers the evidence from 2024."
        revised, _ = humanize_scholarly_text(source, "deep", preserve_terms=["the present study"])
        self.assertIn("the present study", revised)
        self.assertNotIn("It is important to note that", revised)
        valid, issues = validate_humanizer_preservation(source, revised, max_word_change_ratio=0.6, preserve_terms=["the present study"])
        self.assertTrue(valid, issues)

    def test_focused_editor_respects_frozen_words(self):
        source = "It is important to note that the study made use of various factors. Moreover, the study explores the institutional landscape."
        report = dashboard_report(source)
        revised, _ = humanize_signal_guided(source, report["ai_detector"], "deep", segments=report["segments"], preserve_terms=["made use of"])
        self.assertIn("made use of", revised)

    def test_contextual_rewriter_receives_terms_and_rejects_a_term_change(self):
        config = RefinerConfig(provider="openai", model="gpt-5.6-terra", base_url="https://example.invalid", api_key="test")
        source = "The equilibrium prior anchors the allocation. The comparison may change with the sample."
        def unsafe(text, *_args, **kwargs):
            self.assertIn('"equilibrium prior"', kwargs["context_before"])
            return text.replace("equilibrium prior", "equilibrium baseline")
        with patch("services.model_refiner._refine_openai", side_effect=unsafe):
            revised, report = refine_paragraphs_with_model([{"paragraph_index": 0, "text": source}], mode="deep", config=config, preserve_terms=["equilibrium prior"])
        self.assertEqual(revised[0], source)
        self.assertEqual(report["batches"][0]["rejected_paragraphs"], 1)

    def test_plain_text_workflow_reports_protected_terms(self):
        result = humanize(HumanizeRequest(text=SOURCE, mode="deep", preserve_terms=["conducted an analysis of"]))
        self.assertIn("conducted an analysis of", result["text"])
        self.assertEqual(result["protected_terms"], ["conducted an analysis of"])
        self.assertTrue(result["preservation_certificate"]["checks"]["user_terms"])

    def test_oversized_protected_term_is_rejected(self):
        with self.assertRaises(HTTPException) as context:
            humanize(HumanizeRequest(text=SOURCE, preserve_terms=["x" * 121]))
        self.assertEqual(context.exception.status_code, 422)

    def test_word_export_keeps_checked_layout_and_rejects_stale_text(self):
        document = Document()
        document.sections[0].header.paragraphs[0].text = "Protected header"
        document.add_paragraph(SOURCE)
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "MRK"
        table.cell(0, 1).text = "18%"
        stream = io.BytesIO()
        document.save(stream)
        uploaded = asyncio.run(upload(UploadFile(file=io.BytesIO(stream.getvalue()), filename="source.docx")))
        result = humanize(HumanizeRequest(text=uploaded["text"], document_id=uploaded["document_id"], mode="deep", preserve_terms=["conducted an analysis of"]))
        self.assertIn("conducted an analysis of", result["text"])
        self.assertTrue(result["preservation_certificate"]["checks"]["user_terms"])
        response = export_docx(ExportRequest(text=result["text"], document_id=result["document_id"], humanize_job_id=result["humanize_job_id"]))
        checked = Document(io.BytesIO(response.body))
        self.assertEqual(checked.tables[0].cell(0, 1).text, "18%")
        self.assertEqual(checked.sections[0].header.paragraphs[0].text, "Protected header")
        edited = result["text"] + "\nThis sentence was added manually."
        with self.assertRaises(HTTPException) as context:
            export_docx(ExportRequest(text=edited, document_id=result["document_id"], humanize_job_id=result["humanize_job_id"]))
        self.assertEqual(context.exception.status_code, 409)
        plain = export_docx(ExportRequest(text=edited))
        self.assertIn("This sentence was added manually.", "\n".join(p.text for p in Document(io.BytesIO(plain.body)).paragraphs))


if __name__ == "__main__":
    unittest.main()
