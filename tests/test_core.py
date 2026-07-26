from __future__ import annotations

import unittest

from scholarly_humanizer import humanize_scholarly_text, validate_humanizer_preservation
from services.analyzer import dashboard_report
from services.document_io import build_docx


SAMPLE = """1.1 Background to the Study

It is important to note that the present study plays a crucial role in examining various factors. Moreover, the study considers evidence from 2024 (Adam, 2024). Moreover, the study also considers institutional context. Moreover, the study explains the issue in a sentence that is deliberately extended with many connected clauses, repeated claims, and several additional qualifications so that the diagnostic can identify excessive sentence length without changing the cited evidence or the date."""


class ScholarlyHumanizerTests(unittest.TestCase):
    def test_dashboard_has_segments(self) -> None:
        report = dashboard_report(SAMPLE)
        self.assertGreater(report["style_concern_percentage"], 0)
        self.assertTrue(report["segments"])
        self.assertIn("risk-segment", report["highlighted_html"])

    def test_local_humanizer_preserves_evidence(self) -> None:
        revised, report = humanize_scholarly_text(SAMPLE, "balanced")
        valid, issues = validate_humanizer_preservation(SAMPLE, revised, max_word_change_ratio=0.55)
        self.assertTrue(valid, issues)
        self.assertIn("2024", revised)
        self.assertIn("(Adam, 2024)", revised)
        self.assertTrue(report["preservation_passed"])

    def test_docx_export(self) -> None:
        content = build_docx(SAMPLE)
        self.assertGreater(len(content), 1000)
        self.assertTrue(content.startswith(b"PK"))


if __name__ == "__main__":
    unittest.main()
