from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scholarly_humanizer import humanize_scholarly_text, validate_humanizer_preservation
from services.analyzer import dashboard_report
from services.document_io import build_docx
from services.model_refiner import RefinerConfig, provider_status


SAMPLE = """1.1 Background to the Study

It is important to note that the present study plays a crucial role in examining various factors. Moreover, the study considers evidence from 2024 (Adam, 2024). Moreover, the study also considers institutional context. Moreover, the study explains the issue in a sentence that is deliberately extended with many connected clauses, repeated claims, and several additional qualifications so that the diagnostic can identify excessive sentence length without changing the cited evidence or the date."""


class ScholarlyHumanizerTests(unittest.TestCase):
    def test_dashboard_has_segments(self) -> None:
        report = dashboard_report(SAMPLE)
        self.assertGreater(report["ai_detection_percentage"], 0)
        self.assertTrue(report["segments"])
        self.assertIn("risk-segment", report["highlighted_html"])
        self.assertEqual(len(report["ai_signal_breakdown"]), 9)
        self.assertIn(report["ai_verdict"], {"Human", "Likely Human", "Uncertain", "Likely AI", "AI"})
        self.assertGreaterEqual(report["naturalness_percentage"], 0)
        self.assertLessEqual(report["naturalness_percentage"], 100)

    def test_local_humanizer_preserves_evidence(self) -> None:
        revised, report = humanize_scholarly_text(SAMPLE, "balanced")
        valid, issues = validate_humanizer_preservation(SAMPLE, revised, max_word_change_ratio=0.55)
        self.assertTrue(valid, issues)
        self.assertIn("2024", revised)
        self.assertIn("(Adam, 2024)", revised)
        self.assertTrue(report["preservation_passed"])
        self.assertEqual(report["engine"], "engine1")

    def test_docx_export(self) -> None:
        content = build_docx(SAMPLE)
        self.assertGreater(len(content), 1000)
        self.assertTrue(content.startswith(b"PK"))

    def test_openai_standard_environment_variables(self) -> None:
        env = {
            "HUMANIZER_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RefinerConfig.from_env()
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.model, "test-model")
            self.assertEqual(config.base_url, "https://api.openai.com/v1")
            self.assertTrue(provider_status(config)["configured"])


    def test_openai_defaults_to_terra(self) -> None:
        env = {
            "HUMANIZER_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RefinerConfig.from_env()
            self.assertEqual(config.model, "gpt-5.6-terra")
            self.assertTrue(provider_status(config)["configured"])

    def test_openai_allows_luna_override(self) -> None:
        env = {
            "HUMANIZER_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "gpt-5.6-luna",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RefinerConfig.from_env()
            self.assertEqual(config.model, "gpt-5.6-luna")

    def test_openai_requires_api_key(self) -> None:
        env = {
            "HUMANIZER_PROVIDER": "openai",
            "OPENAI_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = RefinerConfig.from_env()
            self.assertFalse(provider_status(config)["configured"])

    def test_frontend_cache_bust_and_legacy_checked_guard(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('/static/app.js?v=1.3.1', html)
        self.assertIn('id="useModel"', html)
        self.assertNotIn("$('useModel').checked", js)

    def test_ai_detector_aggressively_corroborates_formulaic_text(self) -> None:
        ai_like = """It is important to note that digital transformation has become increasingly important. Furthermore, organizations often face key challenges in this rapidly evolving landscape. Moreover, it is clear that robust and comprehensive strategies can often lead to significant improvements. Additionally, these approaches facilitate innovation, foster collaboration, and streamline operations. What surprised me was the consistency of the pattern. The key insight is that success is not just about technology, but also about people. Taken together, this highlights the importance of a nuanced and multifaceted approach. In conclusion, organizations should leverage these insights to achieve enduring success."""
        report = dashboard_report(ai_like)
        self.assertGreaterEqual(report["ai_score"], 14)
        self.assertIn(report["ai_verdict"], {"Likely AI", "AI"})
        self.assertGreaterEqual(report["ai_detection_percentage"], 50)

    def test_academic_calibration_does_not_treat_normal_style_as_ai(self) -> None:
        academic = """Public procurement systems shape how public agencies convert budgets into goods and services. In Ghana, procurement entities operate under statutory rules that assign responsibilities to tender committees, evaluation panels, heads of entities and oversight bodies. These arrangements matter because weak compliance can delay projects and increase transaction costs. Prior empirical work reports mixed effects of digital procurement on competition and disclosure, partly because implementation quality differs across agencies (Mensah, 2024). This study therefore examines whether system integration is associated with procurement transparency while controlling for institutional capacity."""
        report = dashboard_report(academic)
        self.assertLess(report["ai_score"], 9)
        self.assertIn(report["ai_verdict"], {"Human", "Likely Human"})

    def test_dashboard_is_ai_detector_first(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('<h2>AI Detector</h2>', html)
        self.assertIn('id="aiScore"', html)
        self.assertIn('id="naturalScore"', html)
        self.assertIn('data-tab="detector"', html)
        self.assertIn('renderDetector(detector)', js)

    def test_local_humanizer_never_reduces_naturalness(self) -> None:
        before = dashboard_report(SAMPLE)["naturalness_percentage"]
        revised, report = humanize_scholarly_text(SAMPLE, "deep")
        after = dashboard_report(revised)["naturalness_percentage"]
        self.assertGreaterEqual(after, before)
        self.assertEqual(report["naturalness_gain"], after - before)

    def test_naturalness_gain_is_visible_in_frontend(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="naturalGain"', html)
        self.assertIn('naturalness_improvement', js)


if __name__ == "__main__":
    unittest.main()
