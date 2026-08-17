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
        self.assertEqual(report["human_like_style_percentage"], 100 - report["ai_detection_percentage"])

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
        self.assertIn('/static/app.js?v=1.7.0', html)
        self.assertIn('/static/style.css?v=1.7.0', html)
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
        self.assertIn('id="humanLikeScore"', html)
        self.assertNotIn('>Naturalness<', html)
        self.assertIn('data-tab="detector"', html)
        self.assertIn('renderDetector(detector)', js)

    def test_local_humanizer_never_reduces_naturalness(self) -> None:
        before = dashboard_report(SAMPLE)["naturalness_percentage"]
        revised, report = humanize_scholarly_text(SAMPLE, "deep")
        after = dashboard_report(revised)["naturalness_percentage"]
        self.assertGreaterEqual(after, before)
        self.assertEqual(report["naturalness_gain"], after - before)


    def test_humanizer_recomputes_and_can_reduce_ai_signal_index(self) -> None:
        before = dashboard_report(SAMPLE)
        revised, _ = humanize_scholarly_text(SAMPLE, "deep")
        after = dashboard_report(revised)
        self.assertGreater(after["naturalness_percentage"], before["naturalness_percentage"])
        self.assertLess(after["ai_detection_percentage"], before["ai_detection_percentage"])

    def test_render_blueprint_preconfigures_engine2_openai(self) -> None:
        root = Path(__file__).resolve().parents[1]
        render = (root / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("HUMANIZER_PROVIDER\n        value: openai", render)
        self.assertIn("OPENAI_MODEL\n        value: gpt-5.6-terra", render)
        self.assertIn("OPENAI_API_KEY\n        sync: false", render)

    def test_frontend_shows_complementary_ai_and_humanlike_meters(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="aiScoreBar"', html)
        self.assertIn('id="humanLikeScoreBar"', html)
        self.assertIn('id="aiGain"', html)
        self.assertIn('ai_signal_improvement', js)

    def test_humanlike_gain_is_visible_in_frontend(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="humanLikeGain"', html)
        self.assertIn('human_like_style_improvement', js)
        self.assertNotIn("$('naturalScore')", js)

    def test_humanize_endpoint_never_lowers_humanlike_style(self) -> None:
        from app import HumanizeRequest, humanize
        data = humanize(HumanizeRequest(text=SAMPLE, mode="deep", engine="engine1"))
        self.assertGreaterEqual(
            data["report"]["human_like_style_percentage"],
            data["original_report"]["human_like_style_percentage"],
        )

    def test_humanlike_style_is_exact_complement_after_humanize(self) -> None:
        before = dashboard_report(SAMPLE)
        revised, _ = humanize_scholarly_text(SAMPLE, "deep")
        after = dashboard_report(revised)
        self.assertEqual(before["human_like_style_percentage"], 100 - before["ai_detection_percentage"])
        self.assertEqual(after["human_like_style_percentage"], 100 - after["ai_detection_percentage"])

    def test_engine2_option_is_never_disabled_by_configuration_status(self) -> None:
        root = Path(__file__).resolve().parents[1]
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("opt.disabled = !engine2?.configured", js)
        self.assertIn("opt.disabled = false", js)

    def test_frontend_allows_terra_or_luna_for_engine2(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "templates" / "index.html").read_text(encoding="utf-8")
        js = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="engine2Model"', html)
        self.assertIn('value="gpt-5.6-terra"', html)
        self.assertIn('value="gpt-5.6-luna"', html)
        self.assertIn("engine2_model", js)

    def test_humanize_request_accepts_engine2_model_choice(self) -> None:
        from app import HumanizeRequest
        terra = HumanizeRequest(text=SAMPLE, engine="engine2", engine2_model="gpt-5.6-terra")
        luna = HumanizeRequest(text=SAMPLE, engine="engine2", engine2_model="gpt-5.6-luna")
        self.assertEqual(terra.engine2_model, "gpt-5.6-terra")
        self.assertEqual(luna.engine2_model, "gpt-5.6-luna")

    def test_dashboard_explanatory_counts_match_detector_basis(self) -> None:
        report = dashboard_report(SAMPLE)
        self.assertEqual(report["active_signal_categories"], sum(1 for item in report["ai_signal_breakdown"] if item["score"] > 0))
        self.assertEqual(report["signal_evidence_items"], sum(len(item.get("evidence", [])) for item in report["ai_signal_breakdown"]))
        self.assertGreaterEqual(report["flagged_sentence_count"], 0)

    def test_academic_technical_terms_do_not_create_false_ai_signal(self) -> None:
        academic = (
            "Both ordinary least squares with Newey-West inference and a Huber robust regression "
            "produced similar coefficients. The strongest absolute correlation is between SMB and RMW. "
            "The result provides a numerical robustness check rather than a separate economic sensitivity test."
        )
        report = dashboard_report(academic)
        by_name = {item["name"]: item for item in report["ai_signal_breakdown"]}
        self.assertEqual(by_name["Perplexity"]["score"], 0)
        self.assertEqual(by_name["Rhetorical scaffolding"]["score"], 0)

    def test_zero_flagged_sentences_cannot_show_high_ai_index_without_strong_global_corroboration(self) -> None:
        academic = (
            "Portfolio selection requires a balance between expected return, risk, concentration, and model uncertainty. "
            "Markowitz formalised this trade-off by evaluating portfolios through expected returns and covariance structure. "
            "The analysis compares constrained allocations with an unconstrained benchmark and reports out-of-sample results."
        )
        report = dashboard_report(academic)
        if report["flagged_sentence_count"] == 0 and report["ai_score"] < 14:
            self.assertLessEqual(report["ai_detection_percentage"], 34)

    def test_provider_auto_detects_openai_key_when_provider_not_named(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            config = RefinerConfig.from_env()
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "gpt-5.6-terra")
            self.assertTrue(provider_status(config)["configured"])

    def test_provider_status_identifies_missing_openai_key(self) -> None:
        with patch.dict(os.environ, {"HUMANIZER_PROVIDER": "openai", "OPENAI_MODEL": "gpt-5.6-terra"}, clear=True):
            status = provider_status()
            self.assertFalse(status["configured"])
            self.assertIn("OPENAI_API_KEY", status.get("missing", []))

    def test_engine2_missing_key_uses_explicit_engine1_fallback(self) -> None:
        from app import HumanizeRequest, humanize
        with patch.dict(os.environ, {"HUMANIZER_PROVIDER": "openai", "OPENAI_MODEL": "gpt-5.6-terra"}, clear=True):
            data = humanize(HumanizeRequest(text=SAMPLE, mode="deep", engine="engine2"))
        self.assertEqual(data["actual_engine"], "engine1_fallback")
        self.assertTrue(data["engine_2"].get("fallback_used"))
        self.assertIn("OPENAI_API_KEY", data["engine_2"].get("reason", ""))

    def test_deep_humanizer_makes_safe_formulaic_changes_and_does_not_worsen_ai_index(self) -> None:
        formulaic = (
            "It is important to note that the analysis provides a comprehensive assessment of the evidence. "
            "Moreover, the study considers institutional context. Moreover, the study examines implementation conditions. "
            "Moreover, the study explains the implications for practice and policy while preserving the cited evidence (Adam, 2024)."
        )
        before = dashboard_report(formulaic)
        revised, report = humanize_scholarly_text(formulaic, "deep")
        after = dashboard_report(revised)
        self.assertNotEqual(revised, formulaic)
        self.assertTrue(report["preservation_passed"])
        self.assertLessEqual(after["ai_detection_percentage"], before["ai_detection_percentage"])
        self.assertIn("(Adam, 2024)", revised)



if __name__ == "__main__":
    unittest.main()
