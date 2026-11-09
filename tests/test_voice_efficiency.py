from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scholarly_humanizer as voice
from services.ai_detector import ai_check_report, sentence_ai_signal
from services.analyzer import dashboard_report
from services.calibration import FEATURE_SCHEMA_VERSION, load_model
from services.model_refiner import RefinerConfig, refine_paragraphs_with_model


CLEAN = (
    "The capped portfolio returned 11.60% during the test quarter. Its maximum drawdown was 2.92%. "
    "These estimates describe this sample and may differ in another period."
)
FORMULAIC = (
    "It is important to note that institutions face various factors in this rapidly evolving landscape. "
    "Furthermore, organizations leverage a multifaceted approach to foster enduring improvements. "
    "Moreover, the key insight is that progress is not just about technology but also about people."
)


class NaturalVoiceEfficiencyTests(unittest.TestCase):
    def test_direct_actor_construction_retains_results_and_qualifications(self):
        source = "We conducted an analysis of daily returns from 2024 (Adam, 2024). The association may depend on the sample."
        revised, report = voice.humanize_scholarly_text(source, "deep")
        self.assertIn("We analysed daily returns", revised)
        self.assertIn("may depend", revised)
        self.assertIn("(Adam, 2024)", revised)
        self.assertTrue(report["preservation_passed"])

    def test_conditional_scope_and_intention_are_not_converted_to_facts(self):
        source = "Insofar as the model assumptions hold, this study aims to contribute to the debate. This means that an association does not establish causality."
        for mode in ("light", "balanced", "deep"):
            revised, _ = voice.humanize_scholarly_text(source, mode)
            self.assertIn("Insofar as", revised)
            self.assertIn("aims to", revised)
            self.assertIn("This means that", revised)
            self.assertNotIn("Because the model assumptions hold", revised)

    def test_claim_polarity_modality_intention_and_acronyms_are_checked(self):
        examples = [
            ("The estimate does not establish causality.", "The estimate does establish causality."),
            ("The estimate may change.", "The estimate will change."),
            ("The study aims to explain concentration.", "The study explains concentration."),
            ("The BL allocation includes MRK.", "The BL allocation includes GOOG."),
        ]
        for original, candidate in examples:
            valid, issues = voice.validate_humanizer_preservation(original, candidate, max_word_change_ratio=1)
            self.assertFalse(valid, (original, candidate))
            self.assertTrue(issues)

    def test_identical_local_candidates_do_not_repeat_full_analysis(self):
        with patch.object(voice, "_refine_mixed_block", side_effect=lambda text, *_args, **_kwargs: text), \
             patch.object(voice, "analyse_scholarly_style", wraps=voice.analyse_scholarly_style) as analyse:
            revised, _ = voice.humanize_scholarly_text(CLEAN, "deep")
        self.assertEqual(revised, CLEAN)
        self.assertEqual(analyse.call_count, 2)

    def test_balanced_word_processing_skips_clean_prose_and_accounts_for_it(self):
        items = [{"paragraph_index": 0, "text": CLEAN}, {"paragraph_index": 1, "text": FORMULAIC}]
        cfg = RefinerConfig(provider="openai", model="gpt-5.6-terra", base_url="https://example.invalid", api_key="test")
        checkpoints = []
        with patch("services.model_refiner._refine_openai", side_effect=lambda text, *_args, **_kwargs: text) as remote:
            revised, report = refine_paragraphs_with_model(items, config=cfg, mode="balanced", checkpoint_callback=checkpoints.append)
        self.assertEqual(remote.call_count, 1)
        self.assertNotIn(CLEAN, remote.call_args.args[0])
        self.assertEqual(revised[0], CLEAN)
        self.assertEqual(report["skipped_paragraphs"], 1)
        self.assertEqual(report["model_words"], len(FORMULAIC.split()))
        self.assertEqual(checkpoints[-1]["completed_words"], report["total_words"])
        self.assertEqual(checkpoints[-1]["completed_paragraphs"], 2)

    def test_deep_word_processing_can_still_revise_clean_prose(self):
        cfg = RefinerConfig(provider="openai", model="gpt-5.6-terra", base_url="https://example.invalid", api_key="test")
        with patch("services.model_refiner._refine_openai", side_effect=lambda text, *_args, **_kwargs: text) as remote:
            _, report = refine_paragraphs_with_model([{"paragraph_index": 0, "text": CLEAN}], config=cfg, mode="deep")
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(report["skipped_paragraphs"], 0)

    def test_selective_targets_are_batched_across_unchanged_paragraphs(self):
        cfg = RefinerConfig(provider="openai", model="gpt-5.6-terra", base_url="https://example.invalid", api_key="test")
        items = [{"paragraph_index": index, "text": FORMULAIC if index % 2 else CLEAN} for index in range(8)]
        with patch("services.model_refiner._refine_openai", side_effect=lambda text, *_args, **_kwargs: text) as remote:
            revised, report = refine_paragraphs_with_model(items, config=cfg, mode="balanced")
        self.assertEqual(remote.call_count, 1)
        self.assertEqual(len(revised), 8)
        self.assertEqual(report["skipped_paragraphs"], 4)

    def test_unsafe_model_paragraph_is_retained_without_losing_other_paragraphs(self):
        cfg = RefinerConfig(provider="openai", model="gpt-5.6-terra", base_url="https://example.invalid", api_key="test")
        source = "The estimate does not establish causality."
        with patch("services.model_refiner._refine_openai", side_effect=lambda text, *_args, **_kwargs: text.replace("does not", "does")):
            revised, report = refine_paragraphs_with_model([{"paragraph_index": 0, "text": source}], config=cfg, mode="deep")
        self.assertEqual(revised[0], source)
        self.assertEqual(report["batches"][0]["rejected_paragraphs"], 1)


class SignalIndexEfficiencyTests(unittest.TestCase):
    def test_all_paragraphs_including_late_hotspots_are_scored(self):
        paragraphs = [CLEAN] * 125 + [FORMULAIC] * 10
        report = dashboard_report("\n\n".join(paragraphs), use_calibrator=False)
        self.assertEqual(report["ai_detector"]["segment_count"], 135)
        self.assertEqual(report["analysis_coverage"]["assessed_paragraphs"], 135)
        self.assertGreater(report["ai_detector"]["segment_profile"][-1]["ai_signal"], 0)

    def test_reference_text_cannot_change_prose_scores(self):
        source = "\n\n".join([CLEAN, FORMULAIC])
        base = dashboard_report(source, use_calibrator=False)
        extended = dashboard_report(source + "\n\nReferences\n\n" + FORMULAIC * 15, use_calibrator=False)
        self.assertEqual(base["ai_detection_percentage"], extended["ai_detection_percentage"])
        self.assertEqual(base["human_like_style_percentage"], extended["human_like_style_percentage"])
        self.assertEqual(base["analysis_coverage"]["assessed_words"], extended["analysis_coverage"]["assessed_words"])
        self.assertGreater(extended["analysis_coverage"]["excluded_words"], 0)

    def test_short_sample_is_provisional_and_complements_remain_exact(self):
        report = dashboard_report(CLEAN, use_calibrator=False)
        self.assertEqual(report["decision_status"], "Indeterminate")
        self.assertEqual(report["ai_confidence"], "Low")
        self.assertFalse(report["sufficient_prose"])
        self.assertEqual(report["human_like_style_percentage"] + report["ai_detection_percentage"], 100)

    def test_structure_only_input_does_not_offer_a_human_style_result(self):
        report = dashboard_report("# Results\n\n| Asset | Weight |\n| MRK | 18% |", use_calibrator=False)
        self.assertFalse(report["score_available"])
        self.assertEqual(report["analysis_coverage"]["assessed_words"], 0)

    def test_sentence_scoring_is_reused_within_each_request(self):
        with patch("services.ai_detector.sentence_ai_signal", wraps=sentence_ai_signal) as score:
            dashboard_report("\n\n".join([CLEAN] * 30), use_calibrator=False)
        self.assertLessEqual(score.call_count, 3)

    def test_ai_vocabulary_does_not_match_inside_unrelated_words(self):
        report = ai_check_report("The unleveraged asset was assessed separately.")
        self.assertEqual(next(item["score"] for item in report["signals"] if item["key"] == "A"), 0)

    def test_old_calibration_requires_refitting_for_the_new_feature_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model = {"trained": True, "model_type": "logistic", "weights": {}}
            path.write_text(json.dumps(model))
            with patch.dict(os.environ, {"HUMANIZER_CALIBRATION_MODEL": str(path)}):
                self.assertIsNone(load_model())
                model["feature_schema_version"] = FEATURE_SCHEMA_VERSION
                path.write_text(json.dumps(model))
                self.assertIsNotNone(load_model())


if __name__ == "__main__":
    unittest.main()
