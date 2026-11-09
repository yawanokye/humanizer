from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app import HumanizeRequest, humanize, status
from services.model_refiner import (
    RefinerConfig, _refine_openai, refine_paragraphs_with_model,
    resolve_engine2_model,
)


class SolConfigurationTests(unittest.TestCase):
    def test_configured_selection_respects_the_deployment_model(self):
        for selection in (None, "", "configured"):
            self.assertEqual(resolve_engine2_model(selection, "gpt-6.1-sol"), ("gpt-6.1-sol", "configured"))
            self.assertEqual(resolve_engine2_model(selection, "private-model"), ("private-model", "configured"))
        self.assertEqual(resolve_engine2_model("gpt-6.1-sol", "gpt-5.6-terra"), ("gpt-6.1-sol", "gpt-6.1-sol"))
        self.assertEqual(HumanizeRequest(text="A sample.", engine="engine2").engine2_model, "configured")

    def test_standard_sol_environment_and_status_do_not_expose_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-test-value", "OPENAI_MODEL": " gpt-6.1-sol ", "OPENAI_BASE_URL": " https://api.openai.com/v1/ "}, clear=True):
            cfg = RefinerConfig.from_env()
            self.assertEqual(cfg.provider, "openai")
            self.assertEqual(cfg.model, "gpt-6.1-sol")
            self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
            report = status()
            self.assertTrue(report["engines"]["engine2"]["configured"])
            self.assertEqual(report["engines"]["engine2"]["model"], "gpt-6.1-sol")
            self.assertNotIn("private-test-value", str(report))

    def test_missing_key_is_distinguished_from_model_support(self):
        with patch.dict(os.environ, {"HUMANIZER_PROVIDER": "openai", "OPENAI_MODEL": "gpt-6.1-sol"}, clear=True):
            engine = status()["engines"]["engine2"]
            self.assertFalse(engine["configured"])
            self.assertEqual(engine["configuration_missing"], ["OPENAI_API_KEY"])

    def test_responses_payload_uses_sol_and_compatible_reasoning(self):
        cfg = RefinerConfig(provider="openai", model="gpt-6.1-sol", base_url="https://example.invalid/v1", api_key="test")
        response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Checked text."}]}]}
        for mode, effort in (("balanced", "low"), ("deep", "medium")):
            with patch("services.model_refiner._post_json", return_value=response) as post:
                self.assertEqual(_refine_openai("Source text.", cfg, mode), "Checked text.")
                url, payload, _, _ = post.call_args.args
                self.assertTrue(url.endswith("/responses"))
                self.assertEqual(payload["model"], "gpt-6.1-sol")
                self.assertEqual(payload["reasoning"]["effort"], effort)
                self.assertFalse(payload["store"])
                self.assertNotIn("temperature", payload)

    def test_plain_text_request_does_not_replace_sol_with_a_legacy_alias(self):
        text = "The equilibrium prior may improve the balance of the portfolio. We compared daily returns from 2024 and reviewed the allocation across the selected assets."
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "gpt-6.1-sol"}, clear=True):
            with patch("services.model_refiner._refine_openai", side_effect=lambda text, cfg, *args, **kwargs: text) as model:
                result = humanize(HumanizeRequest(text=text, engine="engine2", mode="deep", preserve_terms=["equilibrium prior"]))
            self.assertTrue(model.called)
            self.assertEqual(model.call_args.args[1].model, "gpt-6.1-sol")
            self.assertEqual(result["engine_2"]["model"], "gpt-6.1-sol")
            self.assertEqual(result["actual_engine"], "engine2")

    def test_word_paragraph_path_also_keeps_the_configured_model(self):
        cfg = RefinerConfig(provider="openai", model="gpt-6.1-sol", base_url="https://example.invalid/v1", api_key="test")
        text = "The equilibrium prior may improve the balance of the portfolio. We reviewed daily returns from 2024."
        with patch("services.model_refiner._refine_openai", side_effect=lambda text, cfg, *args, **kwargs: text) as model:
            result, report = refine_paragraphs_with_model([{"paragraph_index": 3, "text": text}], config=cfg, mode="deep", preserve_terms=["equilibrium prior"])
        self.assertEqual(result[3], text)
        self.assertEqual(model.call_args.args[1].model, "gpt-6.1-sol")
        self.assertEqual(report["model"], "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
