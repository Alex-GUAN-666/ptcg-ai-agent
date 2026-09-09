"""Public tests exercising the actual released model and preserved weights."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.decision_demo import run_demo
from ptcg_agent import Agent, agent
from ptcg_agent.weights import ASSETS, EXPECTED_FLOATS, load_parameters
from tools.compare_release import observations
from tools.inspect_submission import ROOT
from tools.smoke_submission import validate_selection


class ReleasedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.player = Agent()

    def test_parameter_layout(self):
        self.assertEqual(len(self.player.model.w), 133)
        self.assertEqual(sum(w.size for w in self.player.model.w.values()), EXPECTED_FLOATS)
        self.assertEqual(self.player.model.w["base.deck.weight"].shape, (204, 32))

    def test_weight_views_read_only(self):
        self.assertTrue(all(not w.flags.writeable for w in self.player.model.w.values()))

    def test_auxiliary_heads_not_in_export(self):
        self.assertFalse(any(key.startswith("plan_") for key in self.player.model.w))

    def test_deck_callback(self):
        deck = self.player({"select": None})
        self.assertEqual(len(deck), 60)
        self.assertTrue(all(type(card) is int and card > 0 for card in deck))

    def test_public_callback(self):
        self.assertEqual(len(agent({"select": None})), 60)

    def test_all_synthetic_selection_contracts(self):
        player = Agent()
        for label, raw in observations():
            with self.subTest(case=label):
                selected = player(copy.deepcopy(raw))
                if raw.get("select") is not None:
                    validate_selection(selected, raw)

    def test_golden_selections(self):
        report = json.loads((ROOT / "evidence/release_parity.json").read_text())
        player = Agent()
        for (label, raw), expected in zip(observations(), report["cases"], strict=True):
            with self.subTest(case=label):
                self.assertEqual(label, expected["case"])
                self.assertEqual(player(copy.deepcopy(raw)), expected["selected_indices"])

    def test_history_instances_are_independent(self):
        first, second = Agent(), Agent()
        first.memory.logs_seen = 7
        self.assertEqual(second.memory.logs_seen, 0)

    def test_deck_request_resets_history(self):
        player = Agent()
        player.memory.logs_seen = 7
        player.last_step = 99
        player({"select": None})
        self.assertEqual(player.memory.logs_seen, 0)
        self.assertEqual(player.last_step, -1)

    def test_demo_has_finite_scores_and_one_selection(self):
        result = run_demo()
        self.assertEqual(result["scope"], "synthetic_interface_demo_not_a_match")
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertEqual(result["count_logit_classes"], 17)
        self.assertTrue(all(np.isfinite(row["score"]) for row in result["options"]))

    def test_generated_release_hashes(self):
        manifest = json.loads((ROOT / "evidence/release_manifest.json").read_text())
        for path, expected in manifest["generated_files"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_reject_modified_weight_buffer(self):
        with tempfile.TemporaryDirectory(prefix="ptcg_corrupt_weights_") as temporary:
            folder = Path(temporary)
            (folder / "layout.json").write_bytes((ASSETS / "layout.json").read_bytes())
            payload = bytearray((ASSETS / "parameters.f32").read_bytes())
            payload[0] ^= 1
            (folder / "parameters.f32").write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_parameters(folder)

    def test_reject_inconsistent_layout(self):
        with tempfile.TemporaryDirectory(prefix="ptcg_bad_layout_") as temporary:
            folder = Path(temporary)
            (folder / "parameters.f32").write_bytes((ASSETS / "parameters.f32").read_bytes())
            metadata = json.loads((ASSETS / "layout.json").read_text())
            metadata["tensor_layout"][0][2] = 1
            (folder / "layout.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "offsets"):
                load_parameters(folder)


if __name__ == "__main__":
    unittest.main()
