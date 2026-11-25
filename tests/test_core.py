import json
import unittest

from dari.schema import AnchorRecord, Draft, SpecialistRecord, Task
from dari.costs import CostLedger, ModelProfile, Usage


class ProtocolTests(unittest.TestCase):
    def test_anchor_rejects_incomplete_or_wrongly_typed_records(self):
        with self.assertRaises(ValueError):
            AnchorRecord.parse('{"step":"unfinished"')
        with self.assertRaises(ValueError):
            AnchorRecord.parse(json.dumps(dict(step="ok", refs="s1", goal=None, answer=None)))
        record = AnchorRecord.parse(json.dumps(dict(step="derive", refs=["p1"], goal=None, answer="4")))
        self.assertEqual(record.answer, "4")

    def test_only_draft_is_serialized_for_receiver(self):
        text = json.dumps({"work": "PRIVATE SEARCH", "draft": "x=-1; use absolute value"})
        result = SpecialistRecord.parse(text)
        self.assertNotIn("PRIVATE", result.draft.text)
        self.assertEqual(result.draft.text, "x=-1; use absolute value")
        with self.assertRaises(ValueError):
            Draft.parse({"target": "s1"})
        with self.assertRaises(ValueError):
            SpecialistRecord.parse(json.dumps({"work": "reasoning", "draft": {"target": "s1"}}))

    def test_task_has_no_gold_or_reward_field(self):
        task = Task("t1", "What is 2+2?", {"p1": "addition"})
        self.assertNotIn("gold", task.__dict__)
        self.assertNotIn("reward", task.__dict__)


class CostTests(unittest.TestCase):
    def profiles(self):
        return {
            "large": ModelProfile("large", "synthetic_units", 1.0, 4.0, 0.1, provenance="fixture", synthetic=True),
            "small": ModelProfile("small", "synthetic_units", .25, 1.0, .025, provenance="fixture", synthetic=True),
        }

    def test_heterogeneous_tokens_are_not_used_as_compute_cost(self):
        ledger = CostLedger(self.profiles())
        ledger.charge(Usage("large", 100, 20))
        ledger.charge(Usage("small", 100, 20))
        self.assertAlmostEqual(ledger.total, 225.0)
        self.assertEqual(ledger.tokens_by_model()["large"]["input"], 100)
        self.assertEqual(ledger.tokens_by_model()["small"]["output"], 20)

    def test_cached_input_is_not_charged_twice(self):
        ledger = CostLedger(self.profiles())
        ledger.charge(Usage("large", 100, 20, cached_input_tokens=80))
        self.assertAlmostEqual(ledger.total, 108.0)

    def test_incompatible_units_and_unknown_models_fail(self):
        profiles = self.profiles()
        profiles["api"] = ModelProfile("api", "usd", .001, .003, provenance="API pricing")
        with self.assertRaises(ValueError):
            CostLedger(profiles)
        with self.assertRaises(KeyError):
            CostLedger(self.profiles()).charge(Usage("unknown", 1, 1))

    def test_invalid_counts_and_unmeasured_cost_are_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            Usage("large", 10, 1, cached_input_tokens=11)
        with self.assertRaises(ValueError):
            ModelProfile("large", "gpu_ms", -1., 1., provenance="bad")

    def test_observed_resource_cost_overrides_profile_estimate(self):
        ledger = CostLedger(self.profiles())
        ledger.charge(Usage("large", 100, 20, observed_cost=12.5, observed_unit="synthetic_units"))
        self.assertEqual(ledger.total, 12.5)
        with self.assertRaises(ValueError):
            ledger.charge(Usage("large", 1, 1, observed_cost=1., observed_unit="usd"))


if __name__ == "__main__":
    unittest.main()
