from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from dari.context import anchor_input
from dari.pilot import Experiment, eligible, state_id
from dari.pilot_evaluation import final_answer, numeric, score
from tests import test_runtime as fixtures
from tests.test_runtime import make_runtime, anchor


class PilotTests(unittest.TestCase):
    def test_answer_normalization_and_dataset_specific_metrics(self):
        self.assertEqual(numeric("FINAL: -1,200.50"), numeric("-2401/2"))
        self.assertEqual(numeric(r"\boxed{\frac{1}{2}}"), numeric("0.5"))
        self.assertEqual(score("gsm8k", "FINAL: 0.5", "1/2")["reward"], 1)
        self.assertEqual(score("gsm8k", '{"total_pets": 28}', "28")["reward"], 1)
        self.assertEqual(score("gsm8k", "40 minutes", "40")["reward"], 1)
        self.assertEqual(score("gsm8k", "400 - 355", "45")["reward"], 1)
        self.assertEqual(score("gsm8k", "(4+1)/2", "2.5")["reward"], 1)
        self.assertIsNone(numeric("1 or 2"))
        self.assertIsNone(numeric("__import__('os')"))
        self.assertEqual(score("gsm8k", '{"a": 28, "b": 3}', "28")["reward"], 0)
        self.assertEqual(score("hotpotqa", "The River", "river")["reward"], 1)
        self.assertEqual(score("hotpotqa", "no, yes", "yes")["f1"], 0)
        self.assertEqual(final_answer(r"derive: \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        try:
            import math_verify
        except ImportError:
            self.skipTest("optional Phase2A evaluator missing")
        self.assertEqual(score("math500", r"\frac{1}{2}", "0.5")["reward"], 1)
        self.assertEqual(score("math500", "2", "3")["reward"], 0)

    def test_stepwise_reuses_runtime_without_request_prompt_or_history_field(self):
        runtime=make_runtime(generate_help_request=False)
        state=runtime.initial(fixtures.RuntimeTests().task())
        packet=anchor_input(state,"Continue")
        self.assertNotIn("help_request",packet.text)
        self.assertEqual(len(state.steps),1)

    def test_format_retry_reports_specific_error_without_relaxing_schema(self):
        runtime=make_runtime(anchor_outputs=[anchor(refs=["invented"]),anchor(refs=["p1"])])
        state=runtime.initial(fixtures.RuntimeTests().task())
        self.assertEqual(state.calls["anchor"],2)
        self.assertIn("Unknown Anchor reference",state.events[-1].prompt)
        self.assertNotIn("invented",state.steps[0].record.refs)

    def test_counterfactual_branches_keep_prefix_and_equal_common_step_allowance(self):
        outputs=[anchor(answer=None)]+[anchor(step="new work",answer="1",goal=None)]*20
        runtime=make_runtime(anchor_outputs=outputs)
        experiment=Experiment.__new__(Experiment)
        experiment.runtime=runtime
        experiment.config={"context_width":"narrow","continuation_steps":3}
        prefix=runtime.initial(fixtures.RuntimeTests().task())
        before=state_id(prefix)
        self.assertTrue(eligible(runtime,prefix,"narrow"))
        for method in ("continue","self_repair","self_repair_extended","help_submit"):
            state,meta=experiment.branch(prefix,method,7)
            self.assertEqual(state_id(prefix),before)
            self.assertEqual(meta["common_continuation_step_cap"],3)
            self.assertEqual(state.candidate,"1")
            self.assertEqual(meta["dynamic_request"]["instruction"],prefix.help_request.instruction)
            self.assertEqual(meta["intervention_steps"],2 if method=="self_repair_extended" else 1)
            future=sum(row["calls"] for row in meta["intervention_resources"].values())
            self.assertEqual(future,2 if method in {"self_repair_extended","help_submit"} else 1)
        self.assertIsNone(prefix.help)
        self.assertEqual(prefix.calls["specialist"],0)

    def test_candidate_prefix_is_not_eligible_and_missing_draft_is_not_discarded(self):
        runtime=make_runtime(specialist_outputs=["not JSON"])
        prefix=runtime.initial(fixtures.RuntimeTests().task())
        self.assertFalse(eligible(runtime,prefix,"narrow"))
        prefix.candidate=None
        experiment=Experiment.__new__(Experiment)
        experiment.runtime=runtime
        experiment.config={"context_width":"narrow","continuation_steps":3}
        state,meta=experiment.branch(prefix,"help_submit",7)
        self.assertFalse(meta["submitted"])
        self.assertEqual(state.calls["specialist"],1)
        self.assertTrue(any(not event.valid for event in state.events))

    def test_oracle_retains_ties_and_uses_only_complete_action_groups(self):
        from dari.pilot_analysis import matched_groups,local_oracle
        from dari.pilot import BRANCHES
        rows=[{"prefix_id":"p1","method":method,"reward":float(method=="help_submit"),"dataset":"gsm8k"} for method in BRANCHES]
        rows.append({"prefix_id":"incomplete","method":"continue","reward":1.,"dataset":"gsm8k"})
        groups=matched_groups(rows)
        self.assertEqual(set(groups),{"p1"})
        report=local_oracle(groups)
        self.assertEqual(report[0]["local_oracle"],1.)
        self.assertEqual(report[0]["oracle_gap"],0.)

    def test_resume_rejects_incompatible_manifest_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"samples.jsonl").write_text("")
            (root/"manifest.json").write_text(json.dumps({"experiment_id":"incompatible"}))
            config=root/"config.json"
            config.write_text(json.dumps({"output_dir":str(root),"sample_file":str(root/"samples.jsonl")}))
            with self.assertRaisesRegex(ValueError,"Resume rejected"):
                Experiment(config)

    def test_saved_prefix_roundtrip_contains_no_offline_labels(self):
        import torch
        runtime=make_runtime(anchor_outputs=[anchor(answer=None)])
        state=runtime.initial(fixtures.RuntimeTests().task())
        with tempfile.TemporaryDirectory() as directory:
            experiment=Experiment.__new__(Experiment)
            experiment.root=Path(directory)
            experiment.runtime=runtime
            experiment.backend=runtime.backend
            experiment.dimensions={"anchor_dim":12,"specialist_dim":9}
            experiment.identity="fixture"
            prefix_id=experiment.save_prefix(state,{"task_id":state.task.task_id,"gold":"OFFLINE_SECRET"})
            path=Path(directory)/"e1/prefixes"/(prefix_id+".pt")
            restored=torch.load(path,weights_only=False)
            self.assertEqual(state_id(restored),prefix_id)
            self.assertNotIn("OFFLINE_SECRET",path.with_suffix(".json").read_text())
            self.assertFalse(hasattr(restored.task,"gold"))
