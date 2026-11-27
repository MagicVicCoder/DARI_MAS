from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from dari.backends import Generation, ScriptedBackend
from dari.costs import ModelProfile, Usage
from dari.evaluation import ExactMatch
from dari.runtime import Decision, Runtime, RuntimeConfig
from dari.schema import Kind, Task
from dari.receiver_collection import collect_specialist
from dari.demo import collect_demo_targets
from dari.train_controller import train_from_targets


class ReceiverBackend(ScriptedBackend):
    def __init__(self):
        super().__init__({})
        self.actual_specialist_calls = 0

    def generate(self, request):
        if request.role == "anchor":
            answer = "1" if "Correct local result" in request.prompt else "-1"
            obj = {"step": "Reason about sign", "refs": ["p1"], "goal": "Check sign", "answer": answer}
        else:
            self.actual_specialist_calls += 1
            good = request.seed % 2 == 0
            obj = {"work": "A principal root is nonnegative", "draft":
                "Correct local result: use 1" if good else "Wrong local result: use -1"}
        text = json.dumps(obj)
        features = {k: self._feature(request.role, t) for k, t in request.feature_texts.items()}
        features["output"] = self._feature(request.role, text)
        if request.role == "specialist":
            features["draft"] = self._feature(request.role, json.dumps(obj["draft"]))
        return Generation(text, Usage(request.model_id, self.count_tokens(request.model_id, request.prompt),
                                      self.count_text_tokens(request.model_id, text)), features)


def make_receiver_runtime():
    profiles = {key: ModelProfile(key, "synthetic_units", .1, .2, provenance="fixture", synthetic=True)
                for key in ("A", "S")}
    return Runtime(ReceiverBackend(), profiles, RuntimeConfig())


def submit_policy(state, actions):
    for kind in (Kind.SUBMIT, Kind.FINISH, Kind.CONTINUE, Kind.ABORT):
        for action in actions:
            if action.kind == kind:
                return Decision(action, 0., state.ledger.unit)
    raise ValueError("No test action")


class ReceiverCollectionTests(unittest.TestCase):
    def test_receiver_rollouts_prefer_help_that_changes_the_actual_answer(self):
        runtime = make_receiver_runtime()
        prefix = runtime.initial(Task("t", "sqrt(x*x), x=-1", {"p1": "x=-1"}))
        before = deepcopy(prefix)
        action = next(a for a in runtime.actions(prefix) if a.kind == Kind.HELP and
                      a.target == "s1" and a.quota == "short")
        result = collect_specialist(runtime, prefix, action, submit_policy, ExactMatch(), "1",
                                    candidates=2, repeats=2, snapshot_id="frozen-test", lambda_cost=0.)
        self.assertEqual(len(result.preferences), 1)
        self.assertIn("Correct local result", result.preferences[0]["chosen"])
        self.assertEqual(runtime.backend.actual_specialist_calls, 2)  # replay does not resample
        self.assertEqual(prefix.ledger.total, before.ledger.total)
        self.assertEqual(prefix.calls, before.calls)
        self.assertTrue(all(row["synthetic"] for row in result.preferences))

    def test_forced_early_submit_is_only_auxiliary_and_does_not_replace_policy_returns(self):
        runtime = make_receiver_runtime()
        prefix = runtime.initial(Task("t", "question", {"p1": "constraint"}))
        action = next(a for a in runtime.actions(prefix) if a.kind == Kind.HELP and a.target == "s1")

        def abort_policy(state, legal):
            selected = next((a for a in legal if a.kind == Kind.ABORT), None)
            return Decision(selected, 0., state.ledger.unit) if selected else submit_policy(state, legal)

        result = collect_specialist(runtime, prefix, action, abort_policy, ExactMatch(), "1",
                                    candidates=2, repeats=1, snapshot_id="abort-test", lambda_cost=0., early_probe=True)
        self.assertEqual(result.preferences, [])
        self.assertTrue(any("Correct local result" in row["response"] for row in result.auxiliary))
        self.assertTrue(all(row["utility"] == 0. for row in result.outcomes))

    def test_controller_resume_preserves_config_and_trains_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/"targets.jsonl"
            collect_demo_targets(data)
            train_from_targets(data, root/"first.pt", steps=2, dim=16)
            result = train_from_targets(data, root/"next.pt", steps=2, dim=16, resume=root/"first.pt")
            self.assertTrue((root/"next.pt").exists())
            self.assertEqual(result["initialized_from"], str(root/"first.pt"))


if __name__ == "__main__":
    unittest.main()
