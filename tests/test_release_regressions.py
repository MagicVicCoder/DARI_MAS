import json
from pathlib import Path
import tempfile
import unittest

from dari.backends import Generation, GenerationRequest, ScriptedBackend
from dari.collection import deployment_id
from dari.configuration import make_policy
from dari.costs import ModelProfile, Usage
from dari.demo import collect_demo_targets
from dari.evaluation import ExactMatch
from dari.hf_backend import HuggingFaceBackend, ModelHandle, PromptCodec
from dari.receiver_collection import collect_specialist
from dari.runtime import Runtime
from dari.schema import Action, Kind, Task
from dari.train_controller import train_from_targets
from tests.model_fixtures import CharacterTokenizer, DeterministicCachedLM
from tests.test_receiver_collection import ReceiverBackend, make_receiver_runtime, submit_policy


class ChangedIdentityBackend(ScriptedBackend):
    deployment_identity = {"engine": "another-engine", "representation": "another-hidden-space", "adapters": {}}


class EmptyLoserBackend(ReceiverBackend):
    def generate(self, request):
        if request.role == "specialist" and request.seed % 2:
            self.actual_specialist_calls += 1
            return Generation("", Usage(request.model_id, self.count_tokens(request.model_id, request.prompt), 1))
        return super().generate(request)


class ReleaseRegressionTests(unittest.TestCase):
    def test_hf_pooling_uses_named_value_range_instead_of_first_word_match(self):
        backend = HuggingFaceBackend({"A": ModelHandle(DeterministicCachedLM(), PromptCodec(CharacterTokenizer()),
                                                       identity={"model":"tensor-fixture-v1"})})
        prompt = 'Return JSON step. {"task":"step"}'
        request = GenerationRequest("A", "anchor", prompt, 4, 0, {"task":"step"})
        # This named range is produced by serialization, not substring matching.
        from dataclasses import replace
        if hasattr(request, "feature_spans"):
            request = replace(request, feature_spans={"task":(27,31)})
        result = backend.generate(request)
        self.assertGreater(result.features["task"].values[0], 30000.)

    def test_named_ranges_distinguish_task_sources_and_instruction_collisions(self):
        runtime = make_receiver_runtime()
        state = runtime.initial(Task("collision", "step", {"p1": "step"}))
        request = runtime.request_for_action(state, Action(Kind.CONTINUE))
        self.assertTrue(hasattr(request, "feature_spans"), "Requests need ID-specific ranges")
        task_range = request.feature_spans["task"]
        source_range = request.feature_spans["p1"]
        self.assertNotEqual(task_range, source_range)
        self.assertGreater(task_range[0], request.prompt.index("\n"))
        self.assertEqual(request.prompt[slice(*task_range)], "step")
        self.assertEqual(request.prompt[slice(*source_range)], "step")

    def test_changed_backbone_identity_is_rejected_even_at_same_hidden_dimension(self):
        with tempfile.TemporaryDirectory() as folder:
            data, ckpt = Path(folder)/"targets.jsonl", Path(folder)/"theta.pt"
            collect_demo_targets(data)
            train_from_targets(data, ckpt, steps=1, dim=16)
            from dari.demo import build_demo
            original, _ = build_demo()
            changed = Runtime(ChangedIdentityBackend({}), original.profiles, original.config)
            config = {"observation": {"anchor_dim":12,"specialist_dim":9}}
            with self.assertRaisesRegex(ValueError, "deployment|identity"):
                make_policy(changed, config, ckpt)

    def test_eos_only_loser_is_recorded_and_paid_but_never_exported_as_untrainable_pair(self):
        runtime = make_receiver_runtime()
        runtime = Runtime(EmptyLoserBackend(), runtime.profiles, runtime.config)
        prefix = runtime.initial(Task("t", "question", {"p1":"constraint"}))
        action = next(a for a in runtime.actions(prefix) if a.kind == Kind.HELP and a.target == "s1")
        result = collect_specialist(runtime, prefix, action, submit_policy, ExactMatch(), "1",
                                    candidates=2, repeats=1, snapshot_id="empty-loser", lambda_cost=0.)
        self.assertEqual(len(result.outcomes), 2)
        self.assertTrue(any(row["output"] == "" for row in result.outcomes))
        self.assertEqual(runtime.backend.actual_specialist_calls, 2)
        self.assertFalse(any(not row["rejected"].strip() for row in result.preferences))


if __name__ == "__main__":
    unittest.main()
