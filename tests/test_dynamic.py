import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from dari.context import build_pack
from dari.hf_backend import output_value_spans
from dari.schema import Kind
from tests import test_runtime as fixtures
from tests.test_runtime import anchor, specialist, make_runtime, pick


def dynamic(answer=None, **extra):
    obj = json.loads(anchor(answer=answer, refs=["p1"]))
    obj.update(help_request={"instruction": "Find an alternative derivation", "target": "current_step", "required_refs": ["p2"]})
    obj.update(extra)
    return json.dumps(obj)


class DynamicTests(unittest.TestCase):
    def test_same_request_conditions_self_repair_and_help_and_context(self):
        runtime = make_runtime(anchor_outputs=[dynamic()])
        state = runtime.initial(fixtures.RuntimeTests().task())
        self.assertEqual(state.help_request.target, "s1")
        for kind in (Kind.REPAIR, Kind.HELP):
            action = pick(runtime, state, kind)
            self.assertIn(state.help_request.instruction, runtime.request_for_action(state, action).prompt)
        self.assertIn("p2", build_pack(state, "s1", "narrow").span_ids)
        self.assertTrue(all(not action.template for action in runtime.actions(state)))

    def test_invalid_expansion_retains_last_valid_draft_and_private_history(self):
        runtime = make_runtime(anchor_outputs=[dynamic()], specialist_outputs=[specialist(), "bad"])
        state = runtime.initial(fixtures.RuntimeTests().task())
        state = runtime.step(state, pick(runtime, state, Kind.HELP, quota="short"))
        previous = state.help.draft
        state = runtime.step(state, pick(runtime, state, Kind.EXPAND))
        self.assertEqual(state.help.draft, previous)
        self.assertEqual(len(state.help.private_outputs), 1)
        self.assertIn(Kind.SUBMIT, [action.kind for action in runtime.actions(state)])

    def test_candidate_revocation_is_append_only(self):
        runtime = make_runtime(anchor_outputs=[dynamic("old"), dynamic(invalidate_answer=True)])
        state = runtime.initial(fixtures.RuntimeTests().task())
        state = runtime.step(state, pick(runtime, state, Kind.REPAIR))
        self.assertIsNone(state.candidate)
        self.assertEqual(len(state.steps), 2)
        self.assertNotIn(Kind.FINISH, [action.kind for action in runtime.actions(state)])

    def test_three_stages_and_session_isolation(self):
        runtime = make_runtime(anchor_outputs=[dynamic(), dynamic()], specialist_outputs=[specialist()]*4, max_specialist_stages=3)
        state = runtime.initial(fixtures.RuntimeTests().task())
        state = runtime.step(state, pick(runtime, state, Kind.HELP, quota="short"))
        for _ in range(2):
            state = runtime.step(state, pick(runtime, state, Kind.EXPAND))
        self.assertEqual(state.help.stage, 3)
        self.assertNotIn(Kind.EXPAND, [action.kind for action in runtime.actions(state)])
        state = runtime.step(state, pick(runtime, state, Kind.ABORT))
        state = runtime.step(state, pick(runtime, state, Kind.HELP, quota="short"))
        self.assertEqual(state.help.stage, 1)
        self.assertEqual(json.loads(state.events[-1].prompt.split("\n", 1)[1])["previous_private_outputs"], [])

    def test_fallback_is_labeled_without_fabricated_request_feature(self):
        runtime = make_runtime()
        state = runtime.initial(fixtures.RuntimeTests().task())
        self.assertTrue(state.help_request.fallback)
        self.assertNotIn("help_request", state.features)

    def test_unknown_reference_is_billed_and_retried(self):
        runtime = make_runtime(anchor_outputs=[anchor(refs=["missing"]), dynamic()])
        state = runtime.initial(fixtures.RuntimeTests().task())
        self.assertEqual(state.calls["anchor"], 2)
        self.assertFalse(state.events[0].valid)

    def test_json_spans_handle_escaped_strings_and_repeated_keys(self):
        text = dynamic()
        begin, end = output_value_spans(text)["help_request"]
        self.assertEqual(text[begin:end], "Find an alternative derivation")

    def test_runtime_request_metadata_never_leaks_into_anchor_protocol_history(self):
        runtime = make_runtime(anchor_outputs=[dynamic()])
        state = runtime.initial(fixtures.RuntimeTests().task())
        request = runtime.request_for_action(state, pick(runtime, state, Kind.CONTINUE))
        body = json.loads(request.prompt.split("\n", 1)[1])
        self.assertEqual(set(body["steps"][0]), {"id", "step", "refs"})

    def test_real_debug_utility_collection_is_rejected(self):
        from dari.collection import collect_branches
        from dari.evaluation import ExactMatch
        runtime = make_runtime(experiment_mode="real_debug")
        state = runtime.initial(fixtures.RuntimeTests().task())
        with self.assertRaisesRegex(ValueError, "real_debug"):
            collect_branches(runtime, state, [], None, ExactMatch(), "1", 1, "debug", 12, 9)

    def test_candidate_invalidation_clears_reencoded_old_candidate_feature(self):
        runtime = make_runtime(anchor_outputs=[dynamic("obsolete"), dynamic(invalidate_answer=True)])
        state = runtime.initial(fixtures.RuntimeTests().task())
        state = runtime.step(state, pick(runtime, state, Kind.CONTINUE))
        self.assertIsNone(state.candidate)
        self.assertNotIn("candidate", state.features)

    def test_generated_request_pooling_matches_actual_forwarded_token_span(self):
        from dari.backends import GenerationRequest
        from dari.hf_backend import HuggingFaceBackend, ModelHandle, PromptCodec
        from tests.model_fixtures import CharacterTokenizer

        text = dynamic()

        class TokenStateModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(hidden_size=4)

            def forward(self, input_ids, past_key_values=None, **kwargs):
                position = past_key_values or 0
                token = ord(text[position])+3 if position < len(text) else 2
                logits = torch.full((*input_ids.shape, 259), -1000.)
                logits[..., token] = 0
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
                return SimpleNamespace(logits=logits, hidden_states=(hidden,), past_key_values=position+1)

        backend = HuggingFaceBackend({"A": ModelHandle(TokenStateModel(), PromptCodec(CharacterTokenizer()), identity={"fixture": "token-state"})})
        result = backend.generate(GenerationRequest("A", "anchor", "task", len(text)+1, 0))
        expected = torch.tensor([ord(character)+3 for character in "Find an alternative derivation"], dtype=torch.float32).mean()
        self.assertAlmostEqual(result.features["help_request"].values[0], float(expected), places=5)
