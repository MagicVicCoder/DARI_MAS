from dataclasses import replace
import unittest

from dari.backends import Generation
from dari.costs import Usage
from dari.native import NativeAction, NativeConfig, NativeRuntime
from dari.schema import Task


class NativeMock:
    def __init__(self):
        self.calls = []
        self.script = list(map(ord, "We need to verify whether 2 is prime.\n\nFINAL: yes\n"))

    def native_prompt_ids(self, model, prompt):
        self.calls.append(("prompt", prompt))
        return [1, 2]

    def native_encode(self, model, text):
        return list(map(ord, text))

    def native_decode(self, model, ids):
        return "".join(map(chr, ids))

    def native_limit(self, model):
        return 8192

    def native_generate(self, model, ids, cap, seed, rng):
        self.calls.append(("native", list(ids), rng))
        offset = rng or 0
        result = self.script[offset:offset+cap]
        end = offset+len(result)
        return result, end, end == len(self.script), [1.], {}

    def count_tokens(self, model, text):
        return len(text)

    count_text_tokens = count_tokens

    def generate(self, request):
        self.calls.append(("specialist", request.prompt))
        return Generation('{"work":"private calculation","draft":"draft result"}',
                          Usage(request.model_id, 20, 10))


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.backend = NativeMock()
        self.config = NativeConfig(checkpoint_interval=38, boundary_wait=4, anchor_budget=70)
        self.task = Task("demo", "Is 2 prime?")

    def frontier(self, mode):
        runtime = NativeRuntime(self.backend, replace(self.config, delegation_mode=mode))
        initial = runtime.initial(self.task)
        state = runtime.step(initial, NativeAction("continue"))
        self.assertTrue(state.boundary)
        return runtime, state

    def test_continue_preserves_native_tokens_and_rng(self):
        runtime = NativeRuntime(self.backend, self.config)
        uninterrupted = runtime.standard(self.task, 7)
        observed = runtime.run(self.task, lambda state, legal: NativeAction("continue"), 7)
        self.assertEqual(uninterrupted.context_ids, observed.context_ids)
        self.assertEqual(observed.context_ids, [1, 2] + self.backend.script)
        self.assertEqual(observed.candidate, "yes")
        self.assertEqual(sum(call[0] == "prompt" for call in self.backend.calls), 2)

    def test_abort_and_submit_isolation(self):
        runtime, state = self.frontier("frontier")
        helped = runtime.step(state, NativeAction("help"))
        aborted = runtime.step(helped, NativeAction("abort"))
        self.assertEqual((aborted.context_ids, aborted.rng_state), (state.context_ids, state.rng_state))
        self.assertGreater(aborted.specialist_tokens, 0)
        submitted = runtime.step(helped, NativeAction("submit"))
        self.assertEqual(submitted.context_ids[:state.position], state.context_ids)
        self.assertEqual(submitted.generated_ids, state.generated_ids)
        self.assertEqual(submitted.injections[0]["content"], "draft result")
        self.assertNotIn("private calculation", runtime.decode(submitted.context_ids))
        self.assertEqual([call[0] for call in self.backend.calls].count("specialist"), 1)
        self.assertEqual(runtime.step(submitted, NativeAction("continue")).candidate, "yes")

    def test_delegation_masks_and_fallback(self):
        runtime, state = self.frontier("native_intent")
        self.assertTrue(runtime.assistance(state, NativeAction("help")).native_intent)
        state.boundary = replace(state.boundary, source_token_span=(2, 2), original_content="Reasoning.")
        self.assertEqual(runtime.assistance(state, NativeAction("help")).fallback_reason,
                         "no_current_explicit_open_issue")
        runtime, state = self.frontier("controller_guided")
        legal = runtime.actions(state)
        self.assertIn(NativeAction("help", "current_frontier", "verify"), legal)
        self.assertNotIn(NativeAction("help", "current_frontier", "explore"), legal)
        self.assertNotIn(NativeAction("help", "previous_frontier"), legal)
