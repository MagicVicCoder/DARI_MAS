import json
import unittest

from dari.backends import ScriptedBackend
from dari.costs import ModelProfile
from dari.runtime import Runtime, RuntimeConfig
from dari.context import build_pack
from dari.schema import Action, Kind, Task
from dari.evaluation import ExactMatch, AnswerF1, ChoiceAccuracy


def anchor(step="wrong sign", refs=None, answer="-1", goal="check sign"):
    return json.dumps(dict(step=step, refs=refs or [], goal=goal, answer=answer))


def specialist(work="PRIVATE SEARCH", target="s1"):
    return json.dumps(dict(work=work, draft="use absolute value: x is negative, so answer 1"))


def make_runtime(anchor_outputs=None, specialist_outputs=None, **config):
    backend = ScriptedBackend({"anchor": anchor_outputs or [anchor(refs=["p1"]), anchor("corrected", ["p1"], "1", None)],
                               "specialist": specialist_outputs or [specialist(), specialist("PRIVATE EXPANSION")]})
    profiles = {
        "A": ModelProfile("A", "synthetic_units", 1., 2., provenance="fixture", synthetic=True),
        "S": ModelProfile("S", "synthetic_units", .2, .4, provenance="fixture", synthetic=True),
    }
    config.setdefault("max_specialist_stages", 2)
    return Runtime(backend, profiles, RuntimeConfig(**config))


def pick(rt, state, kind, **fields):
    return next(a for a in rt.actions(state) if a.kind == kind and all(getattr(a, k) == v for k, v in fields.items()))


class RuntimeTests(unittest.TestCase):
    def task(self):
        return Task("sign", "Evaluate sqrt(x*x) at x=-1.", {"p1": "x=-1", "p2": "unrelated fact"})

    def test_submit_is_not_automatic_adoption_and_private_work_is_excluded(self):
        rt = make_runtime(anchor_outputs=[anchor(refs=["p1"]), anchor("independent review", [], "1", None)])
        state = rt.initial(self.task())
        helped = rt.step(state, pick(rt, state, Kind.HELP, target="s1", quota="short"))
        self.assertEqual(len(helped.steps), 1)
        self.assertEqual(helped.candidate, "-1")
        result = rt.step(helped, pick(rt, helped, Kind.SUBMIT))
        self.assertEqual(result.candidate, "1")
        self.assertEqual(result.artifacts, {})
        self.assertIn("use absolute value", result.events[-1].prompt)
        self.assertNotIn("PRIVATE SEARCH", result.events[-1].prompt)

    def test_only_explicitly_referenced_draft_is_retained(self):
        rt = make_runtime(anchor_outputs=[anchor(refs=["p1"]), anchor("corrected", ["h1"], "1", None)])
        state = rt.initial(self.task())
        state = rt.step(state, pick(rt, state, Kind.HELP, target="s1", quota="short"))
        state = rt.step(state, pick(rt, state, Kind.SUBMIT))
        self.assertIn("h1", state.artifacts)
        self.assertNotIn("PRIVATE SEARCH", state.artifacts["h1"])
        self.assertIsNone(state.help)

    def test_abort_costs_tokens_without_exposing_private_content(self):
        rt = make_runtime()
        initial = rt.initial(self.task())
        helped = rt.step(initial, pick(rt, initial, Kind.HELP, target="s1", quota="short"))
        result = rt.step(helped, pick(rt, helped, Kind.ABORT))
        self.assertGreater(result.ledger.total, initial.ledger.total)
        self.assertIn("S", result.ledger.tokens_by_model())
        self.assertNotIn("PRIVATE SEARCH", result.events[-1].prompt)
        self.assertNotIn("use absolute value", result.events[-1].prompt)
        self.assertIsNone(result.help)
        self.assertFalse(any(k.startswith("helper_") for k in result.features))

    def test_extension_preserves_only_specialist_private_history(self):
        rt = make_runtime()
        state = rt.initial(self.task())
        state = rt.step(state, pick(rt, state, Kind.HELP, target="s1", quota="short"))
        expanded = rt.step(state, pick(rt, state, Kind.EXPAND))
        self.assertIn("PRIVATE SEARCH", expanded.events[-1].prompt)
        self.assertEqual(expanded.help.stage, 2)
        self.assertNotIn(Kind.EXPAND, [a.kind for a in rt.actions(expanded)])
        one_shot = rt.initial(self.task())
        one_shot = rt.step(one_shot, pick(rt, one_shot, Kind.HELP, target="s1", quota="oneshot"))
        self.assertNotIn(Kind.EXPAND, [a.kind for a in rt.actions(one_shot)])

    def test_bad_helper_output_is_charged_and_cannot_be_submitted(self):
        rt = make_runtime(specialist_outputs=['{"work":"truncated'])
        state = rt.initial(self.task())
        helped = rt.step(state, pick(rt, state, Kind.HELP, target="s1", quota="short"))
        self.assertGreater(helped.ledger.total, state.ledger.total)
        self.assertEqual([a.kind for a in rt.actions(helped)], [Kind.ABORT])

    def test_finish_has_no_extra_model_call_and_actions_must_be_legal(self):
        rt = make_runtime()
        state = rt.initial(self.task())
        with self.assertRaises(ValueError):
            rt.step(state, Action(Kind.EXPAND))
        result = rt.step(state, pick(rt, state, Kind.FINISH))
        self.assertTrue(result.terminal)
        self.assertEqual(result.candidate, "-1")
        self.assertEqual(result.ledger.total, state.ledger.total)

    def test_anchor_parse_retry_and_no_answer_budget_exhaustion_terminate(self):
        rt = make_runtime(anchor_outputs=["bad", "also bad"])
        state = rt.initial(self.task())
        self.assertTrue(state.terminal)
        self.assertEqual(state.calls["anchor"], 2)
        self.assertIsNone(state.candidate)
        rt2 = make_runtime(budget=0.)
        empty = rt2.initial(self.task())
        self.assertTrue(empty.terminal)
        self.assertEqual(empty.calls["anchor"], 0)

    def test_context_pack_preserves_task_and_deduplicates_sources(self):
        rt = make_runtime(anchor_outputs=[anchor(refs=["p1", "p1"])])
        state = rt.initial(self.task())
        pack = build_pack(state, "s1", "narrow")
        self.assertEqual(pack.span_ids.count("p1"), 1)
        self.assertIn(self.task().question, pack.text)
        self.assertNotIn("unrelated fact", pack.text)

    def test_replaying_an_action_does_not_mutate_the_shared_prefix(self):
        rt = make_runtime()
        state = rt.initial(self.task())
        action = pick(rt, state, Kind.HELP, target="s1", quota="short")
        branch1, branch2 = rt.step(state, action), rt.step(state, action)
        self.assertIsNone(state.help)
        self.assertEqual(state.calls["specialist"], 0)
        self.assertEqual(branch1.events, branch2.events)
        self.assertEqual(branch1.ledger.total, branch2.ledger.total)


class OfflineEvaluatorTests(unittest.TestCase):
    def test_fixed_reward_adapters_handle_invalid_and_multiset_f1(self):
        self.assertEqual(ExactMatch().score(None, "1"), 0.)
        self.assertEqual(ExactMatch().score("1", "1"), 1.)
        self.assertEqual(ChoiceAccuracy().score("B", "B"), 1.)
        self.assertEqual(ChoiceAccuracy().score("not an option", "B"), 0.)
        self.assertAlmostEqual(AnswerF1().score("cat cat dog", "cat dog"), .8)


if __name__ == "__main__":
    unittest.main()
