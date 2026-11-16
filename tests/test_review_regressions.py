import unittest
from dataclasses import replace

from dari.collection import collect_branches
from dari.evaluation import ExactMatch
from dari.runtime import Decision, Runtime
from dari.schema import Kind
from dari.controller import encode_state_actions
from tests import test_runtime as fixtures


class ReviewRegressionTests(unittest.TestCase):
    def test_submit_mask_uses_the_exact_executed_prompt_at_tight_budget(self):
        rt = fixtures.make_runtime()
        prefix = rt.initial(fixtures.RuntimeTests().task())
        helped = rt.step(prefix, fixtures.pick(rt, prefix, Kind.HELP, target="s1", quota="short"))
        submit = fixtures.pick(rt, helped, Kind.SUBMIT)
        generous = rt.step(helped, submit)
        executed_prompt = generous.events[-1].prompt
        exact_cap_cost = rt.estimate(rt.config.anchor_model, executed_prompt, rt.config.anchor_output)
        helped.budget = helped.ledger.total + exact_cap_cost - 1.
        self.assertNotIn(submit, rt.actions(helped))
        helped.budget += 1.
        self.assertIn(submit, rt.actions(helped))
        result = rt.step(helped, submit)
        self.assertEqual(result.calls["anchor"], helped.calls["anchor"] + 1)

    def test_ordinary_anchor_actions_expose_known_cost_to_the_controller(self):
        rt = fixtures.make_runtime()
        state = rt.initial(fixtures.RuntimeTests().task())
        action = fixtures.pick(rt, state, Kind.CONTINUE)
        _, features = encode_state_actions(rt, state, [action], 12, 9)
        self.assertGreater(features["numeric"][0, 0, 0].item(), 0.)
        self.assertGreater(features["numeric"][0, 0, 2].item(), 0.)

    def test_scripted_backend_cannot_be_relabelled_as_real_by_pricing_config(self):
        original = fixtures.make_runtime()
        purported_real = {name: replace(profile, synthetic=False, unit="usd")
                          for name, profile in original.profiles.items()}
        with self.assertRaises(ValueError):
            Runtime(original.backend, purported_real, original.config)

    def test_forced_branch_uses_same_remaining_horizon_as_online(self):
        rt = fixtures.make_runtime(max_decisions=1, anchor_outputs=[
            fixtures.anchor(answer="0", goal=None), fixtures.anchor(answer="1", goal=None),
            fixtures.anchor(answer="2", goal=None)])
        prefix = rt.initial(fixtures.RuntimeTests().task())
        action = fixtures.pick(rt, prefix, Kind.CONTINUE)
        policy = lambda s, legal: Decision(next(a for a in legal if a.kind == Kind.CONTINUE), 0., s.ledger.unit)
        online = rt.run_from(prefix, policy)
        row = collect_branches(rt, prefix, [action], policy, ExactMatch(), "2", repeats=1,
            snapshot_id="horizon-boundary", anchor_dim=12, specialist_dim=9)[0]
        self.assertEqual(online.candidate, "1")
        self.assertEqual(row.reward, 0.)
        self.assertAlmostEqual(row.cost, (online.ledger.total-prefix.ledger.total)/rt.config.reference_cost)
        self.assertEqual(online.decisions, 1)
        self.assertEqual(prefix.decisions, 0)


if __name__ == "__main__":
    unittest.main()
