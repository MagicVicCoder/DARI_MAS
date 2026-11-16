import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dari.controller import ControllerConfig, ValueController, encode_state_actions
from dari.training import controller_loss, dpo_loss, sequence_log_probs, inject_lora
from dari.collection import collect_branches, read_targets, write_targets, StageOutcome, preference_pairs
from dari.runtime import Decision
from dari.schema import Kind
from dari.evaluation import ExactMatch
from tests import test_runtime as fixtures

make_runtime, pick = fixtures.make_runtime, fixtures.pick

torch.set_num_threads(1)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.rt = make_runtime()
        self.state = self.rt.initial(fixtures.RuntimeTests().task())
        self.actions = self.rt.actions(self.state)
        self.obs, self.act = encode_state_actions(self.rt, self.state, self.actions, 12, 9)
        self.model = ValueController(ControllerConfig(12, 9, dim=16, heads=4, ff_dim=32))

    def test_dual_values_and_masks_choose_only_legal_actions(self):
        self.act["legal"][:, 1:] = False
        out = self.model(self.obs, self.act)
        self.assertEqual(out["utility"].argmax(-1).item(), 0)
        self.assertTrue(torch.isneginf(out["utility"][0, 1:]).all())
        self.assertTrue(((out["reward"] >= 0) & (out["reward"] <= 1)).all())
        self.assertTrue((out["cost"] >= 0).all())
        self.act["legal"][:] = False
        with self.assertRaises(ValueError):
            self.model(self.obs, self.act)

    def test_padding_cannot_change_valid_state_decisions(self):
        self.model.eval()
        before = self.model(self.obs, self.act)["reward"].detach()
        mask = ~self.obs["anchor_valid"]
        self.assertTrue(mask.any())
        self.obs["anchor"][mask] = 100000.
        after = self.model(self.obs, self.act)["reward"].detach()
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_controller_update_does_not_backpropagate_into_backbone_observations(self):
        self.obs["anchor"].requires_grad_()
        before = self.model.reward_head.weight.detach().clone()
        opt = torch.optim.Adam(self.model.parameters(), lr=.01)
        out = self.model(self.obs, self.act)
        loss = controller_loss(out, torch.ones_like(out["reward"]), torch.zeros_like(out["cost"]), self.act["legal"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(self.obs["anchor"].grad)
        self.assertFalse(torch.equal(before, self.model.reward_head.weight))


class SpecialistTrainingTests(unittest.TestCase):
    def test_dpo_pushes_preferred_output_up_without_training_reference(self):
        chosen = torch.tensor([0.], requires_grad=True)
        rejected = torch.tensor([0.], requires_grad=True)
        ref = torch.tensor([0.], requires_grad=True)
        loss = dpo_loss(chosen, rejected, ref, ref, beta=1.)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0.)
        self.assertGreater(rejected.grad.item(), 0.)
        self.assertIsNone(ref.grad)

    def test_response_log_probabilities_ignore_prompt_and_padding(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 0]])
        result = sequence_log_probs(logits, labels)
        self.assertAlmostEqual(result.item(), -2 * math.log(2), places=6)
        result.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0.)

    def test_only_lora_parameters_train_and_base_stays_frozen(self):
        model = nn.ModuleDict({"q_proj": nn.Linear(4, 4), "head": nn.Linear(4, 2)})
        frozen = model["q_proj"].weight.detach().clone()
        names = inject_lora(model, rank=2, alpha=4)
        self.assertEqual(names, ["q_proj"])
        self.assertTrue(all("lora_" in n for n, p in model.named_parameters() if p.requires_grad))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        loss = model["head"](model["q_proj"](torch.ones(2, 4))).square().sum()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(frozen, model["q_proj"].base.weight))
        self.assertIsNone(model["q_proj"].base.weight.grad)
        self.assertGreater(model["q_proj"].lora_B.abs().sum().item(), 0.)


class CollectionTests(unittest.TestCase):
    def test_collection_preserves_normalization_and_incomplete_cost_status(self):
        rt = make_runtime()
        prefix = rt.initial(fixtures.RuntimeTests().task())
        action = pick(rt, prefix, Kind.CONTINUE)
        policy = lambda s, legal: next(a for a in legal if a.kind == Kind.FINISH)
        with self.assertRaises(ValueError):
            collect_branches(rt, prefix, [action], policy, ExactMatch(), "1", 1, "snapshot", 12, 9)
        rows = collect_branches(rt, prefix, [action], policy, ExactMatch(), "1", 1, "snapshot", 12, 9,
                                allow_unmeasured_overhead=True)
        self.assertEqual(rows[0].reference_cost, 1000.)
        self.assertFalse(rows[0].cost_complete)
        self.assertEqual(len(rows[0].deployment_id), 64)

    def test_same_prefix_collection_excludes_sunk_cost_and_roundtrips(self):
        rt = make_runtime()
        prefix = rt.initial(fixtures.RuntimeTests().task())
        prefix = rt.step(prefix, pick(rt, prefix, Kind.HELP, target="s1", quota="short"))
        sunk = prefix.ledger.total
        actions = [a for a in rt.actions(prefix) if a.kind in (Kind.SUBMIT, Kind.ABORT)]
        policy = lambda s, legal: Decision(next(a for a in legal if a.kind == Kind.FINISH), 0., s.ledger.unit)
        rows = collect_branches(rt, prefix, actions, policy, ExactMatch(), "1", repeats=2,
                                snapshot_id="fixture-v1", anchor_dim=12, specialist_dim=9)
        self.assertEqual(prefix.ledger.total, sunk)
        self.assertEqual(len({r.prefix_id for r in rows}), 1)
        for row in rows:
            terminal = rt.run_from(rt.step(prefix, row.action), policy)
            self.assertAlmostEqual(row.cost, (terminal.ledger.total-sunk)/rt.config.reference_cost)
            self.assertEqual(row.reward, 1.)
            self.assertNotIn("gold", row.observation)
            self.assertTrue(row.synthetic)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.jsonl"
            write_targets(path, rows)
            loaded = read_targets(path)
            self.assertEqual(loaded[0].prefix_id, rows[0].prefix_id)
            self.assertEqual(loaded[0].action, rows[0].action)

    def test_preference_pairs_do_not_cross_inputs_and_skip_noisy_ties(self):
        outcomes = [StageOutcome("u1", "good", 1., .01), StageOutcome("u1", "bad", 0., .01),
                    StageOutcome("u2", "other", 3., .01)]
        pairs = preference_pairs(outcomes)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen, "good")
        self.assertEqual(preference_pairs([StageOutcome("u", "a", .5, .2),
                                           StageOutcome("u", "b", .49, .2)]), [])


if __name__ == "__main__":
    unittest.main()
