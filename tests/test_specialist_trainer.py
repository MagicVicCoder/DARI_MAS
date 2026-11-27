from copy import deepcopy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from dari.hf_backend import PromptCodec
from dari.specialist_data import response_batch, read_records, write_records
from dari.train_specialist import optimize_specialist
from dari.training import inject_lora
from tests.model_fixtures import CharacterTokenizer, TrainableCausalLM

torch.set_num_threads(1)


class SpecialistTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.codec = PromptCodec(CharacterTokenizer())
        self.model = TrainableCausalLM()
        inject_lora(self.model, rank=2, alpha=4)

    def test_prompt_padding_and_eos_labels_match_generation_boundary(self):
        batch = response_batch(self.codec, [("x", "a"), ("long", "bc")], max_length=100)
        # prefix for x is <user>x<assistant>, length 18; response is a, EOS.
        self.assertTrue((batch["labels"][0, :18] == -100).all())
        self.assertEqual(batch["labels"][0, 18:20].tolist(), [100, 2])
        self.assertTrue((batch["labels"][0, 20:] == -100).all())
        self.assertEqual(batch["attention_mask"][0].sum().item(), 20)
        with self.assertRaisesRegex(ValueError, "length|truncat"):
            response_batch(self.codec, [("long", "bc")], max_length=10)
        with self.assertRaises(ValueError):
            response_batch(self.codec, [("x", "")], max_length=100)

    def test_sft_updates_only_adapters_using_actual_tensor_forward(self):
        frozen = self.model.q_proj.base.weight.detach().clone()
        before = self.model.q_proj.lora_B.detach().clone()
        result = optimize_specialist(self.model, self.codec,
            [{"prompt": "x", "response": "a"}], mode="sft", steps=3, lr=.03, max_length=100)
        self.assertEqual(result["steps"], 3)
        self.assertTrue(math.isfinite(result["final_loss"]))
        self.assertTrue(torch.equal(frozen, self.model.q_proj.base.weight))
        self.assertFalse(torch.equal(before, self.model.q_proj.lora_B))

    def test_dpo_uses_frozen_reference_and_increases_preference_margin(self):
        reference = deepcopy(self.model)
        before_ref = {k: v.clone() for k, v in reference.state_dict().items()}
        result = optimize_specialist(self.model, self.codec,
            [{"prompt": "x", "chosen": "a", "rejected": "b"}], mode="dpo",
            reference_model=reference, steps=12, lr=.03, beta=.3, aux_weight=0., max_length=100)
        self.assertGreater(result["final_margin"], result["initial_margin"])
        self.assertTrue(all(torch.equal(before_ref[k], v) for k, v in reference.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_data_reader_rejects_mixed_provenance_and_conflicting_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"sft.jsonl"
            rows = [{"input_id": "same", "prompt": "x", "response": "a", "synthetic": True},
                    {"input_id": "same", "prompt": "y", "response": "a", "synthetic": True}]
            write_records(path, rows)
            with self.assertRaisesRegex(ValueError, "input|prompt"):
                read_records(path, "sft")
            rows[1]["input_id"] = "different"
            rows[1]["synthetic"] = False
            write_records(path, rows)
            with self.assertRaisesRegex(ValueError, "provenance|synthetic"):
                read_records(path, "sft")

    def test_epoch_order_validation_and_progress_do_not_update_frozen_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            progress=Path(folder)/'progress.jsonl'
            before=self.model.q_proj.base.weight.detach().clone()
            result=optimize_specialist(self.model,self.codec,
                [{'prompt':'x','response':'a'},{'prompt':'y','response':'b'}],mode='sft',
                steps=2,batch_size=1,accumulation=2,epoch_order=True,
                validation_records=[{'prompt':'z','response':'c'}],progress_path=str(progress),max_length=100)
            self.assertEqual([r['step'] for r in result['validation_losses']],[0,1,2])
            self.assertTrue(all(math.isfinite(r['loss']) for r in result['validation_losses']))
            self.assertEqual(len(progress.read_text().splitlines()),2)
            self.assertTrue(torch.equal(before,self.model.q_proj.base.weight))
            self.assertTrue(result['frozen_parameters_unchanged'])


if __name__ == "__main__":
    unittest.main()
