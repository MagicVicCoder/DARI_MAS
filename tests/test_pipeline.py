from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dari.configuration import build_runtime, load_dataset, make_policy
from dari.alternate import run_alternating
from dari.specialist_data import read_records
from dari.train_specialist import optimize_specialist
from dari.hf_backend import PromptCodec
from dari.training import inject_lora
from tests.model_fixtures import CharacterTokenizer, TrainableCausalLM
from tests.test_receiver_collection import ReceiverBackend


def receiver_factory(config):
    return ReceiverBackend()


def tiny_trainer(data, output, model_spec, mode="sft", init_adapter=None, reference_adapter=None,
                 aux_data=None, **settings):
    torch.manual_seed(5)
    model = TrainableCausalLM()
    inject_lora(model, rank=2, alpha=4)
    if init_adapter:
        model.load_state_dict(torch.load(Path(init_adapter)/"weights.pt", weights_only=True))
    reference = None
    if mode == "dpo":
        reference = deepcopy(model)
        reference.load_state_dict(torch.load(Path(reference_adapter)/"weights.pt", weights_only=True))
    metrics = optimize_specialist(model, PromptCodec(CharacterTokenizer()), read_records(data, mode), mode,
        reference_model=reference, steps=1, lr=.01, max_length=8192)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output/"weights.pt")
    (output/"training_manifest.json").write_text(json.dumps({"mode": mode, "synthetic": True}))
    return {**metrics, "adapter": str(output)}


def fixture_config(tasks, sft):
    return {"backend": {"type": "factory", "entrypoint": "tests.test_pipeline:receiver_factory"},
        "runtime": {"max_anchor_steps": 4, "max_decisions": 8},
        "profiles": [{"model_id": k, "unit": "synthetic_units", "uncached_input": .1,
            "output": .2, "provenance": "test fixture", "synthetic": True} for k in ("A", "S")],
        "observation": {"anchor_dim": 12, "specialist_dim": 9},
        "policy_overhead": {"cost": 0., "unit": "synthetic_units", "provenance": "fixture"},
        "training": {"tasks": str(tasks), "sft_data": str(sft), "rounds": 1,
            "controller": {"steps": 100, "dim": 16, "lr": .01},
            "specialist": {"model_spec": {"name_or_path": "test-only"}, "steps": 1},
            "collection": {"prefixes_per_task": 2, "actions_per_prefix": 8, "repeats": 1,
                "candidates": 2, "epsilon": 0., "min_gap": .00001, "early_probe": True}}}


class PipelineTests(unittest.TestCase):
    def test_factory_runtime_keeps_gold_out_of_online_task(self):
        with tempfile.TemporaryDirectory() as folder:
            tasks = Path(folder)/"tasks.jsonl"
            tasks.write_text(json.dumps({"task_id": "t", "question": "question", "gold": "secret-answer"})+"\n")
            rows = load_dataset(tasks, require_gold=True)
            self.assertFalse(hasattr(rows[0].task, "gold"))
            cfg = fixture_config(tasks, tasks)
            rt = build_runtime(cfg)
            result = rt.run(rows[0].task, make_policy(rt, cfg))
            self.assertNotIn("secret-answer", result.events[0].prompt)

    def test_alternation_emits_fresh_final_controller_after_adapter_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks, sft = root/"tasks.jsonl", root/"sft.jsonl"
            tasks.write_text(json.dumps({"task_id": "t", "question": "sqrt(x*x), x=-1",
                "sources": {"p1": "x=-1"}, "gold": "1"})+"\n")
            response = {"work": "Check sign", "draft": {"target": "s1", "finding": "Correct local result",
                "support": "A root is nonnegative", "change": "Use 1", "unresolved": ""}}
            sft.write_text(json.dumps({"schema_version": 1, "input_id": "warm", "prompt": "Check sign",
                "response": json.dumps(response), "synthetic": True})+"\n")
            report = run_alternating(fixture_config(tasks, sft), root/"run", specialist_trainer=tiny_trainer)
            self.assertEqual(report["rounds_completed"], 1)
            self.assertTrue(Path(report["controller"]).exists())
            self.assertTrue((Path(report["specialist_adapter"])/"weights.pt").exists())
            self.assertTrue(report["synthetic"])
            self.assertEqual(report["final_controller_adapter"], report["specialist_adapter"])
            self.assertNotEqual(report["initial_phi_snapshot"], report["final_phi_snapshot"])


if __name__ == "__main__":
    unittest.main()
