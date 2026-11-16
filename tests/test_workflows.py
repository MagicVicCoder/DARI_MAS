import tempfile
import unittest
from pathlib import Path

import torch

from dari.demo import run_demo, collect_demo_targets
from dari.collection import read_targets
from dari.train_controller import train_from_targets

torch.set_num_threads(1)


class WorkflowTests(unittest.TestCase):
    def test_demo_exposes_synthetic_costs_and_writes_inspectable_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_demo(output_dir=Path(folder), expand=True)
            self.assertEqual(result["answer"], "1")
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["cost_unit"], "synthetic_units")
            self.assertEqual(result["tokens_by_model"]["S"]["calls"], 2)
            self.assertTrue((Path(folder)/"summary.json").exists())

    def test_collected_records_train_a_real_controller_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            data, checkpoint = Path(folder)/"targets.jsonl", Path(folder)/"controller.pt"
            collect_demo_targets(data)
            self.assertGreaterEqual(len(read_targets(data)), 6)
            result = train_from_targets(data, checkpoint, steps=4, dim=16)
            self.assertTrue(torch.isfinite(torch.tensor(result["final_loss"])))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertIn("model_state", saved)
            self.assertEqual(saved["config"]["anchor_dim"], 12)
            self.assertTrue(saved["data_metadata"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
