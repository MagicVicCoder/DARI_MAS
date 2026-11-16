import unittest
import torch

from dari.backends import GenerationRequest
from dari.hf_backend import HuggingFaceBackend, ModelHandle, PromptCodec
from tests.model_fixtures import CharacterTokenizer, DeterministicCachedLM


class ModelBoundaryTests(unittest.TestCase):
    def test_input_count_includes_chat_template_but_payload_count_does_not(self):
        codec = PromptCodec(CharacterTokenizer())
        self.assertEqual(codec.count_input("hi"), 19)
        self.assertEqual(codec.count_text("hi"), 2)

    def test_generation_pools_existing_forward_activations_and_restores_rng(self):
        model = DeterministicCachedLM()
        codec = PromptCodec(CharacterTokenizer())
        backend = HuggingFaceBackend({"A": ModelHandle(model, codec, context_limit=512, identity={"model":"tensor-fixture-v1"})})
        random_state = torch.random.get_rng_state().clone()
        result = backend.generate(GenerationRequest("A", "anchor", 'task="a\\nb"', 4, 0,
                                                   {"task": "a\nb"}, seed=13, feature_spans={"task":(6,10)}))
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.usage.output_tokens, 3)  # includes EOS
        self.assertEqual(result.usage.input_tokens, len('<user>task="a\\nb"<assistant>'))
        self.assertEqual(model.calls, 3)  # prefill + two decode steps; no feature re-encode
        self.assertIn("task", result.features)
        self.assertEqual(result.features["task"].source, "anchor")
        self.assertEqual(len(result.features["output"].values), 4)
        self.assertTrue(torch.equal(random_state, torch.random.get_rng_state()))
        self.assertFalse(model.marker.requires_grad)

    def test_context_overflow_is_rejected_without_silent_truncation(self):
        model = DeterministicCachedLM()
        backend = HuggingFaceBackend({"A": ModelHandle(model, PromptCodec(CharacterTokenizer()), context_limit=20,
                                                       identity={"model":"tensor-fixture-v1"})})
        with self.assertRaisesRegex(ValueError, "context"):
            backend.generate(GenerationRequest("A", "anchor", "long input", 4, 0))
        self.assertEqual(model.calls, 0)

    def test_duplicate_model_objects_cannot_share_mutable_adapters(self):
        model = DeterministicCachedLM()
        handle = ModelHandle(model, PromptCodec(CharacterTokenizer()))
        with self.assertRaisesRegex(ValueError, "shared|same"):
            HuggingFaceBackend({"A": handle, "S": handle})


if __name__ == "__main__":
    unittest.main()
