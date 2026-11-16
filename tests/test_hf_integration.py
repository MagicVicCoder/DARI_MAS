"""Optional real-library integration, random tiny weights only; no Hub download."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dari.backends import GenerationRequest
from dari.hf_backend import HuggingFaceBackend
from dari.specialist_data import write_records
from dari.train_specialist import train_from_file

HAS_MODELS = bool(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"))


def write_tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    special = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>"]
    vocabulary = {token: i for i, token in enumerate(special+sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>",
        unk_token="<unk>", eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>"])
    fast.chat_template = "{% for m in messages %}{{ '<' + m['role'] + '>' + m['content'] }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    fast.save_pretrained(path)
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(fast), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=1024, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id))
    model.save_pretrained(path)


def valid_response(finding):
    return json.dumps({"work":"Check sign", "draft":{"target":"s1","finding":finding,
        "support":"Nonnegative root", "change":"Use 1", "unresolved":""}})


@unittest.skipUnless(HAS_MODELS, 'Install .[models] for real Transformers/PEFT integration')
class RealLibraryTests(unittest.TestCase):
    def test_hf_loader_and_cached_forward_with_actual_qwen2_class(self):
        with tempfile.TemporaryDirectory() as folder:
            write_tiny_model(folder)
            backend = HuggingFaceBackend.from_specs({"A":{"name_or_path":folder,
                "deployment_revision":"random-tiny-fixture-v1", "dtype":"float32", "device":"cpu", "context_limit":1024}})
            result = backend.generate(GenerationRequest("A", "anchor", "Check sign", 4, 0,
                {"task":"Check sign"}, feature_spans={"task":(0,10)}))
            self.assertGreater(result.usage.output_tokens, 0)
            self.assertLessEqual(result.usage.output_tokens, 4)
            self.assertEqual(len(result.features["task"].values), 16)
            self.assertEqual(len(result.features["output"].values), 16)
            self.assertIn("random-tiny-fixture-v1", str(backend.deployment_identity))

    def test_real_peft_sft_and_fixed_reference_dpo_save_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root/"base"
            base.mkdir()
            write_tiny_model(base)
            spec = {"name_or_path":str(base), "dtype":"float32", "device":"cpu"}
            sft, preference = root/"sft.jsonl", root/"preferences.jsonl"
            write_records(sft, [{"input_id":"toy", "prompt":"Check sign", "synthetic":True,
                                 "response":valid_response("Correct result")}])
            warm = train_from_file(sft, root/"warm", spec, steps=2, lr=.01, max_length=512,
                                   lora_rank=2, lora_alpha=4, gradient_checkpointing=False)
            self.assertTrue((root/"warm"/"adapter_config.json").exists())
            self.assertTrue(warm["synthetic"])
            write_records(preference, [{"input_id":"toy", "prompt":"Check sign", "synthetic":True,
                "chosen":valid_response("Correct result"), "rejected":valid_response("Wrong result")}])
            trained = train_from_file(preference, root/"dpo", spec, mode="dpo",
                init_adapter=str(root/"warm"), reference_adapter=str(root/"warm"), steps=2,
                lr=.01, max_length=512, gradient_checkpointing=False)
            self.assertTrue((root/"dpo"/"adapter_model.safetensors").exists())
            self.assertTrue(torch.isfinite(torch.tensor(trained["final_loss"])))
            backend = HuggingFaceBackend.from_specs({"S":{**spec, "adapter":str(root/"dpo"),
                "deployment_revision":"random-tiny-fixture-v1", "context_limit":1024}})
            self.assertNotEqual(backend.deployment_identity["adapters"]["S"], "base")


if __name__ == "__main__":
    unittest.main()
