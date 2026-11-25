# Model interface

Backend retains count_tokens, count_text_tokens and generate(GenerationRequest).
Generation returns text, actual Usage, named Features and optional latency. Usage
separates per-model input/output/cache tokens. CallEvent records prompt, protocol
text, validity/error, usage and latency. Native thinking is removed from protocol
text but all emitted tokens remain billed.

HuggingFaceBackend independently loads A/S model/tokenizer objects, freezes them,
applies their chat templates and optional adapters, and checks context bounds.
Settings: name_or_path/model_path, dtype, device/device_map, temperature, top_p,
chat_template_kwargs/enable_thinking, context_limit and deployment_revision.
Local files load without download; remote code is disabled. Local immutable
deployment_revision is researcher-supplied; archive original weight checksums for
formal reproducibility. Tokenizer/template/adapter/sampling identities are recorded.

To replace Anchor edit backend.models.A; to replace Specialist edit backend.models.S.
Use tokenizer to select a separate tokenizer path. Do not set hidden sizes manually
for HF: build_runtime discovers and verifies them. Set device to cuda:N and dtype
to bfloat16/float16/float32; device_map is inference-only. Set adapter to a PEFT
adapter path or use top-level model_adapters keyed by model ID. Retrain the
Controller and recalibrate costs when replacing the backbone or tokenizer.

Actual hidden_size and max_position_embeddings determine dimensions/default
limits. Named JSON input ranges disambiguate repeated text. Normal cached forwards
pool final-layer states without extra encoding or cross-call KV. Output JSON values
are located by a decoder and aligned to decoded original token prefixes. An emitted
token not yet forwarded is excluded from pooling but remains charged. Request
pooling selects its instruction span. Fallback has no generated request span:
the target proxy and fallback indicator are explicit, not fabricated activations.
Scripted hash vectors exist only in synthetic fixtures. Text-only backends cannot
deploy NeuralPolicy.

Qwen3 debug uses the inspected template's enable_thinking=false. Full model config
and template are saved in the validation manifest. Enabling thinking separates
text after </think>; all generation remains billed and affects output pooling.
Real thinking-mode validation is still required before formal experiments.

Role/call/episode seeds isolate random streams. PEFT training owns Specialist and
optimizes default LoRA only. Fixed reference adapter scores precede policy updates.
Reports include per-step losses, gradient/update norms, LR and frozen tensor-version
checks. Ordinary-Linear LoRA fixtures are not PEFT checkpoint formats.

Formal cost labels need calibrated profiles and policy overhead. real_debug uses
uncalibrated scheduling units with cost_complete=false; latency is not GPU resource
cost. Budget estimates are guards rather than hard time/resource guarantees.

## Current refinement protocol (v3)

Specialist emits exactly two nonempty strings, work and draft. work contains actual
local reasoning; draft contains directly usable findings, supporting calculations
or evidence and remaining uncertainty as needed. Runtime owns all target/session/
stage metadata. The online parser rejects historical object drafts; a read-only
legacy parser supports archival inspection. Submit exposes only draft text.

Anchor history serializes only IDs, steps and refs; the current candidate is separate.
Full original questions always reach Specialist, separately from selected evidence,
fallible Anchor reasoning and isolated session-private outputs. Narrow/Wide selection
remains deterministic and is chosen before scoring. Draft text cap defaults to 384;
full output caps default to [512,512,512]. Experiment branches use a single stage.
Input features track exact partition value spans; output Draft pooling excludes JSON
string quotes. SFT/DPO data and Controller checkpoints use schema 3. Regenerate
older data/policies rather than silently mixing incompatible representations.
The Phase 2A sections above describe the historical experimental protocol.

Completed refinement inference uses independent frozen BF16 Qwen3-8B objects.
R1/R3 E0 split Anchor across physical GPUs 0/1/5 and keep unused Specialist on
CPU; paired R2/R3 branches split Anchor across 6/7 and Specialist across 2/3.
All controls within a paired checkpoint share the latter deployment. Exact maps,
sampling, tokenizer/template and code identities reside in per-stage manifests.
The R3 branch directory links to original immutable E0 indexes and prefix files;
archive links with their targets. A failed concurrent model load occurred before
R3 branch inference; it produces no scored condition. Subsequent 56 branches
complete after resource release. This does not calibrate GPU resource costs.
