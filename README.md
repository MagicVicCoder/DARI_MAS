# DARI-MAS V1: dynamic local assistance

The frozen Anchor maintains an append-only global reasoning chain and proposes
open-ended natural-language assistance. A lightweight action-conditioned dual-value
Controller chooses Continue, SelfRepair, Help or Finish. A LoRA Specialist maintains
an isolated, bounded local chain; the Controller chooses Submit, Expand or Abort.
There are no fixed Specialist roles, operation classes, or LLM Controller.

Refactored from upstream `54786cebb133b49cab446ae8cae1cc07fc05a476`, on local branch
`dari-v1-dynamic-assistance`. This is engineering validation, not benchmark evidence.

## Protocol

```json
{"step":"New reasoning","refs":["p1"],"goal":"Unresolved subproblem","help_request":{"instruction":"Find another derivation","target":"current_step","required_refs":["p1"]},"answer":null,"invalidate_answer":false}
```

Runtime assigns s/g/h IDs, resolves current_step and validates references. Unknown
references cause billed bounded retry. Candidate answers are strings or null;
invalidate_answer explicitly revokes an obsolete answer. Null/missing requests use
a deterministic fallback with recorded origin/reason. SelfRepair and Help address
the same dynamic task. Finish returns an existing candidate without another call.

Every valid Specialist stage emits work and draft. Draft fields are target,
finding, support, change and unresolved; change means a general recommendation,
not necessarily an error correction. Narrow context includes question/constraints,
target/owner, required_refs and direct refs; Wide adds recent global records.
Expand retains valid private history; invalid expansion retains the last valid
draft. Submit passes only the draft; Abort sends nothing. Explicit Anchor citation
retains the original draft. Citation is not a measure of all actual information use.

## Run

The server executes Linux commands in `/data/users/csh/MAS_v1`. The local
`.venv` reuses Qwen3/PyTorch libraries and adds PEFT without changing shared Conda.
For a fresh environment, install appropriate PyTorch then `pip install -e '.[models]'`.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m dari.demo --expand --output outputs/synthetic/demo --collect outputs/synthetic/targets.jsonl
.venv/bin/python -m dari.train_controller --data outputs/synthetic/targets.jsonl --output outputs/synthetic/controller.pt --steps 3
.venv/bin/python -m dari.validate --config configs/qwen3_8b_debug.json --output outputs/qwen3/validation --train
```

Check GPU availability before reusing configured devices 4/7; do not stop others'
jobs. `dari.validate` runs twelve functional samples across forced policies,
including multiple sessions, and saves full inputs/outputs, model metadata and
random-controller forward scores. Its DPO pair is synthetic gradient-smoke data,
not a receiver-utility label. See the phase report for actual validation outcomes.

## Models, costs and training

`configs/qwen3_8b_debug.json` independently configures A/S devices, dtype, sampling
and native thinking. Dimensions/context limits come from actual models. Runtime
configures stage count/budgets, draft cap and history windows. Default maximum is
three stages; oneshot remains an ablation. Qwen3 native thinking is off by default.
Only actual model hidden states feed neural inference. Request instruction spans
are pooled from generated tokens. Fallback uses a labeled target-state proxy.
Text-only providers cannot deploy this hidden-state Controller.

Per-model token counts, latency and memory are diagnostics. Formal utility remains
`expected_reward - lambda * expected_normalized_future_cost`. Uncalibrated
`real_debug` uses explicit debug units and `cost_complete=false`; it cannot produce
formal utility-training labels. Never assert calibration to bypass checks.

Same-prefix counterfactual collection, Huber dual-head training, receiver-utility
DPO, protocol/auxiliary SFT and alternating optimization remain. Anchor stays
frozen; only Specialist LoRA trains. Preferences compare identical inputs/budgets
using actual downstream utility. Recollect labels after phi updates and refresh
the final Controller. Controller/data schema v2 rejects incompatible old artifacts;
SFT/DPO prompt-response containers retain schema v1.

```bash
python -m dari.collect --config configs/local.json --tasks data/train.jsonl --mode controller --output outputs/theta_data
python -m dari.train_controller --data outputs/theta_data/targets.jsonl --output outputs/theta.pt
python -m dari.collect --config configs/local.json --tasks data/train.jsonl --mode specialist --controller outputs/theta.pt --output outputs/phi_data
python -m dari.train_specialist --mode sft --model-config configs/specialist.example.json --data data/protocol_sft.jsonl --output outputs/sft_adapter
python -m dari.alternate --config configs/local.json --output outputs/training
```

`configs/local.json` and `data/train.jsonl` are researcher-supplied formal
configuration/data. Formal costs require measured profiles, policy overhead and
`calibration_confirmed=true`. No large-scale training is performed in phase 1.

## Evidence and maintenance

See [method](docs/METHOD.md), [model interface](docs/MODEL_INTERFACE.md),
[phase report](outputs/phase1/phase1_report.md) and [maintenance rules](AGENTS.md).
Original tests/commit: `outputs/baseline/`; real traces/adapters:
`outputs/qwen3/`; synthetic Controller: `outputs/synthetic/`.
Artifacts are ignored by Git and must be archived separately. Historical SFT
examples are synthetic fixtures, not a replacement for dynamic-task training data.

## Phase 2A exploratory E0/E1

Phase 1 was published to upstream `main` as commit `2faaaaa`. Phase 2A uses frozen
base Qwen3-8B weights without smoke adapters or learned-policy decisions. E0 compares
single-call Standard CoT, stepwise CoT without requests, and DARI Anchor-only. E1
replays one preselected unfinished prefix for Continue, SelfRepair, two-call
SelfRepair-Extended and single-stage Help-Submit. All branches have the same number
of new public continuation steps after intervention; extra integration belongs to
the intervention. This is approximate configured output-budget matching, not equal
measured GPU cost. Labels are exploratory scores, never formal utility labels.

```bash
# Data preparation requires the three source files listed in pilot_data.py.
.venv/bin/python -m dari.pilot_data
# Recheck GPU availability; A/S use independent instances on these two devices.
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage pilot
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage expand
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage robustness
.venv/bin/python -m dari.pilot_analysis
```

Install optional Phase 2A dependencies with `pip install -e '.[experiments]'`
(`math-verify==0.8.0`, `pyarrow`, `matplotlib`, `numpy`). Data source URLs,
content fingerprints and original rows are retained
under `data/phase2a/` and `outputs/phase2a/`; GSM8K reuses a local test copy,
MATH-500 is from HuggingFaceH4, HotpotQA is distractor validation. Development
samples are fixed before generation (8/20 per dataset for pilot/total). Do not
reuse them as independent tests after training on their trajectories.

The runner saves complete model inputs/outputs, prefixes, per-model token usage,
seeds and conditions, and resumes completed runs only under identical code,
prompt, evaluation, model and sample identities. Analysis creates CSV tables,
PNG/PDF figures and `outputs/phase2a/phase2a_report.md` from completed real runs;
that report states the actual finished scope. No accuracy threshold or collaboration
gain is presumed. Unparsed expressions remain flagged for manual review.
