# DARI-MAS: native autoregressive reasoning

`dari-run` defaults to native reasoning. Standard CoT and Anchor-only use
the same token engine, prompt, sampling stream, budget and answer extraction.
The runtime observes generation every 256 tokens by default. It waits a
bounded number of tokens for a safe textual boundary before offering Help.
Continue passes the exact existing token IDs to the engine. The HF backend
replays prefill at chunk boundaries and caches within each chunk.

Use the `native` configuration object to set `checkpoint_interval`,
`boundary_wait`, `supporting_tokens`, `anchor_budget`, `specialist_budget`,
`stage_budget`, `draft_budget` and `stages`. Set `native_policy` to
`standard`, `anchor_only`, `help_once` or `forced`. The default policy is
Anchor-only. Example overrides for an existing HF deployment:

```json
{
  "reasoning_mode": "native",
  "native": {"delegation_mode": "controller_guided", "checkpoint_interval": 256},
  "native_policy": "forced",
  "forced_action": {"kind": "help", "target": "current_frontier", "mode": "verify"}
}
```

Run `python -m dari.run --config configs/local.json --tasks data/tasks.jsonl
--output outputs/native.jsonl --trace`. Simply set `delegation_mode` to
`frontier` (default) or `native_intent` and use `help_once` for the other
two mechanisms. See [method](docs/METHOD.md) and [model interface](docs/MODEL_INTERFACE.md).

The native controller adapter uses schema 4 and the existing dual heads.
Historical controller checkpoints and training collectors use the JSON
state/action schema. Native counterfactual datasets, SFT/DPO adaptation,
alternating optimization, cost calibration and real model validation are
pending. All experiments below describe the historical JSON runtime.
Set `reasoning_mode: legacy_json` for the old inference CLI.

## Historical JSON runtime

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

Protocol v3 Specialist output is `{"work":"actual local reasoning","draft":"actionable assistance"}`.
Draft is plain text; Runtime owns target, session and stage metadata. Draft quota
counts decoded text tokens, independently of the full generation budget. Narrow context includes question/constraints,
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

Phase 3A uses frozen base Qwen3-8B for both roles and fits the existing
dual-value Controller offline. The experiment stores profiling, paired snapshots,
development metrics and closed-loop traces in `outputs/phase3a/`.
Run on the same four-device placement for calibration and generation:

```bash
python -m dari.phase3a_cost
python -m dari.phase3a collect --stop 9
python -m dari.phase3a_audit
python -m dari.phase3a collect --start 9 --stop 36
python -m dari.phase3a collect --split dev --stop 12
python -m dari.phase3a_audit
python -m dari.phase3a_train
python -m dari.phase3a closed --stop 12
python -m dari.phase3a_report
```

Completed rollout snapshots are reused when collection restarts with the same
experiment identity. Calibration must be rerun for a different placement in a new
experiment directory. See the phase report for executed sample counts.

See [method](docs/METHOD.md), [model interface](docs/MODEL_INTERFACE.md),
[phase report](experiments/phase1/phase1_report.md) and [maintenance rules](AGENTS.md).
Original tests/commit: `outputs/baseline/`; real traces/adapters:
`outputs/qwen3/`; synthetic Controller: `outputs/synthetic/`.
Raw run outputs, caches, and weights are ignored by Git. Published evidence lives in experiments/ Historical SFT
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
# Reuse existing sources first; these fetch only missing local files.
mkdir -p data/phase2a
test -f data/phase2a/gsm8k.jsonl || curl -fL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o data/phase2a/gsm8k.jsonl
test -f data/phase2a/math500.jsonl || curl -fL https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl -o data/phase2a/math500.jsonl
test -f data/phase2a/hotpot_validation.parquet || curl -fL https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet -o data/phase2a/hotpot_validation.parquet
.venv/bin/python -m dari.pilot_data
# Recheck GPU availability; A/S use independent instances on these two devices.
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage pilot
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage expand
CUDA_VISIBLE_DEVICES=0,2 .venv/bin/python -m dari.pilot --stage robustness
.venv/bin/python -m dari.pilot_analysis
```

If two full instances no longer fit, the recorded Phase 2A robustness placement
uses one Anchor GPU and a Specialist sharded over two GPUs:
`CUDA_VISIBLE_DEVICES=7,4,6 PYTHONPATH=. .venv/bin/python scripts/phase2a_robustness.py`.
Its placement is saved separately and latency is not directly comparable to the
main two-GPU runs. Check available memory before using either command.

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

Completed Phase 2A: 60 development tasks, 180 E0 runs, 36 eligible prefixes with
144 E1 branches, and 64 robustness runs on 8 seeded checkpoints. Help versus
Continue yields 0 wins / 34 ties / 2 losses under the fixed evaluator. Only 14/36
Help drafts are submitted; 22 exceed the draft-token limit, and 26/36 requests
are fallback. These are diagnostic findings, not proof of collaboration benefit
or failure. Full scores, evaluator sensitivity, resource costs and limitations
are in the Phase 2A report. The regression suite has 71 passing tests.

## Phase 2A Refinement

Current Anchor history keeps step IDs, reasoning and references, with the current
candidate separate. Anchor solves the complete original question and proposes
concrete unresolved tasks; Controller still chooses the action. Specialist acts
as an independent local solver with separate original-question, evidence,
Anchor-context and session-private-history partitions. Defaults: three 512-token
stages and 384-token Draft; refinement counterfactuals use one stage.

Data and Controller checkpoint schema is now 3; historical v1/v2 training data
and policies require explicit regeneration. Historical experimental outputs remain
unchanged. Frozen Qwen3 refinement runs (no training or push):

```bash
CUDA_VISIBLE_DEVICES=6,7,2,3 python -m dari.refinement r0
CUDA_VISIBLE_DEVICES=0,1,5 python -m dari.refinement r1
CUDA_VISIBLE_DEVICES=6,7,2,3 python -m dari.refinement r2
CUDA_VISIBLE_DEVICES=0,1,5 python -m dari.pilot --config configs/refinement_r3.json --stage e0 --per-dataset 8
CUDA_VISIBLE_DEVICES=6,7,2,3 python -m dari.pilot --config configs/refinement_r3_branches.json --stage e1 --per-dataset 8
python -m dari.refinement_analysis
```

GPU placement must be reviewed against available memory. Outputs and the unified
report are under `outputs/phase2a_refinement/`; R0 replays historical generations,
R1 reuses original matched baselines, R2 reruns all four downstream controls, and
R3 uses fixed disjoint development samples. Results are exploratory, not training labels.

Completed refinement: R0 replays 12 historical outputs; R1 reruns 48 conditions;
R2 runs all 144 branches; R3 runs 72 E0 conditions and 56 branches on 14 eligible
prefixes. Help delivers 33/36 and 13/14 drafts in R2/R3. Help versus Continue is
2/32/2 and 1/13/0 wins/ties/losses. Anchor accuracy does not clearly improve;
some work remains advisory, with real integration failures and harmful assistance.
The unified report separates these mechanisms, score/parser limitations and the
small oracle gains. Training and automatic push remain disabled. Use committed
configs for replay; `refinement setup` is a first-time initializer and rejects
different existing configurations. R3 branch artifacts link to the immutable R3
E0 index and prefixes; preserve those links when archiving.

## Phase 2B protocol adaptation

`RuntimeConfig.repair_protocol_json` optionally applies conservative syntax-only
repair before parsing: whole JSON fences and illegal JSON escape characters.
It never completes truncated records, missing fields, reasoning or answers.
Raw output, actual token usage and generated hidden features remain unchanged;
strict mode remains the default. Phase 2B enables the same repair for all methods.
`dari.phase2b_evaluation` uniformly extracts one explicit final math expression;
historical scores are retained and sensitivity rescoring is separate.

Protocol SFT uses `train_specialist` with response-only masks, optional heldout
validation loss, epoch-order batches, progress logs and an explicit training device
map. Validation must remain independent of training instances; synthetic task
families are not a test of heldout-template generalization. Only LoRA is trained.

Phase 2B artifacts are in `outputs/phase2b/`: data/source/leakage manifests,
200 constructed training and 40 validation tasks, actual adapter and losses,
T0 Base/SFT outputs, new 36-task Anchor trajectories, five-way same-prefix T1,
and paired same-Specialist-state Submit/Abort T2. The unified report distinguishes
bounded result-match proxies from reviewed semantic behavior. No Controller or
Receiver-Utility/DPO training, no automatic push and no calibrated-cost claims.

```bash
CUDA_VISIBLE_DEVICES=4,6 python -m dari.phase2b t0
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m dari.phase2b downstream
python -m dari.phase2b_analysis
```

Commands resume existing outputs; preserve manifests and adapter identity. Inspect
current free GPU memory before replay. The setup/data commands initialize new
artifacts and must not overwrite fixed datasets or experiments.

Phase 2B completed: 50 LoRA optimizer steps (two epochs), 80 T0 generations,
90 paired T1 branches on 18 eligible prefixes and 17 T2 pairs. T0 Draft target
matches decline from 32/40 to 20/40; these are limited proxies, with qualitative
review confirming regressions. Help-Base and Help-SFT both score13/18, paired
0 wins/18 ties/0 losses. T2 Submit-Abort has3 positive/14 neutral/0 negative
outcomes. Oracle D-C=0: no observed extra SFT coverage over Base. Training loss
improves without demonstrated collaboration improvement. See the Phase 2B report
for format artifacts, exactness effects and restricted synthetic data coverage.
The trained adapter remains local at `outputs/phase2b/adapter`. Training replay
uses `python -m dari.phase2b_train` and refuses to overwrite this valid adapter.

The Phase 3A closed-loop watcher (`scripts/wait_phase3a_closed.sh`) polls all GPUs
and starts on four cards with at least 12 GiB free and at most 10% utilization.
On this server all eight cards have the same model. The original calibration
was measured on physical GPUs 0, 1, 2 and 6. Each closed-loop result records its
physical GPU assignment so elapsed time can be interpreted with placement in view.
## Phase 3B collaboration experiments

Run the historical export with `python -m dari.phase3b_history`. Run the paired
screen with `CUDA_VISIBLE_DEVICES=2,6,7 python -m dari.phase3b`.
The screen places Anchor on one device and shards Specialist over the other two.
Results and complete branch traces accumulate in `outputs/phase3b_collaboration/`.

Run `CUDA_VISIBLE_DEVICES=1,3 python -m dari.phase3b_receiver` in parallel
to replay saved Phase 3A development drafts. This worker loads only Anchor,
sharded across two GPUs, and records its placement in
`integration_history/manifest.json`. Compare rewards within each saved state.
Elapsed times are descriptive under concurrent GPU load.

The Phase 3B supplement uses `python -m dari.phase3b --supplement matched_sampling`
to replay screening prefixes with greedy decoding in both roles. Use
`python -m dari.phase3b --supplement combined_validation` for a new 24-task pool
with paired Directed, Autonomous Independent, Critical Integration, and internal
controls. Each supplement records its own manifest and resumable results.
## Challenging benchmarks

`python -m dari.challenging prepare` records seeded samples and dataset hashes
for MATH-500, HotpotQA, HumanEval+ v0.1.10 and the pinned LiveCodeBench test6 shard.
Run `python -m dari.challenging math500 --limit 3` for the initial compatibility
check, with four visible GPUs in the configured logical order. The same command
accepts `hotpotqa`, `humaneval` and `livecodebench`. Results and prompts are written
under `outputs/phase3b_collaboration/challenging_benchmarks/`. Run
`python -m dari.challenging summarize` for sample counts, checkpoint coverage and
available scores. The scoring launcher writes official grades alongside the
generation records.

With enough free memory for one BF16 Qwen3-8B instance on each of two
visible GPUs, pass `--two-devices`. This writes a separate deployment
manifest and resumes from the existing per-task results.

For single GPU inference, `--shared-base` loads one frozen BF16 model with
separate role sampling settings. Each generation starts with an empty KV cache.
The placement and sharing choice are recorded in `manifest_shared_base.json`.

### Offline code scoring

Install EvalPlus 0.3.1 in `.work/evaldeps` with
`python -m pip install --target .work/evaldeps evalplus==0.3.1`.
Check out LiveCodeBench commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`
under `.work/lcb-pinned`. With Bubblewrap and system Python 3.10, run
`python scripts/score_challenging.py humaneval` and
`python scripts/score_challenging.py livecodebench`. Each command consumes
saved generations and writes official scores. The release evidence is in
[experiments](experiments/).
