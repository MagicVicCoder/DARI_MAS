# DARI-MAS V1 method

Anchor owns the global chain. Its checkpoint includes step, existing refs, optional
goal, dynamic help request, candidate and explicit invalidation. Runtime allocates
s/g/h IDs and resolves current_step. History appends; invalidation precedes answer
replacement. Runtime fallback metadata never enters the model output schema.

Continue advances the global task. SelfRepair executes the current dynamic request
with full global history. Help delegates the same request using narrow/wide context
and short/oneshot quota. Finish needs a candidate and invokes no LLM. Null requests
receive a deterministic generic inspection task, labeled with origin and reason.
Its semantic proxy is the target's actual hidden representation plus a fallback
indicator; it is not represented as generated request activations.

Narrow includes the complete question/constraints, target/owner, required_refs and
direct references. Wide adds configurable recent records and their direct refs.
Context source IDs are deduplicated and logged. Unrelated source documents are not
automatically included. Excessive Specialist input masks the action; models do not
truncate prompts. Necessary constraints must be in the original question or refs.

Help opens empty private history. Each valid stage adds its complete work and draft
within that session only. Configured stage caps bound expansion. Expand leaves
Anchor frozen. Invalid stages are charged but excluded from private history and do
not replace the last valid draft/features. Latest legal draft is preferred without
a monotonic-quality assumption. No quality-based draft selection is implemented.

Submit passes only the draft as temporary hN evidence. Explicit citation retains
its original content for later prompts. Anchor may incorporate content without
citation, so citation rate is not information-use rate. Abort passes no private
content. Both clear helper state/features and retain sunk costs. Logs do not enter
prompts. Retained evidence can re-enter future sessions by context selection.

Controller preserves the lightweight two-block Transformer, action-conditioned
reward/future-cost heads and Rhat-lambda*Khat utility. Model-specific projections
accept actual dimensions. Observations include task, configurable recent global
records, goal, request, candidate, local stages/draft and numeric budget/call/session
features. Missing features are masked. Semantic request projection replaces the
operation embedding. The serialized Action.template slot remains empty, solely
for container compatibility. Controller compression is not Anchor's full history
and carries no Markov guarantee.

Same-prefix branches deep-copy history, candidate, request, cost and feature state;
seeds isolate continuations. Labels subtract prefix costs and include all future
calls/overheads. Schema-v2 rows record requests, seeds and deployment/policy
snapshots. Huber heads predict terminal reward and normalized future cost.
Specialist candidates share exact input/history/quota; executed downstream utility
and uncertainty-aware gaps determine preferences. Context/quota selection varies
rather than fixing narrow-short-check. Coverage still depends on visited prefixes.

Alternation retains protocol SFT, frozen-phi theta updates, frozen-theta LoRA
DPO/SFT, recollection and final theta refresh. Reference scores are fixed before
optimization. Uncalibrated debug labels cannot enter formal utility training.
Synthetic pairs validate plumbing only. Budget-matched SelfRepair/Help comparisons
can set equal output caps but still require measured repeated-input/resource costs.

Limits and exhaustion terminate with explicit reasons. One Anchor formatting retry
is billed. Forced policies cover paths but cannot establish trained-policy quality.
There are no claims of convergence, efficiency or benchmark accuracy.

## Phase 2A controlled exploratory comparisons

`generate_help_request=false` removes request generation and request fields from
Anchor history prompts while retaining the same Runtime/parser for stepwise CoT.
The default remains true. Structured prompts explicitly request substantive new
progress and concise final answers. This protocol adjustment applies uniformly
to stepwise/DARI conditions; Standard CoT uses the same task/sources and asks for
a final answer marker. All prompts and identities are archived before execution.

The exploratory runner chooses the first valid unfinished checkpoint supporting
all comparisons, before observing branch outcomes. No candidate means Finish is
unavailable. First-step completed tasks have no E1 prefix. Saved state hashes,
deep copies and post-branch integrity checks protect same-prefix comparisons.

Continue and SelfRepair each have one initial Anchor step; SelfRepair-Extended
has a local-task step and a refinement/integration step with the SAME original
request; Help-Submit has one Specialist stage and one Anchor integration call.
Each output cap is 512 in the initial config. Missing/invalid Draft invokes Abort
and its actual Anchor continuation without dropping the failed branch. After
intervention, every condition receives at most eight additional valid Anchor
steps and finishes as soon as a candidate exists. Absolute max-step counts are
adjusted per intervention so extra internal steps do not reduce common opportunity.

Same-prefix scores report raw terminal reward and actual resources separately;
uncalibrated costs are not promoted to formal Controller/DPO labels. Local Oracle
is a finite-rollout, post-hoc exploratory upper envelope, not deployable behavior.

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

Refinement verification completes 48 revised E0 runs on the original pilot,
144 paired branches on the original 36 prefixes, and 72 E0 runs plus 56 branches
on 24 new tasks (14 eligible prefixes). Help delivery is 33/36 and 13/14;
Help versus Continue wins/ties/losses are 2/32/2 and 1/13/0. External oracle
marginal successes B-A / D-C are 2/1 and 1/1. These are exploratory findings:
Anchor accuracy does not clearly improve, some private work remains advisory,
and score/parser artifacts and harmful integration remain. See the unified
refinement report and immutable traces before interpreting mechanism or training.

## Phase 2B: protocol adaptation

Phase 2B keeps Anchor frozen, LoRA Specialist only, and fixed action policies.
Conservative syntax repair is optional and uniform across comparisons; missing
fields, invalid types and truncated outputs remain failures. Math final-expression
extraction is versioned separately, with historical sensitivity results preserved.
SFT uses 200 explicitly constructed supervised local tasks and 40 distinct-instance
validation tasks; the ten task families overlap across splits and this limitation
is explicit. No benchmark development trajectories enter supervision.

T0 pairs identical Base/SFT local inputs, seeds and budgets. Expected-result matches
are bounded lexical/numeric proxies; qualitative inspection checks actual work,
correctness and advisory behavior without an extra judge. T1 uses five same-prefix
conditions: Continue, Repair, Extended, Help-Base and Help-SFT. LoRA disabling/enabling
changes only Specialist; Anchor is a separate immutable model. Common continuation
stops at a candidate with the same new-step allowance. T2 reuses the exact legal
SFT-side state before Submit/Abort, bills identical sunk generation cost and reports
future resources separately. No further Help occurs. Its post-hoc outcome difference
does not establish that a deployed Controller can recognize helpful information.
Oracle A/B/C/D follow Phase 2B's internal/base/SFT action-set definitions, rather
than the earlier refinement set naming. No cost proxy enters formal labels.

Completed Phase 2B observes T0 degradation after this single synthetic-data SFT
configuration and no T1 gain over Base (both13/18, all18 paired rewards equal).
T2 has3 beneficial/14 neutral/0 harmful single-rollout outcomes; examples include
answer-format paths and exact-expression scoring. These do not establish learned
communication selection. No Controller or receiver-driven optimization is justified
by these results alone. The next diagnostic should improve independent, diverse,
verified local-task supervision and test copying/context sensitivity before formal
utility training, preserving this negative result rather than tuning on its cases.
