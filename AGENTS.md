# DARI-MAS development and experiments

Work only inside /data/users/csh/MAS_v1. Preserve upstream Git history and existing
uncommitted work; do not push automatically or modify original model weights.
Anchor proposes open-ended assistance; the lightweight Controller decides whether
it is worthwhile; Specialist performs it. No fixed specialist roles, extra LLM
controller/verifier, or unapproved research mechanisms. Anchor stays frozen;
Specialist trains LoRA only. Preserve same-prefix and same-input comparisons.

Before and after every experiment inspect temporary scripts, caches, intermediate
data, duplicate results/configurations, unused checkpoints, imports, dependencies,
old prompts and superseded documentation. Put temporary work in .work/. Remove
only items confirmed unused and recoverable; check references, active processes,
research value and recovery before deletion. Do not use indiscriminate recursive
deletions. Keep useful baselines explicitly labeled. Protect formal datasets,
valid weights, original results, losses, seeds, configurations, trajectories,
ablations and unfinished analyses. Never clean files outside this project.

Update README.md whenever architecture, schemas, actions, training, interfaces,
configuration or commands change. Keep current method in docs/METHOD.md and model
interface in docs/MODEL_INTERFACE.md; update these in place. Use Git for history,
not backup source/document copies. Retain independently useful phase reports and
experimental artifacts; remove reports only when fully superseded without unique
research value. Distinguish implemented, tested, unverified and planned features.

Every task must end with relevant regression tests, Git diff/status inspection,
.gitignore and directory review, documentation updates, a concise account of added,
changed and deleted files, artifact paths and unresolved issues. Do not weaken
assertions, skip important tests or fabricate outputs to meet acceptance criteria.
Keep weights, caches, large logs and datasets out of source control. Commit tested
coherent changes locally when useful; never overwrite important pending work.

All randomness needs seeds and model/adapter/policy identities. Keep audit logs
separate from model state. Mark synthetic, real-model debug and formal experiments.
Do not use uncalibrated resource labels for formal utility training. Do not stop
other researchers' GPU jobs. Run original tests before refactoring and phase tests
before progressing. Code/documents evolve in place; formal results accumulate;
temporary artifacts are cleaned promptly.
