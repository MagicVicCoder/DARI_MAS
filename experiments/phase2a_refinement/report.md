# DARI-MAS Phase 2A-Refinement 实验报告

状态：实验运行齐全；见下方实际统计及审计。

## 范围与可复现性

冻结原始 Qwen3-8B Anchor/Specialist，无适配器、训练、额外 Agent 或 LLM Judge。本轮为探索性开发复验，非正式独立测试、显著性检验或方法有效性结论。旧结果保持不变；所有失败进入统计。
模型路径 `/data/models/Qwen3-8B`，权重身份沿用原 manifest。每阶段 manifest 记录配置、逐文件代码哈希、样本指纹及独立 A/S 部署身份。选择种子 20261125；R0/R2 先固定随机 12/原有 36 个前缀，R1 使用原 pilot 每集 8 题；R3 每集 8 题，与原 60 题不重复，在推理前固定。HotpotQA 原文与离线 gold 分离，不将 gold/supporting facts 传模型。

## 根因分析

旧 36 个 Specialist 均输出完整 JSON，work 全部是占位字面量 `private local analysis`。代码不自动填充该值，提示伪示例被模型照抄。22 次拒绝均在解析后进行完整序列化 Draft 配额检查时发生，而非生成截断。原题 36/36 完整存在；“缺少售价”的案例属于未正确利用原题，不能归因于原题被删。

旧 Draft compact 长度 min/median/max={'min': 105, 'median': 205.0, 'max': 352}。离线预算可交付数量：{'192': 14, '256': 30, '384': 36, '512': 36} /36。预算放宽只说明长度合法，不能推出内容正确或下游受益。原自主请求 10 个，fallback 26 个，均来自合法 null/missing；非解析器把错误请求静默改成 fallback。Anchor 失败初筛：{'reasoning_or_premature_answer_requires_review': 13, 'format_failure': 9, 'step_limit': 1}。其中有效但错误的答案还需逐例区分真实推理和局部结果提前回答，不把所有错误标为提前 Finish。原轨迹、字段长度及分类保存在 root_cause.json、r0_original_outputs.jsonl 和 anchor_failure_audit.jsonl。

进一步逐条审阅13个合法但失分案例（不覆盖评分）：{'arithmetic_error': 1, 'counting_error': 1, 'intermediate_as_final': 3, 'strict_em_verbose_answer': 2, 'exact_answer_approximation': 1, 'divisibility_reasoning_error': 2, 'strict_em_and_unsupported_inference': 1, 'answer_contradicts_reasoning': 1, 'evidence_comparison_error': 1}。只有3个明确将中间结果作为最终答案；另有真实算术/计数/整除错误，以及 exact-answer/EM 限制。证据和解释在 anchor_manual_review.jsonl。

另发现原始数据一致性问题：gsm8k:952题干为8:00AM到11:00PM，参考推导却按11:00AM计算360；按题干应为1800。原三方法和新版两方法均回答1800且均在主分中失分。data_quality_review.jsonl保留原行核查说明，未修改gold、题目或主分。这是数据噪声，不能算作协议推理失败。

## 实际修改

Specialist 主流程仅生成两个字符串 work/draft，由 Runtime 维护 target/session/stage；Draft 配额按实际文本 token 统计，默认384。完整阶段预算默认3×512，实验单阶段512。要求独立执行开放式局部任务，提供实际计算、证据、结论或有据的部分进展；移除占位示例和固定审查字段。原题、选中原文、Anchor 推理、动态任务及当前会话私有历史分区，原题仅序列化一次。Narrow/Wide 仍按固定规则，不事后选上下文。

Anchor 历史仅保留 ID、step、refs；候选答案另列。强调每步实际推进、回答完整原题、未完成时提出具体局部任务，Controller 决定是否调用，允许无价值任务为 null。保留原状态机、失效候选机制、隔离会话、有限重试及带成本失败。JSON Draft 特征池化只取字符串内容。SFT/DPO 与 Controller checkpoint schema 升为3；历史对象 Draft 仅离线解析。

## R0：交付诊断

离线统计见上。固定12状态旧192结果直接归档复用；旧结构384复用相同历史生成，真实重跑新版 Anchor；新版384使用新版 Specialist 并与 R2 相同记录复用。后两者使用同一下游提示与原始前缀，历史 Specialist 复用调用成本单列。旧192下游是旧版 Anchor，不能直接把差异归因于 Draft 协议；新旧 Specialist 结构/提示均改变，非单因素消融。

|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|archival_old192_old_anchor|3|1.000|—|0|1|2.67|
|gsm8k|old_structure_384_replayed|3|1.000|—|0|2|2.67|
|gsm8k|refined384_new_anchor|3|1.000|—|0|0|1.00|
|hotpotqa|archival_old192_old_anchor|2|0.500|0.700|0|1|1.00|
|hotpotqa|old_structure_384_replayed|2|0.000|0.458|0|0|1.00|
|hotpotqa|refined384_new_anchor|2|0.500|0.625|0|0|1.00|
|math500|archival_old192_old_anchor|7|0.429|—|1|7|2.14|
|math500|old_structure_384_replayed|7|0.571|—|0|2|2.14|
|math500|refined384_new_anchor|7|0.571|—|1|3|2.00|

新版固定12状态已完成 12 条 Help，提交 11 条；长度分布 {'work_tokens': {'min': 93, 'median': 133, 'max': 362}, 'draft_tokens': {'min': 60, 'median': 117, 'max': 185}}。原work均3 tokens且为占位符；新版非占位符不自动等同实质推理，见下方定性审阅。

## R1：Anchor 稳定性

原24题三条件复用；新版两条件实际运行 48/48 次。评分沿用原 rational/math-verify/Hotpot EM/F1，无人工覆盖主分。格式失败为失败调用数，可能包括成功重试。

|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|dari_anchor_only|8|0.875|—|0|0|2.12|
|gsm8k|original_dari_anchor_only|8|0.750|—|0|0|3.38|
|gsm8k|original_standard_cot|8|0.875|—|0|0|0.00|
|gsm8k|original_stepwise_cot|8|0.875|—|0|0|3.38|
|gsm8k|stepwise_cot|8|0.750|—|0|0|1.75|
|hotpotqa|dari_anchor_only|8|0.500|0.647|0|0|1.12|
|hotpotqa|original_dari_anchor_only|8|0.625|0.748|0|0|1.38|
|hotpotqa|original_standard_cot|8|0.500|0.708|0|0|0.00|
|hotpotqa|original_stepwise_cot|8|0.750|0.873|0|1|1.50|
|hotpotqa|stepwise_cot|8|0.750|0.823|0|0|1.25|
|math500|dari_anchor_only|8|0.500|—|3|6|1.25|
|math500|original_dari_anchor_only|8|0.500|—|3|6|1.50|
|math500|original_standard_cot|8|1.000|—|0|0|0.00|
|math500|original_stepwise_cot|8|0.625|—|2|6|1.88|
|math500|stepwise_cot|8|0.625|—|3|8|1.38|

新版全部16个失分运行已逐条核对：{'reference_inconsistency': 2, 'intermediate_as_final': 1, 'no_answer_anchor_format': 6, 'coefficient_reasoning_error': 1, 'verbose_strict_em': 4, 'wrong_answer_type': 1, 'alias_and_verbose_em': 1}。明确中间结果作最终答案1例（gs333只给两项总价、不求差），6例因Anchor格式失败无候选，另有真实系数比较错误、答案类型错误、参考矛盾和严格EM/别名限制。没有把全部失分称为提前Finish；详见r1_failure_review.jsonl。逐调用JSON错误类型及是否触及512输出上限另列protocol_failure_audit.jsonl，不混淆截断和非法转义。

## R2：同前缀协作

原36前缀所有四控制均按新版 Anchor 重新运行，共 144/144 次。先固定12状态诊断，再扩展。Continue/Repair 各一调用；Extended 两次 Anchor；Help 一次 Specialist 加一次整合；之后共同至多8个新有效步骤，候选即终止。共享前缀只计算一次，分支资源统计为新增调用。原始前缀逐次哈希保持不变。

|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|continue|14|0.714|—|1|2|2.07|
|gsm8k|help_submit|14|0.786|—|1|0|1.86|
|gsm8k|self_repair|14|0.714|—|2|5|2.14|
|gsm8k|self_repair_extended|14|0.786|—|2|4|2.86|
|hotpotqa|continue|7|0.429|0.486|2|0|3.29|
|hotpotqa|help_submit|7|0.429|0.546|1|0|2.14|
|hotpotqa|self_repair|7|0.286|0.485|1|0|2.43|
|hotpotqa|self_repair_extended|7|0.429|0.634|0|0|2.29|
|math500|continue|15|0.800|—|0|1|2.00|
|math500|help_submit|15|0.733|—|1|5|1.53|
|math500|self_repair|15|0.733|—|4|10|1.20|
|math500|self_repair_extended|15|0.800|—|3|6|2.73|

R2：完整配对 36 个状态，成功交付 33 个。
- intention_to_help Help vs continue：Win/Tie/Loss=[2, 32, 2]。
- intention_to_help Help vs self_repair：Win/Tie/Loss=[6, 26, 4]。
- intention_to_help Help vs self_repair_extended：Win/Tie/Loss=[4, 27, 5]。
- delivered_conditional Help vs continue：Win/Tie/Loss=[2, 29, 2]。
- delivered_conditional Help vs self_repair：Win/Tie/Loss=[5, 26, 2]。
- delivered_conditional Help vs self_repair_extended：Win/Tie/Loss=[4, 26, 3]。

Local Oracle 成功数（分母 36）：{'A': 27.0, 'B': 29.0, 'C': 29.0, 'D': 30.0, 'B-A': 2.0, 'D-C': 1.0}。这是事后上界，非可部署策略。

Hotpot 条件内 F1 Oracle（n=7）：{'A': 0.5673469387755102, 'B': 0.710204081632653, 'C': 0.710204081632653, 'D': 0.710204081632653, 'B-A': 0.1428571428571428, 'D-C': 0.0}。各控制的Help-F1差异另见summary.json。

Delivered 子集为条件性机制诊断，不能替代总体，也不消除选择偏差。引用保留不等于实际采用，正确答案不等于局部推理完全正确。

## R3：新开发样本

实际 E0 72/72 次，分支 56 次；一步完成者没有 checkpoint，coverage.jsonl 逐题记录，不人为延长推理。未根据新结果调整提示。
为复用空闲设备，R3分为两个执行manifest：r3在GPU0/1/5分片Anchor，未调用的Specialist留CPU，仅跑E0；r3_branches在GPU6/7和2/3分别分片A/S，重用原样保存的前缀跑全部四种E1动作。模型、提示、采样和评分不变；部署差异明确记录，不能直接比较跨阶段墙钟速度。分支目录只链接不可变E0索引和前缀，不重新生成或覆盖它们。

|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|dari_anchor_only|8|0.750|—|0|0|1.88|
|gsm8k|standard_cot|8|0.875|—|0|0|0.00|
|gsm8k|stepwise_cot|8|0.875|—|0|0|1.88|
|hotpotqa|dari_anchor_only|8|0.375|0.573|1|2|1.12|
|hotpotqa|standard_cot|8|0.250|0.538|0|0|0.00|
|hotpotqa|stepwise_cot|8|0.250|0.415|1|2|1.00|
|math500|dari_anchor_only|8|0.625|—|1|0|2.88|
|math500|standard_cot|8|0.750|—|0|0|0.00|
|math500|stepwise_cot|8|0.625|—|2|4|2.75|

|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|
|---|---|---:|---:|---:|---:|---:|---:|
|gsm8k|continue|5|0.800|—|0|0|1.40|
|gsm8k|help_submit|5|1.000|—|0|0|1.40|
|gsm8k|self_repair|5|0.800|—|0|1|2.00|
|gsm8k|self_repair_extended|5|0.600|—|1|0|3.80|
|hotpotqa|continue|2|0.500|0.833|0|0|1.00|
|hotpotqa|help_submit|2|0.500|0.833|0|0|1.00|
|hotpotqa|self_repair|2|0.500|0.562|0|0|1.00|
|hotpotqa|self_repair_extended|2|0.500|0.562|0|1|2.00|
|math500|continue|7|0.571|—|1|0|2.43|
|math500|help_submit|7|0.571|—|1|1|2.29|
|math500|self_repair|7|0.857|—|1|2|1.43|
|math500|self_repair_extended|7|0.857|—|1|3|2.29|

R3：完整配对 14 个状态，成功交付 13 个。
- intention_to_help Help vs continue：Win/Tie/Loss=[1, 13, 0]。
- intention_to_help Help vs self_repair：Win/Tie/Loss=[1, 11, 2]。
- intention_to_help Help vs self_repair_extended：Win/Tie/Loss=[2, 10, 2]。
- delivered_conditional Help vs continue：Win/Tie/Loss=[1, 12, 0]。
- delivered_conditional Help vs self_repair：Win/Tie/Loss=[1, 10, 2]。
- delivered_conditional Help vs self_repair_extended：Win/Tie/Loss=[2, 9, 2]。

Local Oracle 成功数（分母 14）：{'A': 11.0, 'B': 12.0, 'C': 11.0, 'D': 12.0, 'B-A': 1.0, 'D-C': 1.0}。这是事后上界，非可部署策略。

Hotpot 条件内 F1 Oracle（n=2）：{'A': 0.8333333333333333, 'B': 0.8333333333333333, 'C': 0.8333333333333333, 'D': 0.8333333333333333, 'B-A': 0.0, 'D-C': 0.0}。各控制的Help-F1差异另见summary.json。

## 资源与诊断文件

真实增量 token/调用统计：`{"A": {"input": 686575, "output": 127336, "calls": 695, "cached_input": 0}, "S": {"input": 23362, "output": 15099, "calls": 50, "cached_input": 0}}`。运行计时合计 10924.3 秒，含 Runtime/日志与共享服务器波动，不含模型加载，不是已校准 GPU 成本。旧生成复用不冒充新调用；原 token 在 R0 replay 记录单列。
各阶段、数据集、条件、模型的输入/输出tokens、调用数及用时总和另列 resources.csv。

审计 Specialist 调用 50，合法 46，原占位符 0；失败原因 {'Output must be one complete JSON object': 4}。specialist_audit.jsonl 保留 work/draft/长度/错误；request_audit.csv 保留每步请求及推理供具体性、一致性和可执行性人工复核。不将非占位 work 自动等同正确推理，不以正则冒充语义评价。

评分器待复核记录 31 条，单列 evaluator_review.jsonl，包含所有MATH非空但失分答案，未覆盖主分。math-verify的scored标志并不能保证解析语义可靠：例如24313的整句“The simplified expression is … cos(x)/sin(x)”被解析成字母乘积而失分，单独cot(x)则得分；该差异不能解释为真实推理能力差距。

新增有效步骤的请求统计（unfinished仅指模型未给候选，不代表客观未解；无候选且null时Runtime提供fallback）：`{"r1/stepwise_cot": {"valid_steps": 35, "unfinished_steps": 14, "authored_unfinished": null, "fallback_unfinished": null, "exact_repeats": 0}, "r1/dari_anchor_only": {"valid_steps": 36, "unfinished_steps": 15, "authored_unfinished": 9, "fallback_unfinished": 6, "exact_repeats": 0}, "r2/continue": {"valid_steps": 82, "unfinished_steps": 49, "authored_unfinished": 29, "fallback_unfinished": 20, "exact_repeats": 11}, "r2/self_repair": {"valid_steps": 65, "unfinished_steps": 36, "authored_unfinished": 24, "fallback_unfinished": 12, "exact_repeats": 10}, "r2/self_repair_extended": {"valid_steps": 97, "unfinished_steps": 46, "authored_unfinished": 22, "fallback_unfinished": 24, "exact_repeats": 12}, "r2/help_submit": {"valid_steps": 64, "unfinished_steps": 31, "authored_unfinished": 19, "fallback_unfinished": 12, "exact_repeats": 7}, "r3/stepwise_cot": {"valid_steps": 45, "unfinished_steps": 24, "authored_unfinished": null, "fallback_unfinished": null, "exact_repeats": 0}, "r3/dari_anchor_only": {"valid_steps": 47, "unfinished_steps": 25, "authored_unfinished": 15, "fallback_unfinished": 10, "exact_repeats": 5}, "r3_branches/continue": {"valid_steps": 26, "unfinished_steps": 13, "authored_unfinished": 9, "fallback_unfinished": 4, "exact_repeats": 6}, "r3_branches/self_repair": {"valid_steps": 22, "unfinished_steps": 9, "authored_unfinished": 4, "fallback_unfinished": 5, "exact_repeats": 1}, "r3_branches/self_repair_extended": {"valid_steps": 39, "unfinished_steps": 18, "authored_unfinished": 10, "fallback_unfinished": 8, "exact_repeats": 6}, "r3_branches/help_submit": {"valid_steps": 25, "unfinished_steps": 12, "authored_unfinished": 9, "fallback_unfinished": 3, "exact_repeats": 7}}`。E1不重复计算共享前缀；原R2请求身份保持原始10自主/26fallback，不将新版后续请求当作原前缀改善。
同24题原始请求统计：`{"original_r1/stepwise_cot": {"valid_steps": 54, "unfinished_steps": 32, "authored_unfinished": null, "fallback_unfinished": null, "exact_repeats": 0}, "original_r1/dari_anchor_only": {"valid_steps": 50, "unfinished_steps": 29, "authored_unfinished": 5, "fallback_unfinished": 24, "exact_repeats": 1}}`。Stepwise关闭请求生成，相关比例不适用；这些统计也不代表每一步都有可执行的Help动作。
R1新版9个自主请求逐条定性审阅（r1_request_quality_review.jsonl）：9/9具有具体对象、与当前步骤一致且可从已有输入执行；其中EF长度已在原图标7，属于具体但低边际价值的任务。旧10自主请求也均具体，但买房原价请求重复题干已给值，出生年请求在现有原文中无精确数据（original_request_review.jsonl）。具体性不等同有价值，更不等同值得调用。

## 局部求解内容的定性审阅

已逐条审阅首批 9 条真实交付，work类别为 {'substantive': 3, 'mostly_advisory': 4, 'evidence_composition': 1, 'partial_constraints_not_requested_area': 1}，详细说明在 specialist_qualitative_review.jsonl。这是有限、可复核的轨迹审阅，非独立语义评分、无额外Judge、未覆盖全部状态。部分work仍只建议Anchor计算，而draft才实际给出算式与结果；例如烘焙长度draft算出3360英寸/12=280英尺，work并未执行这些运算。因此局部求解与交付已有真实实例，但Private Work字段的功能适应并未完全解决，不能只按占位符消失判定成功。

## 局限与维护

精简提示、历史和 Draft 一并改变；无法将全部差异归于单一因素。小样本及旧前缀分布限制推广；旧开发样本经多轮诊断，R3 亦非最终测试集。解析失败需复核，主分不擅自修正。尚未校准成本，不构造正式效用训练标签。不从少量成功/负例宣布协作整体有效或失效。

代码复用现有 Runtime、Backend、数据加载和评分、反事实 runner；新增 refinement 编排/根因审计/汇总，更新协议与训练数据版本、默认配置和回归；README/METHOD/MODEL_INTERFACE 同步。旧报告/原始轨迹/有效权重保留；输出受 .gitignore 保护，不自动 push、不修改原权重、不停止他人任务。测试及 Git 检查见 logs。

## 代表性真实案例

- 成功局部求解及整合：R3 gsm8k:1165，Specialist算出640户、总1040户、3120次、收入1248；Anchor后续用相同计算完成1248。三种内部动作都误答374.40，因此这是本次新样本中额外成功状态。未引用h1，不能仅以引用判断未采用；单次轨迹不能证明因果泛化。
- 正确内容但整合失败：R2 prealgebra/1423，Draft给出18π×2/3=12π，Anchor两次因非法JSON转义失败，无答案；三个内部分支均正确。这是有效内容遭遇协议整合失败。
- 有害辅助：R2 hotpotqa:5a7d19d85542995ed0d165e8，Specialist把会议总部城市问题改解为球队主场城市，交付Knoxville，Anchor采用；其他三个动作答对Birmingham。属于问题理解偏移造成的负面影响。
- 局部任务未完成、无收益：R2 counting_and_probability/188，Draft列出三个正确三角不等式，但没有完成所请求的区域面积。Anchor随后错误算5/6，四动作主分均失败，不能仅以交付率认定辅助有效。
- 错误辅助后自行修正：R2 gsm8k:1241，Draft把1/3过早取0.33，利润错成7.80；Anchor自行计算精确收入20，利润8，三个内部动作因缺字段终止。这贡献Oracle新增成功，但不是正确局部知识带来的可靠增益。
- 有帮助但内部扩展也覆盖：R2 hotpotqa:5abd72d155429933744ab7e0，Draft比较两位导演共同媒体为film，Anchor得到film；Extended亦成功。这增加B相对A的覆盖，却不增加D相对C覆盖。

完整事件均在各阶段e1/traces.jsonl，prefix_id对应保存前缀；定性说明另存representative_cases.jsonl。不要求每种现象都出现；尚无足够证据确认“正确但完全未使用”的纯内容案例。

## 判断与下一步

交付瓶颈显著缓解：原14/36变为33/36，新样本13/14；剩余4次Specialist失败均为完整JSON未成功解析，包含输出上限截断，非384文本配额拒绝。work占位消失，但仍有建议式work和未完成局部任务。R1原Standard/Stepwise/DARI为19/18/15，新版Stepwise/DARI为17/15，未见明确Anchor准确率改善；R3三条件为15/14/14。

R2四动作正确数为25/23/26/25；R3为9/11/10/10。Help有少量挽救及真实负例，整体未全面超过内部计算。R2 Oracle B-A=2、D-C=1，R3均为1；至少一个旧状态的额外成功受格式路径变化和错误Draft自行修正影响，不能把这些上界直接解释为可靠协作效用。

下一步优先做有限的协议适应诊断：用未参与本轮评估的少量示例研究JSON合法性、真实work执行及精确数值交付，可另行设计小规模Protocol SFT；这只是建议，本轮没有训练。之后再预注册Dynamic Assistance Comparison（区分null fallback、具体低价值请求与真正未解任务），确认帮助内容质量与整合稳定性。Adaptive Depth、Selective Communication和Controller/Receiver-Utility训练应在此后，并先校准资源成本、隔离训练/开发/独立测试样本；当前不构造大规模DPO标签。

## 最终验收记录

实际结果数48个R1、144个R2、72个R3 E0、56个R3分支；R0复放12条。integrity_audit.json核对唯一条件、完整配对、旧/新前缀哈希及新样本不重叠；全套75项回归通过，git diff检查通过，日志见logs。临时准备/文档脚本及重复临时测试日志已删除，正式旧产物保持不变。设备并发加载曾产生一次OOM，发生在R3推理前且无分支结果，R2退出后重启成功；首次stderr被重试日志覆盖，logs/startup_failure.json明确标为从工具输出重建的故障记录，不冒充原始stderr。没有停止他人任务。研究产物保持本地、不推送。
