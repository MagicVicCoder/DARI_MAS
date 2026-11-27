# DARI-MAS V1 第一阶段报告

日期：2026-11-25（Asia/Shanghai）。本阶段完成原仓库基础上的核心改造及小规模
工程验证，不代表正式 benchmark 或方法有效性验证，不能标记所有研究验收项通过。

## 部署、版本和环境

项目：`/data/users/csh/MAS_v1`。原始目标目录为空，完整 clone 保留 Git 历史。
原始 SHA / 当前 HEAD：`54786cebb133b49cab446ae8cae1cc07fc05a476`。
本地分支：`dari-v1-dynamic-assistance`；修改保留为可审查的未提交工作区，无 push。
工具实际运行在 Linux 6.8 / Ubuntu glibc 2.35，通过客户端路径映射访问 Windows 风格路径。

项目内 `.venv` 使用 Python 3.10.19，复用 qwen3env 的 PyTorch 2.10.0+cu128、
Transformers 4.57.6，并单独安装 PEFT 0.21.3。未修改共享 Conda 环境或原模型权重。
初始 8 张 GPU 均有其他任务，选 GPU 4/7 独立加载 A/S；没有终止其他研究任务。
检查记录：`outputs/baseline/environment.txt`、`gpu.txt`、`dependencies.log`。

本地 Qwen3-8B：hidden_size=4096，bfloat16，实际 tokenizer/chat template，
enable_thinking=false。完整配置/模板在 `outputs/qwen3/validation/model_manifest.json`。
模型目录内容指纹：`39d9d997762714f7896f5c9c2a4f8022927af78987925c43351da4a06b21de03`，
文件清单在 `outputs/baseline/model_identity.json`。早期轨迹使用描述性部署版本名；
最终配置改为该真实指纹，模型文件未变化。

## 原实现审计和修改

阅读 README、schema/runtime/context/controller/backends/hf_backend/configuration、
collection/receiver_collection、workflows/alternate、训练损失及测试。原代码提供
合理的异构成本、反事实分支和交替训练接口，但主方法依赖固定 check/recompute、
固定两阶段、无动态任务语义表示；非法 Expand 会丢失前一合法 Draft，引用静默过滤，
候选答案不能撤销。此次在原模块中修改这些根因，保留 dual-value/Huber/DPO 研究定义。

已实现动态 HelpRequest、current_step 解析、严格引用验证、追加全局历史、显式撤销答案、
标记 fallback、同任务 SelfRepair/Help、配置化多阶段、合法私有历史保留、无效 Expand
保留最后合法 Draft、会话隔离和 source ID 可追踪上下文。Submit 只交付 Draft，Abort
清理临时特征且不退成本。保留 change 字段作为通用建议，而非固定错误修复类型。

Controller 使用真实请求 instruction 的输出 token 区间均值，替代固定 operation
embedding；加入 goal/request/candidate 和可配置局部阶段特征。fallback 明确使用目标
状态 proxy 和标志位，不声称编码了不存在的请求输出。HF 自动读取模型维度和上下文上限，
独立 tokenizer/template/device/adapter 配置，拒绝 text-only 伪隐状态。旧 Controller
checkpoint 和分支数据 v1 明确拒绝。训练 prompt-response 容器因结构未变仍为 v1。

反事实标签增加请求、种子和模型/策略快照；Specialist 采集不再固定 narrow-short-check。
保留实际下游效用生成偏好、辅助 SFT、冻结 reference 和最终 theta refresh。正式数据
覆盖率仍需随采集检查，不能声称少量 smoke 覆盖全部任务/上下文组合。

## 测试证据

原始测试：52 项，0 失败，2 项因缺少 PEFT 跳过，9.382 秒。
最终测试：63 项全部通过，无跳过；包括实际 Transformers/PEFT 小型随机权重集成，
SFT/DPO、reference 隔离、checkpoint 重载、同前缀分支和一次合成交替训练。
日志：`outputs/baseline/unittest.log`、`regression.log`、`dynamic.log`。
新增测试覆盖同任务 SelfRepair/Help、三阶段/多会话、fallback、非法引用、候选撤销及
陈旧 candidate 特征清理、无效 Expand 保留、内部元数据不泄漏、请求 token 池化精确对齐。

## 真实模型结果

完整 12 个功能样本：`outputs/qwen3/validation/traces.jsonl`。
10 个 Finish，1 个 step_limit，1 个 anchor_format；45 次合法、6 次非法输出。
错误为答案数字/布尔而非字符串（3 次）、Draft 超限（2 次）、Anchor 字段错误（1 次）。
不是准确率报告。`evidence-2` 的答案错误地从“注册是出席必要条件”推出出席，说明
真实协议闭环不等于推理正确，辅助也没有自动保证质量。

完整四类案例可按 task_id 检查：

- Anchor 独立：constraints-1，多步后 Finish 返回 15。
- SelfRepair：math-2，自行执行局部任务后返回 3；alternative-1 自行代入检验。
- 多阶段 Submit：evidence-2，三次 Specialist 阶段后提交，保留实际错误答案用于分析。
- Abort：multihop-1，辅助结束中止，Anchor 独立返回 Porto；私有内容不进入后续输入。
- 多会话：evidence-1 和 multihop-2，各两个 Help Session，每会话三阶段，输入可检查
  新会话 previous_private_outputs=[]，Expand 包含此前有效 work/draft。

真实动态请求包括 Calculate 17*20 and 17*3 separately；其反复提出也揭示任务重复和
推进不足。math-1 初轮耗尽步数没有答案，不能视作独立推理成功。

发现并修复历史中 Runtime fallback/reason 字段被模型照抄的问题；失败原始轨迹保存
在 `outputs/qwen3/pre_prompt_fix/`，非无用重复结果。强化答案必须字符串及目标 ID
必须精确复制后，`outputs/qwen3/final_checks/` 的四个案例全部 Finish，14 次合法输出，
0 非法；包含原先格式失败的 logic-2。两批提示词不同，不能合并成一个通过率。

额外 `outputs/qwen3/phase_checks/` 验证 Anchor → Specialist 阶段1/2/3 → Anchor
的真实隐状态合法动作评分，均为4096维；清理后不再出现 helper_* 特征。评分来自随机
网络，仅 Forward smoke，真正执行动作为强制 debug policy，不是训练后决策质量。

完整12样本资源：Anchor 32 次调用，input=10382/output=4061；Specialist 19 次，
input=7692/output=4290。跨调用无 KV 复用，每次 prefill 完整计入。总推理延迟
503789.09ms（共享服务器负载影响）；GPU 4/7 进程峰值 allocated 分别为
16804693504 / 16806577664 bytes，不是整卡峰值也不是校准资源成本。
资源汇总：`outputs/qwen3/validation/resources.json`。全部 real_debug，cost_complete=false。

## 训练 smoke

Controller：`outputs/controller_smoke.json` 与 `outputs/synthetic/controller.pt`，
三次优化，真实神经网络/合成成本标签。记录每步 loss/梯度/更新范数/LR；非零更新，
不要求少步 loss 单调。此处不是用未校准 Qwen 成本训练效用模型。
该次 Controller 初始loss=0.14614829、最终全数据loss=0.17690048，每步loss为
0.14614829/0.08834178/0.10588946；梯度范数为2.53933573/0.00225129/0.00010708，
更新范数3.54244649/2.37219167/1.83240977，LR=0.003。

真实 Qwen3 初次 SFT：loss 0.18547124，梯度范数0.15408954，LoRA 更新范数
0.01714980；DPO+辅助SFT：loss 0.71694893，梯度范数4.38299990，更新范数
0.02745417。均一步，1916928 个可训练 LoRA 参数，LR=2e-5。
初次记录：`outputs/qwen3/validation/training_summary.json`；Adapter 在该目录。
DPO pair 明确是 synthetic gradient-smoke 标签，不是实际 receiver utility，不作效果结论。
一步 DPO margin 未改善，仍如实保留。

增加 frozen tensor-version 检查后再次 SFT，确认冻结参数未变；详情在
`outputs/qwen3/training_checks/sft_adapter/training_manifest.json`。
后续复测遇到共享 GPU 显存不足，失败日志保留在 `training_checks_oom*.log` 等文件，
原先成功的 DPO 结果仍有效，但不能把失败复测写成成功。最终隔离可见GPU后重试成功，
记录在 `outputs/qwen3/training_checks/summary.json`：DPO loss=0.71699691、梯度范数
4.38581848、更新范数0.02745265，冻结 base/reference 参数版本未变化。reference/init
adapter 内容指纹一致并已记录。重新加载 DPO adapter 后真实生成10 tokens，隐状态
4096维，Adapter 内容指纹已保存。该 smoke 的 DPO margin 仍下降，不能视作偏好优化
效果通过；仅 loss/backward/update/reference isolation/save/load 接口通过。
OOM 属于资源/上下文约束，未证明由某个其他任务造成；隔离 CUDA_VISIBLE_DEVICES=4 后通过。

## 验收范围和未完成项

核心架构、动态任务、多阶段历史、隔离、Continue/SelfRepair/Help/Expand/Submit/Abort/
Finish、真实 Qwen 输入输出、实际隐藏表示与合法评分、合成反事实/训练接口及回归已验证。
不同 tokenizer/hidden size 通过小模型接口测试；两个不同大型 backbone 尚未实跑。
真实 Qwen 同前缀反事实集合和基于校准真实效用的正式 DPO/交替训练未做，成本未校准，
不能宣称全部正式验收通过。原模型非训练且未修改；新增权重仅在项目 outputs 内。
原生 thinking=true 处理已实现但尚未真实验证；跨调用 KV、质量型 Draft Selection 未实现。

局限：部分请求重复/低价值，候选答案不保证正确；Wide 选择是结构性近邻而非语义检索。
输入限制拒绝/屏蔽，非静默截断。预计预算不是严格实测 GPU 硬上界。显式引用率不等于
实际信息利用率。没有比较方法性能、SOTA、收敛或协作收益结论。

## 文件、清理及复现

修改原 schema/runtime/context/controller/HF接口、collection/workflows、训练统计与
README/defaults/examples；新增 policies、validate、training_smoke、动态回归、Qwen
配置、12样本、AGENTS.md、METHOD/MODEL_INTERFACE。移除固定 operation 主流程、
废弃 short_output/expand_output 参数及旧通用 REPAIR 指令；保留空 template 序列化槽。
临时 .work 编辑脚本和中间 JSON 已逐项删除；未删除原始数据、有效权重或失败证据。
.gitignore 已保护 .venv、outputs、模型和权重；最终检查 Git diff/status 和 diff --check。

复现命令在 README。复测训练可用 `CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m
dari.training_smoke --stage dpo --device cuda:0`；先检查该物理 GPU 可用显存。
下一阶段先稳定动态协议/推进和 debug 训练资源隔离，再独立 profiling/calibration，
收集真实同前缀/同输入效用，最后小规模冻结策略比较；本阶段没有开展大规模训练。
