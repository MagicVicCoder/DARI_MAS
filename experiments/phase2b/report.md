# DARI-MAS Phase 2B：Protocol Adaptation and Specialist Learning

状态：本轮已完成。

## 工程与评价

采用最小确定性语法修复而非约束解码：仅完整JSON围栏移除和非法反斜杠转义保存；不补字段、答案或截断推理。原始模型输出、token和hidden-state对齐均不变，修复只在Runtime解析入口应用。raw模式默认保留，phase2b统一开启。136个历史非法调用中56个可恢复schema合法，详见format_audit.jsonl；这不是重新生成的能力提升。
数学评价器只从最终答案的明确单一数学表达式提取，不从多段计算选择数字。历史716条结果统一离线重评，4条分数变化，旧产物未覆盖。HotpotQA维持严格EM/F1，不凭gold进行别名匹配；参考矛盾保留benchmark分数，沿用历史数据质量说明。评价器身份phase2b-v1-single-final-math-span在训练前冻结。

## 数据与训练

200条训练、40条验证均为明确标记的构造局部任务，参考推导可检查；没有把未经验证的模型输出当标签，没有虚构人工审核。类别覆盖计算、精确有理数、方程、面积、约束/因式分解、两跳证据、矛盾更新、概率、不充分证据、替代代数求解。训练和验证共享10构造家族，推理措辞存在重复，不能宣称广泛开放式任务泛化。未采集多阶段Expand监督，此能力仅保留回归测试。与正式Runtime相同原题/证据/Anchor上下文/请求/私有历史prompt，单阶段512预算。
旧84题、训练、验证、新36道下游题规范化问题哈希零重叠，见leakage_audit.json。没有以数值替换同家族构造题的做法冒充跨模板独立测试；未声称能自动检出所有语义改写重叠。

原始Qwen3-8B独立Specialist，BF16 LoRA rank16/alpha32/dropout0.05，q/k/v/o projections，AdamW2e-5，batch1累积8，2轮50步，seed20261126。Anchor冻结原始模型，不加载adapter；Controller固定策略，无DPO或正式效用训练。训练response-only mask排除prompt/padding，不静默截断，max_length2048。trainable=15335424；冻结参数版本检查=True。验证loss=[{'step': 0, 'loss': 1.3763937488198281}, {'step': 25, 'loss': 0.7387088492512703}, {'step': 50, 'loss': 0.50219048820436}]；训练时间369.5s；分片峰值内存{'0': 8742375424, '1': 8575937536}bytes。训练loss下降不视为协作能力改善。

## T0 局部任务

{
  "base": {
    "n": 40,
    "format_valid": 40,
    "delivered": 40,
    "draft_expected_result_match": 32,
    "work_expected_result_match": 14,
    "both_result_match": 14,
    "mean_work_tokens": 116.175,
    "mean_draft_tokens": 94.225,
    "total_output_tokens": 8941,
    "total_input_tokens": 12431,
    "calls": 40,
    "generation_latency_ms_sum": 513189.5235804841
  },
  "sft": {
    "n": 40,
    "format_valid": 38,
    "delivered": 38,
    "draft_expected_result_match": 20,
    "work_expected_result_match": 21,
    "both_result_match": 18,
    "mean_work_tokens": 41.975,
    "mean_draft_tokens": 38.3,
    "total_output_tokens": 4620,
    "total_input_tokens": 12431,
    "calls": 40,
    "generation_latency_ms_sum": 370183.73822118156
  }
}

自动指标是受限expected-result出现检查，不等同完整语义正确率、任务完成率或Evidence Consistency。可能在过程里提及正确数值但结论错误，也可能输出等价表达而未命中。work/draft双命中仅为结果一致性代理，不能冒充Advisory-only的反向指标。复杂证据行为需检查真实case；不得从模板匹配得分推导一般能力提升。Base/SFT完全相同40个输入、seed、温度0.7/top_p0.95、512token/384draft、语法修复方式。
预先固定前12条的直接轨迹审阅：{'n': 12, 'base_work_executes': 8, 'sft_work_executes': 7, 'base_draft_correct': 10, 'sft_draft_correct': 5}。详见t0/qualitative_review.jsonl，非盲法独立人工标注。SFT在面积题中将实际计算移入work，但多个证据/不充分信息/概率样本照抄输入“局部问题尚未解决”，不能视作可靠不确定性报告；还有明确算术错误。模板适应造成退化可能与狭窄合成监督及复制Anchor输入有关，当前仅为有证据的行为观察，未做因果消融。不追加数据或重训来追逐当前验证分数。

## T1 同前缀下游

36新题先运行冻结DARI Anchor；首个无答案且五比较可行的状态入候选，固定seed打乱后至多30个。selection.json在分支前固定，前12为pilot，后续复用相同配置。所有五条件同前缀、任务、宽度、预算及后续finish-else-continue策略。单阶段Help，Base通过关闭LoRA而SFT启用同一已冻结adapter，A是完全独立原始模型。每分支增量统计包含失败；T2额外调用不算T1资源。
实际36题仅产生18个合法checkpoint，低于建议24–30，本轮不为凑数新增题或人为延长推理。T2因此最多18个合法SFT Draft，实际数按全部尝试报告；该覆盖限制降低结论可信度。

{
  "continue": {
    "n": 18,
    "reward_sum": 10.0,
    "submitted": 0,
    "no_answer": 4,
    "invalid_calls": 5,
    "generation_latency_ms_sum": 531159.3236044282,
    "hotpot_mean_f1": 0.5,
    "tokens": {
      "A": {
        "input": 53957,
        "output": 9501,
        "calls": 46
      },
      "S": {
        "input": 0,
        "output": 0,
        "calls": 0
      }
    }
  },
  "self_repair": {
    "n": 18,
    "reward_sum": 10.0,
    "submitted": 0,
    "no_answer": 3,
    "invalid_calls": 7,
    "generation_latency_ms_sum": 464190.65227743704,
    "hotpot_mean_f1": 0.5,
    "tokens": {
      "A": {
        "input": 39386,
        "output": 8121,
        "calls": 41
      },
      "S": {
        "input": 0,
        "output": 0,
        "calls": 0
      }
    }
  },
  "self_repair_extended": {
    "n": 18,
    "reward_sum": 10.0,
    "submitted": 0,
    "no_answer": 2,
    "invalid_calls": 2,
    "generation_latency_ms_sum": 666197.4324475741,
    "hotpot_mean_f1": 0.5,
    "tokens": {
      "A": {
        "input": 65522,
        "output": 11521,
        "calls": 55
      },
      "S": {
        "input": 0,
        "output": 0,
        "calls": 0
      }
    }
  },
  "help_base": {
    "n": 18,
    "reward_sum": 13.0,
    "submitted": 15,
    "no_answer": 2,
    "invalid_calls": 5,
    "generation_latency_ms_sum": 739076.88474725,
    "hotpot_mean_f1": 0.5,
    "tokens": {
      "A": {
        "input": 38267,
        "output": 7141,
        "calls": 30
      },
      "S": {
        "input": 8727,
        "output": 5623,
        "calls": 18
      }
    }
  },
  "help_sft": {
    "n": 18,
    "reward_sum": 13.0,
    "submitted": 17,
    "no_answer": 3,
    "invalid_calls": 3,
    "generation_latency_ms_sum": 823838.4279278107,
    "hotpot_mean_f1": 0.5,
    "tokens": {
      "A": {
        "input": 49966,
        "output": 8391,
        "calls": 40
      },
      "S": {
        "input": 8727,
        "output": 4400,
        "calls": 18
      }
    }
  }
}

分数据集统计：{"gsm8k": {"continue": {"n": 8, "reward_sum": 5.0, "mean_f1": null}, "self_repair": {"n": 8, "reward_sum": 4.0, "mean_f1": null}, "self_repair_extended": {"n": 8, "reward_sum": 5.0, "mean_f1": null}, "help_base": {"n": 8, "reward_sum": 7.0, "mean_f1": null}, "help_sft": {"n": 8, "reward_sum": 7.0, "mean_f1": null}}, "math500": {"continue": {"n": 8, "reward_sum": 4.0, "mean_f1": null}, "self_repair": {"n": 8, "reward_sum": 5.0, "mean_f1": null}, "self_repair_extended": {"n": 8, "reward_sum": 4.0, "mean_f1": null}, "help_base": {"n": 8, "reward_sum": 5.0, "mean_f1": null}, "help_sft": {"n": 8, "reward_sum": 5.0, "mean_f1": null}}, "hotpotqa": {"continue": {"n": 2, "reward_sum": 1.0, "mean_f1": 0.5}, "self_repair": {"n": 2, "reward_sum": 1.0, "mean_f1": 0.5}, "self_repair_extended": {"n": 2, "reward_sum": 1.0, "mean_f1": 0.5}, "help_base": {"n": 2, "reward_sum": 1.0, "mean_f1": 0.5}, "help_sft": {"n": 2, "reward_sum": 1.0, "mean_f1": 0.5}}}。

Help-SFT相对控制的Win/Tie/Loss：{'help_base': {'n': 18, 'win_tie_loss': [0, 18, 0]}, 'continue': {'n': 18, 'win_tie_loss': [3, 15, 0]}, 'self_repair_extended': {'n': 18, 'win_tie_loss': [3, 15, 0]}}。SFT实际提交条件子集：{'help_base': {'n': 17, 'win_tie_loss': [0, 17, 0]}, 'continue': {'n': 17, 'win_tie_loss': [3, 14, 0]}, 'self_repair_extended': {'n': 17, 'win_tie_loss': [3, 14, 0]}}，有选择偏差，不能替代全尝试总体。
时间采用各T1 trace实际模型调用latency之和，不含额外T2 Abort生成；原results的Help-SFT外层latency包含紧随的T2计算，不能用该字段直接比较五动作墙钟时间。T1 e0仅用于固定前缀生成，其留存旧评分不进入本节五动作结论；本节均用冻结corrected evaluator。
预选前4状态的局部执行质量审阅（8条输出）：{'help_base': {'n': 4, 'categories': {'complete_correct': 2, 'truncated_invalid': 2}}, 'help_sft': {'n': 4, 'categories': {'complete_incorrect': 1, 'complete_correct': 1, 'truncated_invalid': 1, 'advisory_partial': 1}}}，详见t1/local_quality_review.jsonl。未对全部开放式任务给出伪造的自动Local Task Accuracy或语义完成率。

## T2 选择性通信

{
  "n": 17,
  "beneficial": 3,
  "neutral": 14,
  "harmful": 0,
  "future_tokens": {
    "submit": {
      "A": {
        "input": 31200,
        "output": 4887,
        "calls": 31
      },
      "S": {
        "input": 0,
        "output": 0,
        "calls": 0
      }
    },
    "abort": {
      "A": {
        "input": 35191,
        "output": 5997,
        "calls": 37
      },
      "S": {
        "input": 0,
        "output": 0,
        "calls": 0
      }
    }
  }
}

按预定状态顺序取前25个合法SFT Draft，不按奖励筛选。两分支使用同一完整Specialist-side状态，Submit和Abort前成本相同；Abort清除help并不传Draft/私有work，之后均无新Help，公共续写机会一致。已执行Submit直接复用T1结果，Abort真实生成；future资源仅统计决策后的新增调用。记录Delta_R不创建DPO标签。观察有害信息不代表当前Controller能够识别它。

## Local Oracle

{'A': 11.0, 'B': 12.0, 'C': 14.0, 'D': 14.0, 'n': 18, 'C-B': 2.0, 'D-B': 2.0, 'D-C': 0.0}。A=Continue+Repair；B增加Extended；C增加Base；D为B增加SFT。均为有限rollout事后终局上界，非成本感知部署策略。

## 限制与维护

cost_complete=false；输入/输出token不能等同GPU成本，墙钟延迟受分片与共享设备影响。训练分片峰值显存有真实记录，但推理进程未导出peak allocator统计，不能从共享服务器nvidia-smi快照伪造推理峰值；这是本轮资源记录不足。未开展重复rollout，不能判断单状态差异稳定性。训练改善可能主要来自格式和构造家族拟合；不能凭单一case宣称有效，也不从负例否定机制。小规模SFT优先验证适应方向，进入Controller/Receiver-Utility前须验证一般局部任务质量、独立数据和资源校准。本轮无新增Agent/Verifier/RAG、无原权重修改、无push。报告/adapter/原始轨迹和失败证据均保留在outputs/phase2b，.gitignore保护。README/METHOD/MODEL_INTERFACE同步，回归日志见logs。

## 真实案例与解释

- 局部执行改善：T0 constructed:53/173/33，Base work只描述面积计算，SFT在work实际算出6496/11541/2953；两者draft均正确。这是字段行为适应，不能视作新题型泛化。
- 退化：T0 constructed:166/46/158/208/177，Base能按证据更新城市、指出缺失价格或算出概率，SFT却复制“局部问题尚未解决”。短且合法不等于可靠部分进展。
- 错误辅助后自行纠正：T1 gsm8k:992，SFT把教会支票错误算为24个月而给76；Anchor忽略或纠正为52。三个内部动作计算包含52但多字段包装失分。此状态计入Help成功和T2正差，但不证明SFT信息正确或更有用。
- 有效局部计算：T1 gsm8k:471，两种Specialist都给出300→240→80，Anchor答80；Extended也正确。T2 Submit正确、Abort无答案，因此交付有正差；不能把它归为SFT独有改善。
- 未证明的精确化：T1 precalculus/697，SFT work仅建议数值计算，draft给近似-1.732050817552918并声称“Using a calculator”。实际没有调用计算器或其他工具，该语句只是模型文本。Anchor据近似猜成-sqrt(3)，得分优于内部/Abort的短近似，但数值接近不是符号等价证明；此成功限制在冻结评分标准和单次轨迹。
- 无收益：Base/SFT全部18状态终局主分相同，许多正确状态无需额外SFT能力；T2中14/17中性。没有观察到T2负Delta，不虚构有害交付案例。局部错误及T0退化已真实发生，但当前Controller未学习识别这些内容。

案例完整原文在t0、t1、t2 traces及representative_cases.jsonl；分析不覆盖主分。两项Oracle相对内部B都增加2个成功，D-C=0，未显示SFT增加Base无法覆盖的状态。

## 结论与下一步

本组训练稳定、adapter实际加载可用，但T0 Draft目标匹配32/40→20/40、交付40/40→38/40；SFT work匹配14/40→21/40，不能抵消局部结果退化。平均长度下降与work中计算增加仅说明行为改变。T1 Help-Base/Help-SFT都13/18，SFT对Base0/18/0，未观察训练提升向接收者传递。T2 3有益/14中性/0有害，仅说明这些固定rollout下交付存在差异，不证明选择性Controller可学成或当前数据足以训练。

优先重新设计更接近真实动态请求且推理措辞多样的独立监督来源，加入实际Anchor上下文干扰、可靠的精确运算和证据不足示例，检查复制输入和work/draft重复。需要另行预注册小规模数据对比，不能把本轮验证/下游轨迹直接加入训练再声称独立验证。本轮不追加训练或学习率搜索。现有结果尚不足以宣布具备正式Controller/Receiver-Driven训练基础；应先验证一般局部求解质量和接收者真实内容收益，再考虑成本校准及下一阶段学习。

## 完整性与验收

80条T0输出、36个新Anchor任务、18前缀×5=90分支、17对合法SFT Draft的T2均完成。未达到建议24–30前缀/20–30Draft，原因是覆盖率，不以事后加样补齐。integrity_audit.json验证无重复条件、全部前缀哈希、完整私有状态、相同Submit/Abort前成本及无后续Specialist调用。全套81项回归通过，git diff --check通过。训练权重未推送，历史Phase1/2A/Refinement保留；有效adapter和正式原始输出受.gitignore保护，临时可恢复脚本清理。训练/数据审计入口移入dari模块，原执行脚本保存在阶段输出以追溯；测试及Git检查见logs。
