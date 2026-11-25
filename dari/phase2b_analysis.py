"""Phase 2B descriptive summaries with explicit incomplete/semantic limits."""
from collections import Counter, defaultdict
import json

from .phase2b_data import ROOT
from .pilot import read_rows


def paired(rows, left, right):
    groups=defaultdict(dict)
    for row in rows:groups[row['prefix_id']][row['method']]=row
    differences=[v[left]['reward']-v[right]['reward'] for v in groups.values() if left in v and right in v]
    return {'n':len(differences),'win_tie_loss':[sum(d>0 for d in differences),sum(d==0 for d in differences),sum(d<0 for d in differences)]}


def analyze():
    t0={method:read_rows(ROOT/'t0'/f'{method}.jsonl') for method in ['base','sft']}
    from .phase2b import local_assessment
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/models/Qwen3-8B',local_files_only=True)
    class TokenCounter:
        def count_text_tokens(self, model_id, text):
            return len(tokenizer.encode(text,add_special_tokens=False))
    validation={r['task_id']:r for r in read_rows(ROOT/'validation.jsonl')}
    rescored=[]
    for method,records in t0.items():
        for row in records:
            assessment=local_assessment(validation[row['task_id']],row['output'],TokenCounter())
            row.update(assessment)
            rescored.append({'method':method,'task_id':row['task_id'],**assessment})
    (ROOT/'t0/assessment_corrected.jsonl').parent.mkdir(exist_ok=True)
    (ROOT/'t0/assessment_corrected.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rescored))
    rows=read_rows(ROOT/'t1/e1/results.jsonl'); t2=read_rows(ROOT/'t2/results.jsonl')
    traces=read_rows(ROOT/'t1/e1/traces.jsonl')
    for row in rows:
        matching=next((t for t in traces if t['prefix_id']==row['prefix_id'] and t['method']==row['method']),None)
        row['generation_latency_ms']=sum(e.get('latency_ms') or 0 for e in matching['events']) if matching else 0
    grouped=defaultdict(list)
    for row in rows:grouped[row['method']].append(row)
    result={'t0':{},'t1':{},'t2':{},'oracle':{},'cost_complete':False}
    for method,records in t0.items():
        result['t0'][method]={'n':len(records),'format_valid':sum(r['format_valid'] for r in records),
            'delivered':sum(r['delivered'] for r in records),'draft_expected_result_match':sum(r.get('local_result_match',False) for r in records),
            'work_expected_result_match':sum(r.get('work_result_match',False) for r in records),
            'both_result_match':sum(r.get('work_draft_result_match',False) for r in records),
            'mean_work_tokens':sum(r.get('work_tokens',0) for r in records)/max(1,len(records)),
            'mean_draft_tokens':sum(r.get('draft_tokens',0) for r in records)/max(1,len(records)),
            'total_output_tokens':sum(r['usage']['output_tokens'] for r in records),
            'total_input_tokens':sum(r['usage']['input_tokens'] for r in records),
            'calls':len(records),'generation_latency_ms_sum':sum(r['latency_ms'] for r in records)}
    for method,records in grouped.items():
        result['t1'][method]={'n':len(records),'reward_sum':sum(r['reward'] for r in records),'submitted':sum(r.get('submitted',False) for r in records),
            'no_answer':sum(r['no_answer'] for r in records),'invalid_calls':sum(r['format_failures'] for r in records),
            'generation_latency_ms_sum':sum(r['generation_latency_ms'] for r in records),
            'hotpot_mean_f1':sum(r['f1'] for r in records if r['dataset']=='hotpotqa')/max(1,sum(r['dataset']=='hotpotqa' for r in records)),
            'tokens':{model:{metric:sum(r.get('tokens_by_model',{}).get(model,{}).get(metric,0) for r in records) for metric in ['input','output','calls']} for model in ['A','S']}}
    result['t1_pairs']={control:paired(rows,'help_sft',control) for control in ['help_base','continue','self_repair_extended']}
    result['t1_by_dataset']={dataset:{method:{'n':len(group:=[r for r in records if r['dataset']==dataset]),
        'reward_sum':sum(r['reward'] for r in group),'mean_f1':sum(r['f1'] for r in group)/len(group) if dataset=='hotpotqa' and group else None}
        for method,records in grouped.items()} for dataset in ['gsm8k','math500','hotpotqa']}
    delivered=[r for r in rows if r['method']=='help_sft' and r.get('submitted')]
    delivered_ids={r['prefix_id'] for r in delivered}
    result['t1_delivered_conditional']={control:paired([r for r in rows if r['prefix_id'] in delivered_ids],'help_sft',control) for control in ['help_base','continue','self_repair_extended']}
    sets={'A':['continue','self_repair'],'B':['continue','self_repair','self_repair_extended'],
        'C':['continue','self_repair','self_repair_extended','help_base'],'D':['continue','self_repair','self_repair_extended','help_sft']}
    groups=defaultdict(dict)
    for row in rows:groups[row['prefix_id']][row['method']]=row
    complete=[v for v in groups.values() if len(v)==5]
    result['oracle']={name:sum(max(v[m]['reward'] for m in methods) for v in complete) for name,methods in sets.items()}
    result['oracle'].update({'n':len(complete),'C-B':result['oracle']['C']-result['oracle']['B'],
        'D-B':result['oracle']['D']-result['oracle']['B'],'D-C':result['oracle']['D']-result['oracle']['C']})
    result['t2']={'n':len(t2),'beneficial':sum(r['delta_reward']>0 for r in t2),'neutral':sum(r['delta_reward']==0 for r in t2),
        'harmful':sum(r['delta_reward']<0 for r in t2),'future_tokens':{branch:{model:{metric:sum(r[branch+'_future']['tokens_by_model'].get(model,{}).get(metric,0) for r in t2) for metric in ['input','output','calls']} for model in ['A','S']} for branch in ['submit','abort']}}
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else {}
    result['complete']=len(t0['base'])==len(t0['sft'])==40 and bool(selection) and len(complete)==len(selection.get('selected',[]))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    training=json.loads((ROOT/'training_result.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    axes[0].plot(range(1,len(training['per_step_loss'])+1),training['per_step_loss'],label='train minibatch')
    axes[0].plot([r['step'] for r in training['validation_losses']],[r['loss'] for r in training['validation_losses']],marker='o',label='validation')
    axes[0].set(xlabel='optimizer step',ylabel='response loss');axes[0].legend()
    axes[1].bar(['Base','SFT'],[result['t0'][m]['draft_expected_result_match']/40 for m in ['base','sft']])
    axes[1].set(ylim=(0,1),ylabel='Draft expected-result match (proxy)')
    fig.tight_layout();fig.savefig(ROOT/'learning_diagnostics.png',dpi=180);plt.close(fig)
    qualitative=read_rows(ROOT/'t0/qualitative_review.jsonl')
    result['t0_qualitative']={'n':len(qualitative),'base_work_executes':sum(r['base_work_executes_local_reasoning'] for r in qualitative),
        'sft_work_executes':sum(r['sft_work_executes_local_reasoning'] for r in qualitative),
        'base_draft_correct':sum(r['base_draft_correct'] for r in qualitative),'sft_draft_correct':sum(r['sft_draft_correct'] for r in qualitative)}
    local_review=read_rows(ROOT/'t1/local_quality_review.jsonl')
    result['t1_local_review']={method:{'n':sum(r['method']==method for r in local_review),
        'categories':dict(Counter(r['local_task_assessment'] for r in local_review if r['method']==method))} for method in ['help_base','help_sft']}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    text=f'''# DARI-MAS Phase 2B：Protocol Adaptation and Specialist Learning

状态：{'本轮已完成' if result['complete'] else '进行中，以下为当前实际产物'}。

## 工程与评价

采用最小确定性语法修复而非约束解码：仅完整JSON围栏移除和非法反斜杠转义保存；不补字段、答案或截断推理。原始模型输出、token和hidden-state对齐均不变，修复只在Runtime解析入口应用。raw模式默认保留，phase2b统一开启。136个历史非法调用中56个可恢复schema合法，详见format_audit.jsonl；这不是重新生成的能力提升。
数学评价器只从最终答案的明确单一数学表达式提取，不从多段计算选择数字。历史716条结果统一离线重评，4条分数变化，旧产物未覆盖。HotpotQA维持严格EM/F1，不凭gold进行别名匹配；参考矛盾保留benchmark分数，沿用历史数据质量说明。评价器身份phase2b-v1-single-final-math-span在训练前冻结。

## 数据与训练

200条训练、40条验证均为明确标记的构造局部任务，参考推导可检查；没有把未经验证的模型输出当标签，没有虚构人工审核。类别覆盖计算、精确有理数、方程、面积、约束/因式分解、两跳证据、矛盾更新、概率、不充分证据、替代代数求解。训练和验证共享10构造家族，推理措辞存在重复，不能宣称广泛开放式任务泛化。未采集多阶段Expand监督，此能力仅保留回归测试。与正式Runtime相同原题/证据/Anchor上下文/请求/私有历史prompt，单阶段512预算。
旧84题、训练、验证、新36道下游题规范化问题哈希零重叠，见leakage_audit.json。没有以数值替换同家族构造题的做法冒充跨模板独立测试；未声称能自动检出所有语义改写重叠。

原始Qwen3-8B独立Specialist，BF16 LoRA rank16/alpha32/dropout0.05，q/k/v/o projections，AdamW2e-5，batch1累积8，2轮50步，seed20261126。Anchor冻结原始模型，不加载adapter；Controller固定策略，无DPO或正式效用训练。训练response-only mask排除prompt/padding，不静默截断，max_length2048。trainable={training['trainable_parameters']}；冻结参数版本检查={training['frozen_parameters_unchanged']}。验证loss={training['validation_losses']}；训练时间{training['total_wall_seconds']:.1f}s；分片峰值内存{training['peak_memory_bytes']}bytes。训练loss下降不视为协作能力改善。

## T0 局部任务

{json.dumps(result['t0'],ensure_ascii=False,indent=2)}

自动指标是受限expected-result出现检查，不等同完整语义正确率、任务完成率或Evidence Consistency。可能在过程里提及正确数值但结论错误，也可能输出等价表达而未命中。work/draft双命中仅为结果一致性代理，不能冒充Advisory-only的反向指标。复杂证据行为需检查真实case；不得从模板匹配得分推导一般能力提升。Base/SFT完全相同40个输入、seed、温度0.7/top_p0.95、512token/384draft、语法修复方式。
预先固定前12条的直接轨迹审阅：{result['t0_qualitative']}。详见t0/qualitative_review.jsonl，非盲法独立人工标注。SFT在面积题中将实际计算移入work，但多个证据/不充分信息/概率样本照抄输入“局部问题尚未解决”，不能视作可靠不确定性报告；还有明确算术错误。模板适应造成退化可能与狭窄合成监督及复制Anchor输入有关，当前仅为有证据的行为观察，未做因果消融。不追加数据或重训来追逐当前验证分数。

## T1 同前缀下游

36新题先运行冻结DARI Anchor；首个无答案且五比较可行的状态入候选，固定seed打乱后至多30个。selection.json在分支前固定，前12为pilot，后续复用相同配置。所有五条件同前缀、任务、宽度、预算及后续finish-else-continue策略。单阶段Help，Base通过关闭LoRA而SFT启用同一已冻结adapter，A是完全独立原始模型。每分支增量统计包含失败；T2额外调用不算T1资源。
实际36题仅产生18个合法checkpoint，低于建议24–30，本轮不为凑数新增题或人为延长推理。T2因此最多18个合法SFT Draft，实际数按全部尝试报告；该覆盖限制降低结论可信度。

{json.dumps(result['t1'],ensure_ascii=False,indent=2)}

分数据集统计：{json.dumps(result['t1_by_dataset'],ensure_ascii=False)}。

Help-SFT相对控制的Win/Tie/Loss：{result['t1_pairs']}。SFT实际提交条件子集：{result['t1_delivered_conditional']}，有选择偏差，不能替代全尝试总体。
时间采用各T1 trace实际模型调用latency之和，不含额外T2 Abort生成；原results的Help-SFT外层latency包含紧随的T2计算，不能用该字段直接比较五动作墙钟时间。T1 e0仅用于固定前缀生成，其留存旧评分不进入本节五动作结论；本节均用冻结corrected evaluator。
预选前4状态的局部执行质量审阅（8条输出）：{result['t1_local_review']}，详见t1/local_quality_review.jsonl。未对全部开放式任务给出伪造的自动Local Task Accuracy或语义完成率。

## T2 选择性通信

{json.dumps(result['t2'],ensure_ascii=False,indent=2)}

按预定状态顺序取前25个合法SFT Draft，不按奖励筛选。两分支使用同一完整Specialist-side状态，Submit和Abort前成本相同；Abort清除help并不传Draft/私有work，之后均无新Help，公共续写机会一致。已执行Submit直接复用T1结果，Abort真实生成；future资源仅统计决策后的新增调用。记录Delta_R不创建DPO标签。观察有害信息不代表当前Controller能够识别它。

## Local Oracle

{result['oracle']}。A=Continue+Repair；B增加Extended；C增加Base；D为B增加SFT。均为有限rollout事后终局上界，非成本感知部署策略。

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
'''
    (ROOT/'report.md').write_text(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':analyze()
