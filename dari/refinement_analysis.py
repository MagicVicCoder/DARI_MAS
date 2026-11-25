"""Auditable aggregation of refinement artifacts; no model judge or score edits."""
from collections import Counter, defaultdict
import csv
import json
import statistics

from .pilot import read_rows
from .refinement import ROOT, OLD


def mean(rows,key):
    values=[r[key] for r in rows if isinstance(r.get(key),(int,float))]
    return sum(values)/len(values) if values else 0


def table(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['dataset'],row['method'])].append(row)
    lines=['|数据集|方法|n|正确/EM|F1|无答案|格式失败调用|平均新步骤|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for (dataset,method),group in sorted(groups.items()):
        f1=f'{mean(group,"f1"):.3f}' if dataset=='hotpotqa' else '—'
        lines.append(f'|{dataset}|{method}|{len(group)}|{mean(group,"reward"):.3f}|{f1}|{sum(r.get("no_answer",False) for r in group)}|{sum(r.get("format_failures",0) for r in group)}|{mean(group,"reasoning_steps"):.2f}|')
    return '\n'.join(lines)


def collaboration(rows,label):
    groups=defaultdict(dict)
    for row in rows:
        groups[row['prefix_id']][row['method']]=row
    complete={k:v for k,v in groups.items() if set(v)=={'continue','self_repair','self_repair_extended','help_submit'}}
    delivered={k:v for k,v in complete.items() if v['help_submit']['submitted']}
    output={'checkpoints':len(groups),'complete_matched':len(complete),'delivered':len(delivered),'comparisons':{},'oracle':{}}
    lines=[f'{label}：完整配对 {len(complete)} 个状态，成功交付 {len(delivered)} 个。']
    for name,subset in [('intention_to_help',complete),('delivered_conditional',delivered)]:
        output['comparisons'][name]={}
        for control in ['continue','self_repair','self_repair_extended']:
            diffs=[v['help_submit']['reward']-v[control]['reward'] for v in subset.values()]
            wtl=[sum(d>0 for d in diffs),sum(d==0 for d in diffs),sum(d<0 for d in diffs)]
            output['comparisons'][name][control]={'n':len(diffs),'win_tie_loss':wtl,'mean_delta':sum(diffs)/len(diffs) if diffs else None}
            f1diffs=[v['help_submit']['f1']-v[control]['f1'] for v in subset.values() if v[control]['dataset']=='hotpotqa']
            output['comparisons'][name][control]['hotpot_f1_mean_delta']=sum(f1diffs)/len(f1diffs) if f1diffs else None
            lines.append(f'- {name} Help vs {control}：Win/Tie/Loss={wtl}。')
    sets={'A':['continue','self_repair'],'B':['continue','self_repair','help_submit'],
          'C':['continue','self_repair','self_repair_extended'],'D':['continue','self_repair','self_repair_extended','help_submit']}
    for name,methods in sets.items():
        output['oracle'][name]=sum(max(v[m]['reward'] for m in methods) for v in complete.values())
    output['oracle']['B-A']=output['oracle']['B']-output['oracle']['A']
    output['oracle']['D-C']=output['oracle']['D']-output['oracle']['C']
    hotpot=[v for v in complete.values() if v['continue']['dataset']=='hotpotqa']
    output['oracle_hotpot_f1']={name:sum(max(v[m]['f1'] for m in methods) for v in hotpot)/len(hotpot) if hotpot else None for name,methods in sets.items()}
    if hotpot:
        output['oracle_hotpot_f1'].update({'B-A':output['oracle_hotpot_f1']['B']-output['oracle_hotpot_f1']['A'],
            'D-C':output['oracle_hotpot_f1']['D']-output['oracle_hotpot_f1']['C']})
    lines.append(f'\nLocal Oracle 成功数（分母 {len(complete)}）：'+str(output['oracle'])+'。这是事后上界，非可部署策略。')
    lines.append(f'\nHotpot 条件内 F1 Oracle（n={len(hotpot)}）：'+str(output['oracle_hotpot_f1'])+'。各控制的Help-F1差异另见summary.json。')
    return output,'\n'.join(lines)


def audit_traces(paths):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/models/Qwen3-8B',local_files_only=True)
    count=lambda text:len(tokenizer.encode(text,add_special_tokens=False))
    audit=[]; requests=[]; failures=[]
    for path in paths:
        result_index={(r['task_id'],r['method']):r for r in read_rows(path.with_name('results.jsonl'))}
        for row in read_rows(path):
            new_steps=result_index.get((row['task_id'],row['method']),{}).get('reasoning_steps',len(row['steps']))
            for index,step in enumerate(row['steps'][-new_steps:] if new_steps else []):
                rec=step['record']; req=rec.get('help_request')
                requests.append({'stage':str(path.parent.parent.name),'task_id':row['task_id'],'method':row['method'],
                    'step':step['step_id'],'unfinished':rec.get('answer') is None,
                    'authored':req is not None,'instruction':req['instruction'] if req else '',
                    'current_reasoning':rec['step'],'goal':rec.get('goal')})
            for event in row['events']:
                if not event['valid']:
                    failure={'task_id':row['task_id'],'method':row['method'],'stage':str(path.parent.parent.name),
                        'role':event['role'],'runtime_error':event['error'],'output_tokens':event['usage']['output_tokens'],
                        'hit_stage_output_cap':event['usage']['output_tokens']>=512}
                    try:
                        json.loads(event['output'])
                        failure['json_status']='complete_json_rejected_by_schema_refs_or_quota'
                    except json.JSONDecodeError as error:
                        failure['json_status']=error.msg
                    failures.append(failure)
                if event['role']!='specialist':
                    continue
                record={'task_id':row['task_id'],'stage':str(path.parent.parent.name),
                    'prefix_id':row.get('prefix_id'),'valid':event['valid'],'error':event['error'],
                    'output_tokens':event['usage']['output_tokens'],'hit_cap':event['usage']['output_tokens']>=512}
                try:
                    obj=json.loads(event['output']); work=obj['work']; draft=obj['draft']
                    record.update(work=work,draft=draft,work_tokens=count(work),draft_tokens=count(draft),
                        placeholder_work=work.strip().lower()=='private local analysis',
                        json_complete=True,serialized_draft_tokens=count(json.dumps(draft,ensure_ascii=False)))
                except (ValueError,KeyError,TypeError):
                    record.update(json_complete=False,raw_output=event['output'])
                audit.append(record)
    (ROOT/'specialist_audit.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in audit))
    (ROOT/'protocol_failure_audit.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in failures))
    with (ROOT/'request_audit.csv').open('w') as stream:
        if requests:
            writer=csv.DictWriter(stream,fieldnames=list(requests[0])); writer.writeheader(); writer.writerows(requests)
    return audit,requests


def request_summary(requests):
    groups=defaultdict(list)
    for row in requests:
        groups[(row['stage'],row['method'])].append(row)
    output={}
    for (stage,method),rows in groups.items():
        unfinished=[r for r in rows if r['unfinished']]
        by_task=defaultdict(list)
        repeats=0
        for r in rows:
            if r['authored']:
                text=' '.join(r['instruction'].lower().split())
                repeats+=text in by_task[r['task_id']]
                by_task[r['task_id']].append(text)
        output[stage+'/'+method]={'valid_steps':len(rows),'unfinished_steps':len(unfinished),
            'authored_unfinished':sum(r['authored'] for r in unfinished) if method!='stepwise_cot' else None,
            'fallback_unfinished':sum(not r['authored'] for r in unfinished) if method!='stepwise_cot' else None,'exact_repeats':repeats}
    return output


def main():
    plan=json.loads((ROOT/'selection.json').read_text()); root_cause=json.loads((ROOT/'root_cause.json').read_text())
    r1=read_rows(ROOT/'r1/e0/results.jsonl'); r2=read_rows(ROOT/'r2/e1/results.jsonl')
    fresh_branches=ROOT/'r3_branches'
    r3e0=read_rows(ROOT/'r3/e0/results.jsonl'); r3e1=read_rows(fresh_branches/'e1/results.jsonl')
    oldr1=[]
    for row in read_rows(OLD/'e0/results.jsonl'):
        if row['task_id'] in plan['r1_task_ids']:
            oldr1.append({**row,'method':'original_'+row['method']})
    r1=[r for r in r1 if r.get('status')=='completed']; r2=[r for r in r2 if r.get('status')=='completed']
    r3e0=[r for r in r3e0 if r.get('status')=='completed']; r3e1=[r for r in r3e1 if r.get('status')=='completed']
    r2summary,r2text=collaboration(r2,'R2'); r3summary,r3text=collaboration(r3e1,'R3')
    audit,requests=audit_traces([ROOT/'r1/e0/traces.jsonl',ROOT/'r2/e1/traces.jsonl',ROOT/'r3/e0/traces.jsonl',fresh_branches/'e1/traces.jsonl'])
    all_results=r1+r2+r3e0+r3e1+read_rows(ROOT/'r0/replay384_results.jsonl')
    resource_rows=[]
    for stage,rows in [('r0',read_rows(ROOT/'r0/replay384_results.jsonl')),('r1',r1),('r2',r2),('r3_e0',r3e0),('r3_e1',r3e1)]:
        groups=defaultdict(list)
        for row in rows:
            groups[(row['dataset'],row['method'])].append(row)
        for (dataset,method),group in groups.items():
            resource={'stage':stage,'dataset':dataset,'method':method,'n':len(group),'e2e_ms_sum':sum(r.get('e2e_ms',0) for r in group)}
            for model in ['A','S']:
                for metric in ['input','output','calls']:
                    resource[model+'_'+metric]=sum(r.get('tokens_by_model',{}).get(model,{}).get(metric,0) for r in group)
            resource_rows.append(resource)
    if resource_rows:
        with (ROOT/'resources.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(resource_rows[0])); writer.writeheader();writer.writerows(resource_rows)
    resources=defaultdict(Counter)
    for row in all_results:
        for model,counts in row.get('tokens_by_model',{}).items():
            resources[model].update(counts)
    summary={'root_cause':root_cause,'r1_runs':len(r1),'r2':r2summary,'r3_e0_runs':len(r3e0),'r3':r3summary,
        'specialist_calls_audited':len(audit),'valid_specialist':sum(r['valid'] for r in audit),
        'placeholder_work':sum(r.get('placeholder_work',False) for r in audit),'specialist_errors':dict(Counter(r['error'] for r in audit if r['error'])),
        'resources':{k:dict(v) for k,v in resources.items()},'actual_wall_ms_sum':sum(r.get('e2e_ms',0) for r in all_results)}
    summary['requests']=request_summary(requests)
    old_requests=[]
    for row in read_rows(OLD/'e0/traces.jsonl'):
        if row['task_id'] not in plan['r1_task_ids']:
            continue
        for step in row['steps']:
            rec=step['record'];req=rec.get('help_request')
            old_requests.append({'stage':'original_r1','method':row['method'],'task_id':row['task_id'],
                'unfinished':rec.get('answer') is None,'authored':req is not None,'instruction':req['instruction'] if req else ''})
    summary['original_requests']=request_summary(old_requests)
    expected_new_branches=4*sum(bool(r.get('prefix_id')) for r in r3e0 if r['method']=='dari_anchor_only')
    summary['complete']=len(r1)==48 and len(r2)==144 and len(r3e0)==72 and len(r3e1)==expected_new_branches
    review=[r for r in all_results if r.get('evaluator_status') in {'manual_review_required','unparsed_answer'}
            or (r.get('dataset')=='math500' and r.get('reward')==0 and r.get('answer') is not None)]
    summary['evaluator_review_count']=len(review)
    (ROOT/'evaluator_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    manual=read_rows(ROOT/'anchor_manual_review.jsonl')
    summary['anchor_manual_categories']=dict(Counter(row['category'] for row in manual))
    summary['r1_failure_review']=dict(Counter(row['category'] for row in read_rows(ROOT/'r1_failure_review.jsonl')))
    qualitative=read_rows(ROOT/'specialist_qualitative_review.jsonl')
    summary['qualitative_work_review']={'n':len(qualitative),'categories':dict(Counter(r['work_assessment'] for r in qualitative))}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    replay=read_rows(ROOT/'r0/replay384_results.jsonl')
    r0old=[r for r in read_rows(OLD/'e1/results.jsonl') if r['prefix_id'] in plan['r0_prefix_ids'] and r['method']=='help_submit']
    r0new=[r for r in r2 if r['prefix_id'] in plan['r0_prefix_ids'] and r['method']=='help_submit']
    r0audit=[r for r in audit if r['stage']=='r2' and r['prefix_id'] in plan['r0_prefix_ids']]
    r0lengths={key:{'min':min(values),'median':statistics.median(values),'max':max(values)}
        for key in ['work_tokens','draft_tokens'] if (values:=[r[key] for r in r0audit if key in r])}
    r0table=table([{**r,'method':'archival_old192_old_anchor'} for r in r0old]+replay+[{**r,'method':'refined384_new_anchor'} for r in r0new])
    text=f'''# DARI-MAS Phase 2A-Refinement 实验报告

状态：{'实验运行齐全；见下方实际统计及审计。' if summary['complete'] else '进行中；当前统计不完整，尚不能作为阶段最终结论。'}

## 范围与可复现性

冻结原始 Qwen3-8B Anchor/Specialist，无适配器、训练、额外 Agent 或 LLM Judge。本轮为探索性开发复验，非正式独立测试、显著性检验或方法有效性结论。旧结果保持不变；所有失败进入统计。
模型路径 `/data/models/Qwen3-8B`，权重身份沿用原 manifest。每阶段 manifest 记录配置、逐文件代码哈希、样本指纹及独立 A/S 部署身份。选择种子 20261125；R0/R2 先固定随机 12/原有 36 个前缀，R1 使用原 pilot 每集 8 题；R3 每集 8 题，与原 60 题不重复，在推理前固定。HotpotQA 原文与离线 gold 分离，不将 gold/supporting facts 传模型。

## 根因分析

旧 36 个 Specialist 均输出完整 JSON，work 全部是占位字面量 `private local analysis`。代码不自动填充该值，提示伪示例被模型照抄。22 次拒绝均在解析后进行完整序列化 Draft 配额检查时发生，而非生成截断。原题 36/36 完整存在；“缺少售价”的案例属于未正确利用原题，不能归因于原题被删。

旧 Draft compact 长度 min/median/max={root_cause['draft_tokens']}。离线预算可交付数量：{root_cause['offline_budget_sensitivity']} /36。预算放宽只说明长度合法，不能推出内容正确或下游受益。原自主请求 10 个，fallback 26 个，均来自合法 null/missing；非解析器把错误请求静默改成 fallback。Anchor 失败初筛：{root_cause['anchor_failure_categories']}。其中有效但错误的答案还需逐例区分真实推理和局部结果提前回答，不把所有错误标为提前 Finish。原轨迹、字段长度及分类保存在 root_cause.json、r0_original_outputs.jsonl 和 anchor_failure_audit.jsonl。

进一步逐条审阅13个合法但失分案例（不覆盖评分）：{summary['anchor_manual_categories']}。只有3个明确将中间结果作为最终答案；另有真实算术/计数/整除错误，以及 exact-answer/EM 限制。证据和解释在 anchor_manual_review.jsonl。

另发现原始数据一致性问题：gsm8k:952题干为8:00AM到11:00PM，参考推导却按11:00AM计算360；按题干应为1800。原三方法和新版两方法均回答1800且均在主分中失分。data_quality_review.jsonl保留原行核查说明，未修改gold、题目或主分。这是数据噪声，不能算作协议推理失败。

## 实际修改

Specialist 主流程仅生成两个字符串 work/draft，由 Runtime 维护 target/session/stage；Draft 配额按实际文本 token 统计，默认384。完整阶段预算默认3×512，实验单阶段512。要求独立执行开放式局部任务，提供实际计算、证据、结论或有据的部分进展；移除占位示例和固定审查字段。原题、选中原文、Anchor 推理、动态任务及当前会话私有历史分区，原题仅序列化一次。Narrow/Wide 仍按固定规则，不事后选上下文。

Anchor 历史仅保留 ID、step、refs；候选答案另列。强调每步实际推进、回答完整原题、未完成时提出具体局部任务，Controller 决定是否调用，允许无价值任务为 null。保留原状态机、失效候选机制、隔离会话、有限重试及带成本失败。JSON Draft 特征池化只取字符串内容。SFT/DPO 与 Controller checkpoint schema 升为3；历史对象 Draft 仅离线解析。

## R0：交付诊断

离线统计见上。固定12状态旧192结果直接归档复用；旧结构384复用相同历史生成，真实重跑新版 Anchor；新版384使用新版 Specialist 并与 R2 相同记录复用。后两者使用同一下游提示与原始前缀，历史 Specialist 复用调用成本单列。旧192下游是旧版 Anchor，不能直接把差异归因于 Draft 协议；新旧 Specialist 结构/提示均改变，非单因素消融。

{r0table}

新版固定12状态已完成 {len(r0new)} 条 Help，提交 {sum(r['submitted'] for r in r0new)} 条；长度分布 {r0lengths}。原work均3 tokens且为占位符；新版非占位符不自动等同实质推理，见下方定性审阅。

## R1：Anchor 稳定性

原24题三条件复用；新版两条件实际运行 {len(r1)}/48 次。评分沿用原 rational/math-verify/Hotpot EM/F1，无人工覆盖主分。格式失败为失败调用数，可能包括成功重试。

{table(oldr1+r1)}

新版全部16个失分运行已逐条核对：{summary['r1_failure_review']}。明确中间结果作最终答案1例（gs333只给两项总价、不求差），6例因Anchor格式失败无候选，另有真实系数比较错误、答案类型错误、参考矛盾和严格EM/别名限制。没有把全部失分称为提前Finish；详见r1_failure_review.jsonl。逐调用JSON错误类型及是否触及512输出上限另列protocol_failure_audit.jsonl，不混淆截断和非法转义。

## R2：同前缀协作

原36前缀所有四控制均按新版 Anchor 重新运行，共 {len(r2)}/144 次。先固定12状态诊断，再扩展。Continue/Repair 各一调用；Extended 两次 Anchor；Help 一次 Specialist 加一次整合；之后共同至多8个新有效步骤，候选即终止。共享前缀只计算一次，分支资源统计为新增调用。原始前缀逐次哈希保持不变。

{table(r2)}

{r2text}

Delivered 子集为条件性机制诊断，不能替代总体，也不消除选择偏差。引用保留不等于实际采用，正确答案不等于局部推理完全正确。

## R3：新开发样本

实际 E0 {len(r3e0)}/72 次，分支 {len(r3e1)} 次；一步完成者没有 checkpoint，coverage.jsonl 逐题记录，不人为延长推理。未根据新结果调整提示。
为复用空闲设备，R3分为两个执行manifest：r3在GPU0/1/5分片Anchor，未调用的Specialist留CPU，仅跑E0；r3_branches在GPU6/7和2/3分别分片A/S，重用原样保存的前缀跑全部四种E1动作。模型、提示、采样和评分不变；部署差异明确记录，不能直接比较跨阶段墙钟速度。分支目录只链接不可变E0索引和前缀，不重新生成或覆盖它们。

{table(r3e0)}

{table(r3e1)}

{r3text}

## 资源与诊断文件

真实增量 token/调用统计：`{json.dumps(summary['resources'],ensure_ascii=False)}`。运行计时合计 {summary['actual_wall_ms_sum']/1000:.1f} 秒，含 Runtime/日志与共享服务器波动，不含模型加载，不是已校准 GPU 成本。旧生成复用不冒充新调用；原 token 在 R0 replay 记录单列。
各阶段、数据集、条件、模型的输入/输出tokens、调用数及用时总和另列 resources.csv。

审计 Specialist 调用 {len(audit)}，合法 {summary['valid_specialist']}，原占位符 {summary['placeholder_work']}；失败原因 {summary['specialist_errors']}。specialist_audit.jsonl 保留 work/draft/长度/错误；request_audit.csv 保留每步请求及推理供具体性、一致性和可执行性人工复核。不将非占位 work 自动等同正确推理，不以正则冒充语义评价。

评分器待复核记录 {len(review)} 条，单列 evaluator_review.jsonl，包含所有MATH非空但失分答案，未覆盖主分。math-verify的scored标志并不能保证解析语义可靠：例如24313的整句“The simplified expression is … cos(x)/sin(x)”被解析成字母乘积而失分，单独cot(x)则得分；该差异不能解释为真实推理能力差距。

新增有效步骤的请求统计（unfinished仅指模型未给候选，不代表客观未解；无候选且null时Runtime提供fallback）：`{json.dumps(summary['requests'],ensure_ascii=False)}`。E1不重复计算共享前缀；原R2请求身份保持原始10自主/26fallback，不将新版后续请求当作原前缀改善。
同24题原始请求统计：`{json.dumps(summary['original_requests'],ensure_ascii=False)}`。Stepwise关闭请求生成，相关比例不适用；这些统计也不代表每一步都有可执行的Help动作。
R1新版9个自主请求逐条定性审阅（r1_request_quality_review.jsonl）：9/9具有具体对象、与当前步骤一致且可从已有输入执行；其中EF长度已在原图标7，属于具体但低边际价值的任务。旧10自主请求也均具体，但买房原价请求重复题干已给值，出生年请求在现有原文中无精确数据（original_request_review.jsonl）。具体性不等同有价值，更不等同值得调用。

## 局部求解内容的定性审阅

已逐条审阅首批 {len(qualitative)} 条真实交付，work类别为 {summary['qualitative_work_review']['categories']}，详细说明在 specialist_qualitative_review.jsonl。这是有限、可复核的轨迹审阅，非独立语义评分、无额外Judge、未覆盖全部状态。部分work仍只建议Anchor计算，而draft才实际给出算式与结果；例如烘焙长度draft算出3360英寸/12=280英尺，work并未执行这些运算。因此局部求解与交付已有真实实例，但Private Work字段的功能适应并未完全解决，不能只按占位符消失判定成功。

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
'''
    (ROOT/'report.md').write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
