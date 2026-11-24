"""Tables, transparent local-oracle diagnostics and static research figures."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess

from .pilot import METHODS, BRANCHES, read_rows


def mean(values):
    return statistics.mean(values) if values else None


def resource(row, key):
    return sum(values.get(key,0) for values in row.get("tokens_by_model",{}).values())


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row["method"],row["dataset"])].append(row)
    output=[]
    for (method,dataset), group in sorted(groups.items()):
        output.append({"method":method,"dataset":dataset,"n":len(group),
            "accuracy_em":mean([row["reward"] for row in group]),
            "f1":mean([row["f1"] for row in group if row.get("f1") is not None]),
            "input_tokens":mean([resource(row,"input") for row in group]),
            "output_tokens":mean([resource(row,"output") for row in group]),
            "calls":mean([resource(row,"calls") for row in group]),
            "latency_ms":mean([row["e2e_ms"] for row in group]),
            "format_failures":sum(row["format_failures"] for row in group),
            "first_format_failure_rate":mean([row["first_format_failure"] for row in group]),
            "checkpoint_first_format_failure_rate":sum(row.get("checkpoint_first_failures",0) for row in group)/max(1,sum(row.get("checkpoint_first_attempts",0) for row in group)),
            "terminal_format_failure_rate":mean([row["termination_reason"]=="anchor_format" for row in group]),
            "step_limit_rate":mean([row["termination_reason"]=="step_limit" for row in group]),
            "no_answer_rate":mean([row["no_answer"] for row in group]),
            "avg_reasoning_steps":mean([row["reasoning_steps"] for row in group]),
            "request_exact_repetition_rate":sum(row["request_repeated"] for row in group)/max(1,sum(row["request_count"] for row in group)),
            "request_similarity_repetition_rate":sum(row["request_similar"] for row in group)/max(1,sum(row["request_count"] for row in group)),
            "unresolved_evaluator_count":sum(row["evaluator_status"] in {"manual_review_required","unparsed_answer"} for row in group)})
    return output


def matched_groups(rows):
    grouped=defaultdict(dict)
    for row in rows:
        grouped[row["prefix_id"]][row["method"]]=row
    return {prefix:group for prefix,group in grouped.items() if set(BRANCHES)<=set(group)}


def local_oracle(groups):
    output=[]
    for dataset in ("all","gsm8k","math500","hotpotqa"):
        selected=[group for group in groups.values() if dataset=="all" or group["continue"]["dataset"]==dataset]
        if not selected:
            continue
        for label,methods in (("core",("continue","self_repair","help_submit")),("extended",BRANCHES)):
            fixed={method:mean([group[method]["reward"] for group in selected]) for method in methods}
            oracle=mean([max(group[method]["reward"] for method in methods) for group in selected])
            best=max(fixed,key=fixed.get)
            output.append({"action_set":label,"dataset":dataset,"n":len(selected),"local_oracle":oracle,
                "best_fixed_action":best,"best_fixed_reward":fixed[best],"oracle_gap":oracle-fixed[best],
                "states_with_action_disagreement":sum(len({group[method]["reward"] for method in methods})>1 for group in selected),
                **fixed})
    return output


def markdown(rows,columns):
    if not rows:
        return "尚无完整结果。\n"
    def cell(value):
        return f"{value:.4f}" if isinstance(value,float) else str(value if value is not None else "—")
    return "| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(cell(row.get(key)) for key in columns)+" |" for row in rows)+"\n"


def analyze(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text())
    samples=read_rows(root/"samples.jsonl")
    (root/"e1").mkdir(exist_ok=True)
    # Only final completed result for a key; failed attempts remain separately in raw JSONL.
    def completed(path,keys):
        return list({tuple(row.get(key) for key in keys):row for row in read_rows(path) if row["status"]=="completed"}.values())
    e0=completed(root/"e0/results.jsonl",("task_id","method"))
    e1=completed(root/"e1/results.jsonl",("task_id","method","repeat","specialist_temperature"))
    robust=completed(root/"robustness/results.jsonl",("task_id","method","repeat","specialist_temperature"))
    # Derive first-attempt task failure from all Anchor checkpoints, rather than
    # confusing it with only the very first call of a trajectory.
    for directory, rows in (("e0",e0),("e1",e1),("robustness",robust)):
        event_map={(row["task_id"],row["method"],row.get("repeat",0),row.get("specialist_temperature",.7)):row for row in read_rows(root/directory/"traces.jsonl")}
        for row in rows:
            events=event_map.get((row["task_id"],row["method"],row.get("repeat",0),row.get("specialist_temperature",.7)),{}).get("events",[])
            retry=False
            first_failed=0
            first_attempts=0
            for event in events:
                if event["role"]!="anchor":
                    continue
                if not retry:
                    first_attempts+=1
                    first_failed+=not event["valid"]
                    retry=not event["valid"]
                else:
                    retry=False
            row["first_format_failure"]=first_failed>0
            row["checkpoint_first_attempts"]=first_attempts
            row["checkpoint_first_failures"]=first_failed
    groups=matched_groups(e1)
    matched=[row for group in groups.values() for row in group.values()]
    e0summary,e1summary=summary(e0),summary(matched)
    write_csv(root/"e0/summary.csv",e0summary)
    write_csv(root/"e1/summary.csv",e1summary)
    oracle=local_oracle(groups)
    write_csv(root/"e1/oracle.csv",oracle)
    paired=[]
    for dataset in ("all","gsm8k","math500","hotpotqa"):
        selected=[group for group in groups.values() if dataset=="all" or group["continue"]["dataset"]==dataset]
        for baseline in BRANCHES[:-1]:
            differences=[group["help_submit"]["reward"]-group[baseline]["reward"] for group in selected]
            if differences:
                paired.append({"dataset":dataset,"baseline":baseline,"n":len(differences),
                    "win":sum(value>0 for value in differences),"tie":sum(value==0 for value in differences),
                    "loss":sum(value<0 for value in differences),"mean_gain":mean(differences)})
    write_csv(root/"e1/win_tie_loss.csv",paired)
    matrix=[{"prefix_id":prefix,"task_id":group["continue"]["task_id"],"dataset":group["continue"]["dataset"],
             **{method:group[method]["reward"] for method in BRANCHES},
             **{"delta_help_"+method:group["help_submit"]["reward"]-group[method]["reward"] for method in BRANCHES[:-1]}}
             for prefix,group in groups.items()]
    write_csv(root/"e1/state_matrix.csv",matrix)
    assistance=[row for row in e1 if row["method"]=="help_submit"]
    request_stats={"eligible_checkpoints":len(groups),"e0_dari_tasks":sum(row["method"]=="dari_anchor_only" for row in e0),
        "help_branches":len(assistance),"submitted":sum(row["submitted"] for row in assistance),
        "explicitly_retained":sum(row["explicitly_retained"] for row in assistance),
        "fallback":sum(row["dynamic_request"]["fallback"] for row in assistance),
        "request_exact_repetitions":sum(row["request_repeated"] for row in e0 if row["method"]=="dari_anchor_only"),
        "request_total":sum(row["request_count"] for row in e0 if row["method"]=="dari_anchor_only")}
    (root/"e1/request_summary.json").write_text(json.dumps(request_stats,indent=2))
    request_examples=[{"prefix_id":row["prefix_id"],"task_id":row["task_id"],"dataset":row["dataset"],
                      "instruction":row["dynamic_request"]["instruction"],"target":row["dynamic_request"]["target"],
                      "fallback":row["dynamic_request"]["fallback"],"selected_context_ids":row["selected_context_ids"],
                      "submitted":row["submitted"],"reward":row["reward"],"explicitly_retained":row["explicitly_retained"]}
                      for row in assistance[:20]]
    write_csv(root/"e1/request_examples.csv",request_examples)
    f1_paired=[]
    for baseline in BRANCHES[:-1]:
        values=[group["help_submit"]["f1"]-group[baseline]["f1"] for group in groups.values() if group["continue"]["dataset"]=="hotpotqa"]
        if values:
            f1_paired.append({"baseline":baseline,"n":len(values),"tie_tolerance":1e-8,
                "win":sum(value>1e-8 for value in values),"tie":sum(abs(value)<=1e-8 for value in values),
                "loss":sum(value < -1e-8 for value in values),"mean_f1_gain":mean(values)})
    write_csv(root/"e1/f1_win_tie_loss.csv",f1_paired)
    uncertainty=[{key:row.get(key) for key in ("task_id","dataset","method","answer","extracted","evaluator_status")} for row in e0+e1 if row["evaluator_status"] in {"manual_review_required","unparsed_answer"}]
    write_csv(root/"evaluator_review.csv",uncertainty)
    if robust:
        write_csv(root/"robustness/summary.csv",[{"method":method,"specialist_temperature":temp,
            "n":len(sub),"mean_reward":mean([row["reward"] for row in sub]),
            "mean_output_tokens":mean([resource(row,"output") for row in sub])}
            for method in ("self_repair_extended","help_submit") for temp in (0.,.7)
            if (sub:=[row for row in robust if row["method"]==method and row["specialist_temperature"]==temp])])
    figures(root,e0summary,e1summary,paired,matrix,oracle)
    # Select cases by observed action outcomes, for analysis only, never to change samples.
    cases={}
    for label,condition in {
        "help_better_than_continue":lambda g:g["help_submit"]["reward"]>g["continue"]["reward"],
        "self_repair_repairs":lambda g:g["self_repair"]["reward"]>g["continue"]["reward"],
        "help_and_internal_succeed":lambda g:g["help_submit"]["reward"]==g["self_repair_extended"]["reward"]==1,
        "help_no_additional_gain":lambda g:g["help_submit"]["reward"]==g["continue"]["reward"],
        "help_harms":lambda g:g["help_submit"]["reward"]<g["continue"]["reward"],
        "invalid_specialist":lambda g:not g["help_submit"]["submitted"]}.items():
        chosen=next((group for group in groups.values() if condition(group)),None)
        cases[label]={"observed":chosen is not None,"task_id":chosen["continue"]["task_id"] if chosen else None}
    (root/"case_index.json").write_text(json.dumps(cases,indent=2))
    traces=read_rows(root/"e1/traces.jsonl")
    sample_map={row["task_id"]:row for row in samples}
    case_text=[]
    for label,case in cases.items():
        case_text.append("### "+label+"\n")
        if not case["observed"]:
            case_text.append("当前样本未观察到。\n")
            continue
        task_id=case["task_id"]
        sample=sample_map[task_id]
        group=next(group for group in groups.values() if group["continue"]["task_id"]==task_id)
        case_text.append("Task: `"+task_id+"`\n\n问题："+sample["question"]+"\n\n参考答案："+sample["gold"]+"\n\n动态任务："+group["help_submit"]["dynamic_request"]["instruction"]+"\n")
        case_text.append(markdown([{"action":method,"answer":group[method]["answer"],"reward":group[method]["reward"]} for method in BRANCHES],["action","answer","reward"]))
        help_trace=next((row for row in traces if row["task_id"]==task_id and row["method"]=="help_submit"),None)
        if help_trace:
            specialist=next((event for event in help_trace["events"] if event["role"]=="specialist"),None)
            if specialist:
                case_text.append("Specialist 实际输出（合法性="+str(specialist["valid"])+"）：\n\n```json\n"+specialist["output"]+"\n```\n")
    actual={"input_tokens":sum(resource(row,"input") for row in e0+e1+robust),
        "output_tokens":sum(resource(row,"output") for row in e0+e1+robust),
        "calls":sum(resource(row,"calls") for row in e0+e1+robust),
        "measured_rollout_ms":sum(row["e2e_ms"] for row in e0+e1+robust)}
    (root/"resource_summary.json").write_text(json.dumps(actual,indent=2))
    report=f'''# DARI-MAS Phase 2A 探索性实验报告

## 背景与范围

冻结原始 Qwen3-8B Anchor/Specialist，无 LoRA，无正式 Controller 决策或训练。
研究 E0 的协议稳定性，以及 E1 同前缀状态上的不同协作行为。结果不是正式 benchmark、
显著性检验、训练标签或方法有效性结论。仅使用真实生成，不删去负向/失败案例。

## 配置与可复现性

基础 commit `{manifest['git_commit']}`；实验身份 `{manifest['experiment_id']}`。
本地代码归档HEAD：`{subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}`；
实验启动时部分新增代码尚未提交，因此以manifest中的逐文件代码哈希为准。
完整配置、代码哈希、数据指纹和模型身份在 manifest.json、model_manifest.json 和
data_manifest.json。60个预选开发样本，每数据集前8个为Pilot，20个为扩展目标；
选样固定种子20261124，MATH按题型/难度轮转，其他数据集随机无放回。
GSM8K本地副本上游revision未知，使用内容哈希；MATH-500/HotpotQA从HF镜像获取，
HotpotQA为distractor validation，所有10篇原文进入Anchor，supporting facts/gold只在离线数据中。
这些开发样本若用于训练须从后续独立测试中排除。

输入prompt、token/用时/合法性记录于traces.jsonl；原始行含参考答案仅存samples.jsonl，
不传模型。GSM用有理数规范化，MATH用math-verify 0.8.0，Hotpot标准化EM/token F1，
yes/no/noanswer采用特殊规则。无法可靠解析的答案列在evaluator_review.csv，未擅自人工改分。
当前该类待复核数量：{len(uncertainty)}。没有引入LLM Judge。
初始Pilot发现单值JSON包装（如字符串内的total_pets对象）导致数值答案不能被识别。
随后统一为全部条件加入单值包装解开规则，并明确最终值prompt，所有受影响Pilot
重新生成。修正前结果/配置在diagnostics/evaluator_wrapper_pilot，不与主结果混用。
一次Runner维度读取异常在diagnostics/setup_failure，未伪装为推理失败。
第二次Pilot发现模型照抄历史id导致严格字段校验失败，并出现中间结果过早设为答案。
统一明确禁止生成Runtime id、只在原题完整完成时作答，并在有限格式重试里提供错误原因；
全部受影响样本重新生成。第二次修复前证据在diagnostics/schema_id_pilot。
此外最终数值答案可能是未化简算式（例如400-355）。统一GSM评分器支持受限有理数
四则运算/小整数幂，不从长推理文本随意挑数字；所有条件再次使用最终规则运行。
此评分修复前记录在diagnostics/arithmetic_evaluator_pilot，不与主结果混合。
初始模型曾分卡；最终主实验采用独立两卡实例，物理卡映射见execution.json和日志。

## E0

已完成{len(e0)}个方法运行，对应{len(set(row['task_id'] for row in e0))}个原始任务。
Standard单调用最多4096输出；两种stepwise最多8步×512，每次格式重试额外计费。
Stepwise关闭request字段，DARI保留；各方法temperature=0，thinking=false，统一评分。

{markdown(e0summary,['method','dataset','n','accuracy_em','f1','input_tokens','output_tokens','calls','latency_ms'])}

稳定性详情见e0/summary.csv，分别报告首次失败、最终格式终止、步数耗尽、无答案及请求重复。
所有方法使用同原始任务/来源；结构化开销及重复prefill属于被观察现象，并非严格等实际成本。

## E1与公平性

具有四个完整反事实分支的checkpoint：{len(groups)}。已完成分支：{len(e1)}。
各任务选择第一合法、未产生候选答案、Continue/SelfRepair/Help均可行的真实前缀；
一步已完成的任务记无checkpoint，不强造需求。前缀完整状态和Controller observation
保存在e1/prefixes，prefix_id为状态内容哈希，各分支深拷贝后检验前缀未变化。
SelfRepair和Help使用相同原始动态请求；Extended第二次仍使用这一原始请求，而非新请求。
Continue/SelfRepair干预一次Anchor调用；Extended两次Anchor调用；Help一次Specialist
加一次Anchor整合调用，后者属于干预。之后每分支均获最多8个新的公共Anchor步骤，
已有答案则直接Finish。Extended/Help配置干预输出均为1024，实际token并不保证相同。
格式重试如实计入。Help单阶段512、Draft192、narrow，未按最终结果切换宽度。

{markdown(e1summary,['method','dataset','n','accuracy_em','f1','input_tokens','output_tokens','calls','latency_ms'])}

E1仅比较同一批有共同checkpoint的任务，不与E0全样本直接横比。无Draft的Help保留
Abort失败分支。请求/提交/引用统计：`{json.dumps(request_stats,ensure_ascii=False)}`。
显式引用不是实际信息使用率，必要证据不足的局部上下文属于待诊断因素。

## 配对收益与Local Oracle

{markdown(paired,['dataset','baseline','n','win','tie','loss','mean_gain'])}

{markdown(oracle,['action_set','dataset','n','local_oracle','best_fixed_action','best_fixed_reward','oracle_gap','states_with_action_disagreement'])}

HotpotQA F1的连续收益单列于f1_win_tie_loss.csv，平局容差为1e-8。
前20个实际动态请求及真实下游结果在request_examples.csv，供人工核查相关性、具体性、
可执行性和重复；不使用自动LLM评委替代质量判断。

Oracle为当前有限rollout事后选择的探索性上界，受随机性和样本选择影响，不是可部署
策略或泛化保证。state_matrix.csv与Figure4保留平局；未完成分支不混入完整配对统计。

## 真实案例

{''.join(case_text)}

完整前缀、真实输入/输出及历史在e1/prefixes与traces.jsonl，可结合sample_id核对。
这些案例按观察现象选出仅用于解释，不影响预先固定的样本或评分规则。

## 稳健性和资源

已完成稳健性运行{len(robust)}条；从可用checkpoint按预定种子抽最多8个，比较S温度
0/.7与Extended内部确定性对照，每条件2次。确定性重复不代表独立统计样本。
各重复结果保留，未完成则不宣称已验证。summary见robustness/summary.csv。
当前实际执行资源汇总：`{json.dumps(actual)}`。只计已完成记录，失败/中断尝试需结合日志，
不把不完整成本说成完整成本。E1未来成本不重复包含共享前缀；logical_end_to_end_resources
单独包含前缀。前缀复用节省的是实验执行开销，不是部署加速证明。
GPU峰值/设备见execution.json及logs环境记录。异构token汇总仅诊断，未解释为GPU成本；
cost_complete=false，不用于正式Cost Head/DPO训练。

## 诊断、局限和后续建议

先区分格式/解析失败、步骤重复或预算耗尽、真实推理错误、局部证据不足及错误辅助整合。
不能把全部负例归因于未训练，也不能从少量正例断言协作有效。平均收益、状态差异和
原始案例共同决定后续是否值得训练；不要直接开始大规模SFT/DPO或增加Agent。
当前math evaluator的解析限制、少量rollout与共享服务器时延变化均限制结论。
墙钟计时含Runtime及部分审计记录开销，模型加载不计入，未独立校准prefill/decode时间。
优先检查待复核答案、重复请求和未提交Draft，再做协议训练/动作选择研究设计。

## 代码与维护

新增统一pilot_data/pilot_evaluation/pilot/pilot_analysis、配置和回归测试；Runtime复用
原状态机，仅增加关闭request的消融开关，并统一强化实际推进/及时答案prompt。
未复制Runtime、未增加Verifier或RAG、未训练任何模型。README和方法文档同步更新。
Git和最终回归检查见logs。Phase1报告、失败证据与有效权重保留。Phase2A输出及数据
受.gitignore保护，不自动push新实验。研究产物应独立归档。
'''
    (root/"phase2a_report.md").write_text(report)
    print(json.dumps({"e0_runs":len(e0),"e1_runs":len(e1),"matched_prefixes":len(groups),"robustness_runs":len(robust),"oracle":oracle},indent=2))


def figures(root,e0,e1,paired,matrix,oracle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    out=root/"figures"
    out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/(name+".png"),dpi=180)
        fig.savefig(out/(name+".pdf"))
        plt.close(fig)
    if e0:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for dataset in ("gsm8k","math500","hotpotqa"):
            group={row["method"]:row for row in e0 if row["dataset"]==dataset}
            available=[method for method in METHODS if method in group]
            axes[0].plot(available,[group[method]["accuracy_em"] for method in available],"o-",label=dataset)
            axes[1].plot(available,[group[method]["input_tokens"]+group[method]["output_tokens"] for method in available],"o-",label=dataset)
        axes[0].set_ylabel("Accuracy / EM")
        axes[1].set_ylabel("Input + output tokens (diagnostic)")
        for ax in axes:
            ax.tick_params(axis="x",rotation=20)
            ax.legend()
        save(fig,"figure1_anchor_performance_cost")
    if e1:
        fig,ax=plt.subplots(figsize=(8,4))
        width=.24
        for index,dataset in enumerate(("gsm8k","math500","hotpotqa")):
            group={row["method"]:row for row in e1 if row["dataset"]==dataset}
            ax.bar(np.arange(4)+width*(index-1),[group[method]["accuracy_em"] if method in group else np.nan for method in BRANCHES],width,label=dataset)
        ax.set_xticks(range(4),BRANCHES,rotation=15)
        ax.set_ylabel("Accuracy / EM on matched prefixes")
        ax.legend()
        save(fig,"figure2_collaboration")
    overall=[row for row in paired if row["dataset"]=="all"]
    if overall:
        fig,ax=plt.subplots(figsize=(7,3.5))
        left=np.zeros(len(overall))
        for field,color in (("win","#3B8C6E"),("tie","#B6BDC8"),("loss","#C26662")):
            values=np.array([row[field] for row in overall])
            ax.barh([row["baseline"] for row in overall],values,left=left,label=field,color=color)
            left+=values
        ax.set_xlabel("Matched states: Help relative to baseline")
        ax.legend()
        save(fig,"figure3_win_tie_loss")
    if matrix:
        fig,ax=plt.subplots(figsize=(7,max(3,len(matrix)*.16)))
        image=ax.imshow([[row[method] for method in BRANCHES] for row in matrix],vmin=0,vmax=1,aspect="auto",cmap="YlGnBu")
        ax.set_xticks(range(4),BRANCHES,rotation=15)
        ax.set_yticks(range(len(matrix)),[row["task_id"] for row in matrix],fontsize=6)
        fig.colorbar(image,ax=ax,label="Terminal reward (ties retained)")
        save(fig,"figure4_state_matrix")
    oracle=[row for row in oracle if row["action_set"]=="extended"]
    if oracle:
        fig,ax=plt.subplots(figsize=(7,3.5))
        x=np.arange(len(oracle))
        ax.bar(x-.18,[row["best_fixed_reward"] for row in oracle],.36,label="Best fixed")
        ax.bar(x+.18,[row["local_oracle"] for row in oracle],.36,label="Finite-rollout local oracle")
        ax.set_xticks(x,[row["dataset"] for row in oracle])
        ax.set_ylabel("Accuracy / EM")
        ax.legend()
        save(fig,"figure5_local_oracle")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="outputs/phase2a")
    analyze(parser.parse_args().output)
