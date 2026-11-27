"""Protocol adaptation experiments on existing Runtime, trainer and pilot paths."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from .backends import GenerationRequest
from .hf_backend import HuggingFaceBackend
from .identity import artifact_digest
from .phase2b_data import ROOT, normalized_hash
from .phase2b_evaluation import score, EVALUATOR_VERSION
from .pilot import Experiment, append, read_rows, metrics, trace, state_id, runtime_for, help_action, finish_or_continue
from .pilot_data import prepare
from .protocol_json import repair_json
from .schema import SpecialistRecord, Action, Kind


def setup():
    if Path('configs/phase2b.json').exists():
        raise ValueError('Phase 2B configuration already frozen; do not overwrite it')
    excluded=read_rows(Path('outputs/phase2a/samples.jsonl'))+read_rows(Path('outputs/phase2a_refinement/r3/samples.jsonl'))
    samples=prepare(output=ROOT/'t1',seed=20261126,per_dataset=12,exclude=excluded)
    base=json.loads(Path('configs/refinement_r2.json').read_text())
    base.update(protocol_version='phase2b-v1',seed=20261126,output_dir=str(ROOT/'t1'),sample_file=str(ROOT/'t1/samples.jsonl'),methods=['dari_anchor_only'])
    base['runtime']['repair_protocol_json']=True
    for name,devices in [('A',(0,1)),('S',(2,3))]:
        base['backend']['models'][name]['device_map']={'model.embed_tokens':devices[0],**{f'model.layers.{i}':devices[0] if i<18 else devices[1] for i in range(36)},'model.norm':devices[1],'lm_head':devices[1]}
    Path('configs/phase2b.json').write_text(json.dumps(base,indent=2)+'\n')
    groups={'historical':excluded,'train':read_rows(ROOT/'train.jsonl'),'validation':read_rows(ROOT/'validation.jsonl'),'downstream':samples}
    hashes={name:{normalized_hash(row['question']) for row in rows} for name,rows in groups.items()}
    overlaps={left+'/'+right:len(hashes[left]&hashes[right]) for i,left in enumerate(groups) for right in list(groups)[i+1:]}
    assert not any(overlaps.values())
    (ROOT/'leakage_audit.json').write_text(json.dumps({'normalized_question_overlaps':overlaps,'counts':{k:len(v) for k,v in groups.items()},'limitation':'Constructed train/validation share10 task families with different instances. Not a heldout-template test. No claim of exhaustive semantic paraphrase detection; downstream benchmark sources differ from constructed supervision.'},indent=2))


def local_assessment(row, text, backend):
    fixed,changes=repair_json(text)
    try:
        obj=SpecialistRecord.parse(fixed)
    except ValueError as error:
        return {'format_valid':False,'delivered':False,'parse_error':str(error),'repair_changes':changes}
    expected=row['expected_local_result']
    def contains(value):
        if row['analysis_category'] in [4,5,6]:
            return expected.lower() in value.lower()
        if row['analysis_category']==8:
            return bool(re.search(r'underdetermined|unknown|not (?:enough|provided|specified)|missing|cannot (?:determine|calculate)',value,re.I))
        return bool(re.search(r'(?<![\w.])'+re.escape(expected)+r'(?!\w|\.\d)',value))
    work_ok,draft_ok=contains(obj.work),contains(obj.draft.text)
    tokens=backend.count_text_tokens('S',obj.draft.text)
    return {'format_valid':True,'delivered':tokens<=384,'work_tokens':backend.count_text_tokens('S',obj.work),
        'draft_tokens':tokens,'local_result_match':draft_ok,'work_result_match':work_ok,
        'work_draft_result_match':work_ok and draft_ok,'metric_scope':'bounded expected-result occurrence, not proof of semantic correctness or advisory-only absence',
        'repair_changes':changes,'work':obj.work,'draft':obj.draft.text}


def t0():
    config=json.loads(Path('configs/phase2b.json').read_text())
    spec=deepcopy(config['backend']['models']['S'])
    spec['device_map']={key:(value-2) for key,value in spec['device_map'].items()}
    backend=HuggingFaceBackend.from_specs({'S':spec})
    rows=read_rows(ROOT/'validation.jsonl')
    for method in ['base','sft']:
        if method=='sft':
            from peft import PeftModel
            handle=backend.handles['S']
            handle.model=PeftModel.from_pretrained(handle.model,ROOT/'adapter',is_trainable=False)
            handle.model.eval();handle.model.requires_grad_(False)
        done={r['task_id'] for r in read_rows(ROOT/'t0'/f'{method}.jsonl')}
        for index,row in enumerate(rows):
            if row['task_id'] in done:continue
            print('T0',method,index,row['task_id'],flush=True)
            request=GenerationRequest('S','specialist',row['prompt'],512,0,seed=20261126+index)
            generated=backend.generate(request)
            append(ROOT/'t0'/f'{method}.jsonl',{'task_id':row['task_id'],'method':method,'prompt':row['prompt'],'output':generated.text,
                'usage':asdict(generated.usage),'latency_ms':generated.latency_ms,'seed':request.seed,
                'expected_local_result':row['expected_local_result'],**local_assessment(row,generated.text,backend)})
    (ROOT/'t0/manifest.json').write_text(json.dumps({'config':config,'validation_sha256':hashlib.sha256((ROOT/'validation.jsonl').read_bytes()).hexdigest(),'adapter_sha256':artifact_digest(ROOT/'adapter'),'evaluator':EVALUATOR_VERSION,'repair':'protocol_json.repair_json; same for base/sft','budget':512,'draft_limit':384,'temperature':spec['temperature'],'cost_complete':False},indent=2))


def downstream():
    experiment=Experiment('configs/phase2b.json')
    experiment.e0(experiment.samples)
    from peft import PeftModel
    handle=experiment.backend.handles['S']
    handle.model=PeftModel.from_pretrained(handle.model,ROOT/'adapter',is_trainable=False)
    handle.model.eval();handle.model.requires_grad_(False)
    index=[r for r in read_rows(experiment.root/'e0/results.jsonl') if r.get('prefix_id')]
    selection_path=ROOT/'selection.json'
    if selection_path.exists():
        selection=json.loads(selection_path.read_text())
    else:
        import random
        rng=random.Random(20261126);rng.shuffle(index)
        selected=index[:30]
        selection={'seed':20261126,'available':len(index),'selected':[r['prefix_id'] for r in selected],
            'pilot':[r['prefix_id'] for r in selected[:12]],'t2_rule':'first25 selected SFT states with legal draft, before observing Submit/Abort outcomes'}
        selection_path.write_text(json.dumps(selection,indent=2))
    samples={r['task_id']:r for r in experiment.samples}
    lookup={r['prefix_id']:r for r in index}
    done={(r['prefix_id'],r['method']) for r in read_rows(ROOT/'t1/e1/results.jsonl')}
    t2done={r['prefix_id'] for r in read_rows(ROOT/'t2/results.jsonl')}
    for prefix_id in selection['selected']:
        prefix=torch.load(ROOT/'t1/e1/prefixes'/(prefix_id+'.pt'),weights_only=False,map_location='cpu')
        assert state_id(prefix)==prefix_id
        row=samples[lookup[prefix_id]['task_id']]
        for method in ['continue','self_repair','self_repair_extended','help_base','help_sft']:
            if (prefix_id,method) in done:continue
            print('T1',row['task_id'],method,flush=True)
            started=time.perf_counter()
            base={'task_id':row['task_id'],'dataset':row['dataset'],'prefix_id':prefix_id,'method':method,'seed':prefix.seed}
            context=handle.model.disable_adapter() if method!='help_sft' else nullcontext()
            with context:
                if not method.startswith('help_'):
                    state,meta=experiment.branch(prefix,method,prefix.seed)
                else:
                    runtime=runtime_for(experiment.runtime,max_anchor_steps=len(prefix.steps)+2+experiment.config['continuation_steps'])
                    helped=runtime.step(prefix,help_action(runtime,prefix,experiment.config['context_width']))
                    submitted=Action(Kind.SUBMIT) in runtime.actions(helped)
                    helped_path=ROOT/'t1/specialist_states'/f'{prefix_id}_{method}.pt'
                    helped_path.parent.mkdir(exist_ok=True)
                    torch.save(helped,helped_path)
                    state=runtime.step(helped,Action(Kind.SUBMIT if submitted else Kind.ABORT))
                    runtime=runtime_for(runtime,max_anchor_steps=len(state.steps)+experiment.config['continuation_steps'])
                    state=runtime.run_from(state,finish_or_continue)
                    meta={'submitted':submitted,'prefix_unchanged':state_id(prefix)==prefix_id,'specialist_state':str(helped_path)}
                    if method=='help_sft' and submitted and prefix_id not in t2done and len(t2done)<25:
                        abort_runtime=runtime_for(experiment.runtime,max_anchor_steps=len(prefix.steps)+2+experiment.config['continuation_steps'])
                        abort=abort_runtime.step(helped,Action(Kind.ABORT))
                        abort_runtime=runtime_for(abort_runtime,max_anchor_steps=len(abort.steps)+experiment.config['continuation_steps'])
                        abort=abort_runtime.run_from(abort,finish_or_continue)
                        scored=score(row['dataset'],state.candidate,row['gold']);abort_scored=score(row['dataset'],abort.candidate,row['gold'])
                        append(ROOT/'t2/results.jsonl',{**base,'submit':scored,'abort':abort_scored,'delta_reward':scored['reward']-abort_scored['reward'],
                            'submit_future':metrics(state,helped),'abort_future':metrics(abort,helped),'shared_specialist_state_hash':state_id(helped),
                            'submit_answer':state.candidate,'abort_answer':abort.candidate})
                        append(ROOT/'t2/traces.jsonl',{**base,'submit':trace(state,helped),'abort':trace(abort,helped)})
                        t2done.add(prefix_id)
            append(ROOT/'t1/e1/traces.jsonl',{**base,**trace(state,prefix)})
            append(ROOT/'t1/e1/results.jsonl',{**base,'status':'completed',**meta,**metrics(state,prefix),
                **score(row['dataset'],state.candidate,row['gold']),'answer':state.candidate,'latency_ms':(time.perf_counter()-started)*1000})
    (ROOT/'t1/adapter_manifest.json').write_text(json.dumps({'adapter_sha256':artifact_digest(ROOT/'adapter'),'anchor':'frozen original no adapter','evaluator':EVALUATOR_VERSION,'methods':5,'cost_complete':False},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['setup','t0','downstream']);args=parser.parse_args()
    {'setup':setup,'t0':t0,'downstream':downstream}[args.stage]()


if __name__=='__main__':main()
