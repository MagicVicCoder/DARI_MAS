"""Verify Phase 2B pairing, persisted private states, masks and split isolation."""
import hashlib
import json

import torch

from .phase2b_data import ROOT, normalized_hash
from .pilot import read_rows, state_id
from .schema import SpecialistRecord


def verify():
    selection=json.loads((ROOT/'selection.json').read_text())
    rows=read_rows(ROOT/'t1/e1/results.jsonl')
    assert len(rows)==5*len(selection['selected'])
    assert len({(r['prefix_id'],r['method']) for r in rows})==len(rows)
    assert all(r['status']=='completed' and r['prefix_unchanged'] for r in rows)
    methods={'continue','self_repair','self_repair_extended','help_base','help_sft'}
    for prefix_id in selection['selected']:
        assert {r['method'] for r in rows if r['prefix_id']==prefix_id}==methods
        prefix=torch.load(ROOT/'t1/e1/prefixes'/(prefix_id+'.pt'),weights_only=False,map_location='cpu')
        assert state_id(prefix)==prefix_id
        for method in ['help_base','help_sft']:
            helped=torch.load(ROOT/'t1/specialist_states'/f'{prefix_id}_{method}.pt',weights_only=False,map_location='cpu')
            assert len(helped.steps)==len(prefix.steps) and helped.help is not None
            assert helped.task==prefix.task and helped.steps==prefix.steps
    t2=read_rows(ROOT/'t2/results.jsonl')
    legal=[r for prefix_id in selection['selected'] for r in rows if r['prefix_id']==prefix_id and r['method']=='help_sft' and r['submitted']][:25]
    assert {r['prefix_id'] for r in t2}=={r['prefix_id'] for r in legal}
    assert len(t2)==len({r['prefix_id'] for r in t2})
    for row in t2:
        helped=torch.load(ROOT/'t1/specialist_states'/f'{row["prefix_id"]}_help_sft.pt',weights_only=False,map_location='cpu')
        assert state_id(helped)==row['shared_specialist_state_hash']
        assert row['delta_reward']==row['submit']['reward']-row['abort']['reward']
        trace=next(t for t in read_rows(ROOT/'t2/traces.jsonl') if t['prefix_id']==row['prefix_id'])
        assert all(e['role']=='anchor' for branch in ['submit','abort'] for e in trace[branch]['events'])
        abort_prompt=trace['abort']['events'][0]['prompt']
        assert 'temporary_help' not in json.loads(abort_prompt.split('\n',1)[1])
        assert helped.help.private_outputs[0] not in abort_prompt
        assert all(value['S']['calls']==0 for value in [
            {**{'S':{'calls':0}},**row['submit_future']['tokens_by_model']},
            {**{'S':{'calls':0}},**row['abort_future']['tokens_by_model']}])
    for method in ['base','sft']:
        local=read_rows(ROOT/'t0'/f'{method}.jsonl')
        assert len(local)==len({r['task_id'] for r in local})==40
    train=read_rows(ROOT/'train.jsonl');validation=read_rows(ROOT/'validation.jsonl');downstream=read_rows(ROOT/'t1/samples.jsonl')
    for group in [train,validation]:
        for row in group:SpecialistRecord.parse(row['response'])
    sets=[{normalized_hash(r['question']) for r in group} for group in [train,validation,downstream]]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    training=json.loads((ROOT/'training_result.json').read_text())
    assert training['frozen_parameters_unchanged'] and training['initialized_adapter_sha256'] is None
    result={'passed':True,'t0_conditions':80,'t1_conditions':len(rows),'prefixes':len(selection['selected']),
        't2_pairs':len(t2),'shared_prefix_hashes_verified':True,'same_private_state_submit_abort':True,
        'split_hash_overlap':0,'anchor_untrained':True,'specialist_frozen_backbone_version_check':True,
        'adapter_file_sha256':hashlib.sha256((ROOT/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()}
    (ROOT/'integrity_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
