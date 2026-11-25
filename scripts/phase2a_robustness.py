"""Repeat Phase 2A checkpoints with an explicitly recorded memory-safe placement."""
import hashlib
import json
import os
from pathlib import Path
from copy import deepcopy
from dari.pilot import Experiment,read_rows
from dari.configuration import build_runtime
from dari.identity import fingerprint
root=Path('outputs/phase2a')
exp=Experiment.__new__(Experiment)
exp.config=json.loads(Path('configs/phase2a_qwen3_8b.json').read_text())
exp.root=root
exp.samples=read_rows(root/'samples.jsonl')
manifest=json.loads((root/'manifest.json').read_text())
identity={'config':exp.config,'samples':fingerprint(exp.samples),
          'code':{name:hashlib.sha256(Path('dari',name).read_bytes()).hexdigest() for name in manifest['code']},
          'evaluator':manifest['evaluator']}
exp.identity=fingerprint(identity)
if exp.identity!=manifest['experiment_id']:
    raise ValueError('Main experiment code/config/data changed; cannot resume robustness under its identity')
placement=deepcopy(exp.config)
placement['backend']['models']['A']['device']='cuda:0'
s=placement['backend']['models']['S'];s.pop('device')
s['device_map']={'model.embed_tokens':1,**{f'model.layers.{i}':1 if i<18 else 2 for i in range(36)},'model.norm':2,'lm_head':2}
(root/'robustness').mkdir(exist_ok=True)
placement_record={'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'models':placement['backend']['models'],'reason':'Current free GPU memory requires Specialist sharding; same frozen weights, prompts, sampling and budgets. Device placement differs from main experiment; latency is not directly comparable.'}
placement_path=root/'robustness/placement.json'
if placement_path.exists() and json.loads(placement_path.read_text())!=placement_record:
    raise ValueError('Robustness placement changed; do not mix conditions on resume')
placement_path.write_text(json.dumps(placement_record,indent=2))
exp.runtime=build_runtime(placement)
exp.backend=exp.runtime.backend.backend
exp.dimensions=placement['observation']
exp.e1(exp.samples,robustness=True)
import torch
(root/'robustness/execution.json').write_text(json.dumps({'completed':True,'peak_allocated_bytes':{str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}},indent=2))
