"""Actual serial execution-time accounting and held-out local cost profiling."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .backends import GenerationRequest
from .configuration import build_runtime
from .costs import ModelProfile
from .pilot import append, read_rows

ROOT=Path('outputs/phase3a')
UNIT='serial_execution_ms'


class MeasuredBackend:
    """Model-call elapsed time, not summed GPU resource time, excluding load."""
    def __init__(self, backend):
        self.backend=backend;self.synthetic=backend.synthetic;self.calls=[]

    @property
    def deployment_identity(self):return {**self.backend.deployment_identity,'measurement_unit':UNIT,'measurement':'synchronized call wall time; model loading excluded'}

    def count_tokens(self,*args):return self.backend.count_tokens(*args)
    def count_text_tokens(self,*args):return self.backend.count_text_tokens(*args)

    def generate(self,request):
        raw=self.backend.backend if hasattr(self.backend,'backend') else self.backend
        model=raw.handles[request.model_id].model
        devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        for index in devices:torch.cuda.synchronize(index)
        started=time.perf_counter()
        generation=self.backend.generate(request)
        for index in devices:torch.cuda.synchronize(index)
        elapsed=(time.perf_counter()-started)*1000
        self.calls.append({'role':request.role,'model_id':request.model_id,'input_tokens':generation.usage.input_tokens,
            'output_tokens':generation.usage.output_tokens,'total_ms':elapsed,
            'peak_allocated_bytes':{str(i):torch.cuda.max_memory_allocated(i) for i in devices},**raw.last_forward_timing})
        return replace(generation,usage=replace(generation.usage,observed_cost=elapsed,observed_unit=UNIT))


def basis(inputs,outputs):
    return [inputs,outputs,inputs*inputs,inputs*outputs+outputs*(outputs-1)/2,1]


def profile():
    from scipy.optimize import nnls
    config=json.loads(Path('configs/phase3a.json').read_text())
    runtime=build_runtime(config);raw=runtime.backend.backend
    backend=MeasuredBackend(runtime.backend)
    raw.profile_timing=True
    plan=[(256,64),(1024,64),(2048,128),(4096,128),(1024,256),(2048,256),(512,96),(3072,192),(1024,64)]
    path=ROOT/'c0/profiling.jsonl'
    done={(r['model_id'],r['index']) for r in read_rows(path)}
    for model_id,handle in raw.handles.items():
        request=GenerationRequest(model_id,'anchor' if model_id=='A' else 'specialist','Continue listing consecutive integers from1 for as long as possible.',16,0,seed=20261127)
        backend.generate(request)
        for index,(target,cap) in enumerate(plan):
            if (model_id,index) in done:continue
            prefix='Keep copying and extending the following sequence; do not conclude or explain. '
            tokenizer=handle.codec.tokenizer
            tokens=tokenizer.encode(' context datum '*target,add_special_tokens=False)[:max(1,target-60)]
            prompt=tokenizer.decode(tokens)+ '\n'+prefix+'1,2,3,4,5,6,7,8,9,10,'
            request=GenerationRequest(model_id,'anchor' if model_id=='A' else 'specialist',prompt,cap,index,seed=20261127)
            print('PROFILE',model_id,index,target,cap,flush=True)
            backend.generate(request)
            append(path,{**backend.calls[-1],'index':index,'target_input_length':target,'output_cap':cap,
                'split':'validation' if index in [6,7,8] else 'fit','dtype':'bfloat16','batch_size':1,
                'gpu_names':{str(i):torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())},
                'deployment':raw.deployment_identity,'unit':UNIT})
    rows=read_rows(path);profiles=[];errors={}
    for model_id in ['A','S']:
        train=[r for r in rows if r['model_id']==model_id and r['split']=='fit']
        valid=[r for r in rows if r['model_id']==model_id and r['split']=='validation']
        x=np.array([basis(r['input_tokens'],r['output_tokens']) for r in train],dtype=float)
        scale=np.maximum(np.linalg.norm(x,axis=0),1)
        coefficients=nnls(x/scale,np.array([r['total_ms'] for r in train]))[0]/scale
        profile=ModelProfile(model_id,UNIT,uncached_input=float(coefficients[0]),output=float(coefficients[1]),
            prefill_square=float(coefficients[2]),decode_context=float(coefficients[3]),call_overhead=float(coefficients[4]),
            provenance='Phase3A synchronized measured serial call wall ms; independent profiling fit, shared4090 deployment; not GPU resource-ms')
        errors[model_id]=[]
        for row in valid:
            prediction=float(np.dot(basis(row['input_tokens'],row['output_tokens']),coefficients))
            errors[model_id].append({'input':row['input_tokens'],'output':row['output_tokens'],'actual_ms':row['total_ms'],'predicted_ms':prediction,
                'absolute_error_ms':abs(prediction-row['total_ms']),'relative_error':abs(prediction-row['total_ms'])/row['total_ms']})
        profiles.append(asdict(profile))
    relative=np.mean([r['relative_error'] for values in errors.values() for r in values])
    result={'profiles':profiles,'validation':errors,'mean_validation_relative_error':float(relative),
        'calibration_confirmed':bool(relative<.5),'unit':UNIT,'scope':'preliminary model execution-time estimator for the declared placement, not exclusive GPU resource time or monetary cost',
        'reference_cost':10000.,'reference_basis':'fixed10seconds chosen before labels, no eval outcomes used',
        'measurement_note':'prefill and decode-forward synchronized; total includes token selection, pooling, tokenizer and Python call overhead. Shared-load effects remain; independent validation/repeated point reports drift.'}
    (ROOT/'c0/calibration.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':profile()
