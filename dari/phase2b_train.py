"""Reproducible Phase 2B LoRA training entry point."""

def main():
    import json,time,torch
    from pathlib import Path
    from dari.train_specialist import train_from_file
    root=Path('outputs/phase2b')
    if (root/'adapter').exists():
        raise ValueError('Refuse to overwrite a trained adapter; choose a new phase output before training')
    spec={'name_or_path':'/data/models/Qwen3-8B','dtype':'bfloat16','chat_template_kwargs':{'enable_thinking':False},'device_map':{'model.embed_tokens':0,**{f'model.layers.{i}':0 if i<18 else 1 for i in range(36)},'model.norm':1,'lm_head':1}}
    settings={'steps':50,'batch_size':1,'accumulation':8,'lr':2e-5,'max_length':2048,'seed':20261126,'epoch_order':True,'progress_path':str(root/'training_progress.jsonl')}
    (root/'training_config.json').write_text(json.dumps({'model':spec,'settings':settings,'rank':16,'alpha':32,'dropout':.05,'epochs':2,'cost_complete':False},indent=2))
    started=time.perf_counter()
    result=train_from_file(root/'train.jsonl',root/'adapter',spec,lora_rank=16,lora_alpha=32,lora_dropout=.05,validation_data=root/'validation.jsonl',**settings)
    result['total_wall_seconds']=time.perf_counter()-started
    result['peak_memory_bytes']={str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}
    (root/'training_result.json').write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
