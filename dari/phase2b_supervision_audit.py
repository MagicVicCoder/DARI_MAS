"""Reproducible Phase 2B independent supervision target audit."""

def main():
    import json,ast
    from pathlib import Path
    from fractions import Fraction
    from dari.pilot import read_rows
    root=Path('outputs/phase2b')
    rows=read_rows(root/'train.jsonl')+read_rows(root/'validation.jsonl')
    # Independent arithmetic checks from the constructed question text, never model-generated candidate labels.
    import re
    out=[]
    for r in rows:
     q=r['question']; nums=list(map(int,re.findall(r'\d+',q))); k=r['analysis_category']; expected=r['expected_local_result']
     if k==0: actual=str(nums[0]*nums[1]-nums[2])
     elif k==1: actual=str(Fraction(nums[0]*nums[1]*nums[2],7))
     elif k==2: actual=str(Fraction(nums[2]-nums[1],nums[0]))
     elif k==3: actual=str(nums[0]*nums[1]-nums[2]**2)
     elif k==4:actual=f'{nums[0]}*x+{nums[1]}'
     elif k==5:actual='C-'+str(nums[-1])
     elif k==6:actual='B-'+str(nums[-1])
     elif k==7:actual=str(Fraction(nums[0],nums[0]+nums[1]))
     elif k==8:actual='underdetermined'
     else:actual=str(Fraction(nums[0]**2-nums[1]**2,4))
     assert actual==expected,(r['task_id'],actual,expected)
     out.append({'task_id':r['task_id'],'independent_recomputed_result':actual,'expected':expected,'pass':True,'scope':'construction target only; response derivation remains inspectable, not an LLM or invented human review'})
    (root/'deterministic_supervision_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))


if __name__ == "__main__":
    main()
