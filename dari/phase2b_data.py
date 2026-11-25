"""Seeded constructed local tasks with inspectable deterministic supervision."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re

from .demo import build_demo
from .pilot import read_rows
from .runtime import HelpSession
from .schema import Task, Step, AnchorRecord, HelpRequest, Action, Kind
from .specialist_data import write_records

ROOT = Path("outputs/phase2b")


def normalized_hash(text):
    return hashlib.sha256(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


def construct(index, rng):
    a,b,c = rng.sample(range(7,180),3)
    kind = index%10
    if kind == 0:
        question=f'A workshop prepares {a} bundles of {b} components and uses {c} components. How many components remain?'
        request='Compute the remaining component count exactly.'
        value=a*b-c; work=f'The initial stock is {a}*{b}={a*b}. Subtract the used {c}: {a*b}-{c}={value}.'
    elif kind == 1:
        question=f'A service makes {a} visits per day for {b} days and charges {c}/7 dollars per visit. Find exact revenue.'
        request='Derive exact total revenue without rounding the price per visit.'
        value=Fraction(a*b*c,7); work=f'Total visits={a}*{b}={a*b}. Multiply the exact rational price: {a*b}*{c}/7={value}. No decimal approximation is needed.'
    elif kind == 2:
        question=f'Solve {a}*x+{b}={a*c+b}.'
        request='Isolate x and verify the result in the original equation.'
        value=c; work=f'Subtract {b} on both sides: {a}*x={a*c}. Divide by {a}: x={c}. Substitution gives {a}*{c}+{b}={a*c+b}, as required.'
    elif kind == 3:
        question=f'A rectangle measures {a} by {b}. A square of side {c%6+1} is removed. Find the remaining area.'
        request='Calculate the two areas and their difference.'
        side=c%6+1;value=a*b-side**2;work=f'Rectangle area={a}*{b}={a*b}. Removed square area={side}^2={side**2}. Remaining area={a*b}-{side**2}={value}.'
    elif kind == 4:
        question=f'For nonzero real x, simplify ({a}*x*x+{b}*x)/x.'
        request='Give an alternative factorization derivation and preserve the domain restriction.'
        value=f'{a}*x+{b}';work=f'Factor the numerator as x*({a}*x+{b}). Since x is nonzero, cancel x, obtaining {value}. The original domain excludes x=0.'
    elif kind == 5:
        question=f'An archive states that researcher Mira-{a} works at institute I-{b}. Another record says I-{b} is in city C-{c}. Which city contains the workplace?'
        request='Compose the two evidence relations and identify the city.'
        value=f'C-{c}';work=f'The first record links Mira-{a} to I-{b}. The second links the same I-{b} to C-{c}. Therefore the workplace is in {value}.'
    elif kind == 6:
        question=f'An old register lists the office of unit U-{a} as city A-{b}. A later signed correction explicitly replaces that listing with city B-{c}. What is the current office city?'
        request='Resolve the conflict using the stated chronology and correction scope.'
        value=f'B-{c}';work=f'The later record explicitly supersedes the old office listing for U-{a}. Thus A-{b} is historical, and the current office is {value}. This conclusion uses the specified correction, not a general preference for newer claims.'
    elif kind == 7:
        question=f'A box contains {a} red and {b} blue counters. One is sampled uniformly. Find the exact red probability.'
        request='Check the denominator and compute the probability as a reduced rational number.'
        value=Fraction(a,a+b);work=f'There are {a}+{b}={a+b} counters, not {b}. The favorable count is {a}, so probability={a}/{a+b}={value}.'
    elif kind == 8:
        question=f'A shipment has {a} crates, each holding {b} bottles. The record omits the price of each bottle. What is the sales revenue?'
        request='Determine what is identifiable and what additional information revenue requires.'
        value='underdetermined';work=f'The bottle count is {a}*{b}={a*b}. Let the unreported price be p; revenue is {a*b}*p. Different prices give different revenues, so no unique numeric revenue follows.'
    else:
        question=f'Two integers have sum {a+b} and difference {abs(a-b)}. Find their product.'
        request='Derive the product using sum and difference rather than guessing the integers.'
        value=a*b;work=f'For sum S and difference D, 4uv=S^2-D^2. Here S={a+b}, D={abs(a-b)}; product=({a+b}^2-{abs(a-b)}^2)/4={value}.'
    if kind == 8:
        draft=f'The revenue is underdetermined. There are {a*b} bottles; with unknown unit price p it is {a*b}*p. The unit selling price is required.'
    else:
        draft=f'The requested local result is {value}. '+work
    return question,request,str(value),work,draft,kind


def prepare():
    rng=random.Random(20261126)
    runtime,_=build_demo()
    excluded=read_rows(Path('outputs/phase2a/samples.jsonl'))+read_rows(Path('outputs/phase2a_refinement/r3/samples.jsonl'))
    excluded_hashes={normalized_hash(r['question']) for r in excluded}
    rows=[];hashes=set()
    for index in range(240):
        question,request,value,work,draft,kind=construct(index,rng)
        digest=normalized_hash(question)
        if digest in hashes|excluded_hashes:
            raise ValueError('Duplicate question')
        hashes.add(digest)
        task=Task(f'phase2b:constructed:{index}',question,{})
        from .runtime import EpisodeState
        from .costs import CostLedger
        state=EpisodeState(task,CostLedger(runtime.profiles),runtime.config.budget)
        state.steps=[Step('s1',AnchorRecord('The requested local subproblem remains unresolved.',[],request,None))]
        state.help_request=HelpRequest(request,'s1',[])
        action=Action(Kind.HELP,target='s1',width='narrow',quota='short')
        state.help=HelpSession('h1',action)
        prompt=runtime._specialist_request(state).prompt
        row={'input_id':task.task_id,'task_id':task.task_id,'prompt':prompt,'response':json.dumps({'work':work,'draft':draft},ensure_ascii=False),
            'synthetic':True,'source':'explicitly constructed local task; deterministic reference derivation, not model generation',
            'question':question,'normalized_question_sha256':digest,'local_request':request,'expected_local_result':value,
            'analysis_category':kind,'verification':'construction algebra/arithmetic or explicit evidence relation; supervision fully inspectable',
            'multistage':False,'seed':20261126}
        rows.append(row)
    rng.shuffle(rows)
    train,valid=rows[:200],rows[200:]
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,group in [('train',train),('validation',valid)]:
        path=ROOT/(name+'.jsonl')
        if path.exists():
            raise ValueError('Refuse to overwrite fixed data')
        write_records(path,group)
    (ROOT/'data_manifest.json').write_text(json.dumps({'seed':20261126,'train':200,'validation':40,'excluded_historical_tasks':84,
        'exact_id_overlap':0,'normalized_question_overlap':0,'source':'synthetic constructed with distinct numeric/evidence instances',
        'limitation':'Ten construction families; within-family generalization, not proof of broad open-ended reasoning. Single-stage only. No fabricated human review or unverified model labels.',
        'sha256':{name:hashlib.sha256((ROOT/(name+'.jsonl')).read_bytes()).hexdigest() for name in ['train','validation']}},indent=2))


if __name__=='__main__':
    prepare()
