"""One fixed final-test assessment of the dev-selected candidate per attempt."""
import argparse
from pathlib import Path
import mlx.core as mx
from sml_v1.common import read_json,atomic_json
from sml_v1.model import TransformerLM,TransformerConfig
from sml_v1.tokenization import Tokenizer
from .lora_checkpoint import verify,resolve
from .lora_search_common import load_baseline,gate,compact
from .text_eval import evaluate,reference_inference

def run(staging,output):
    plan=read_json(staging/'plan.json');contract=read_json(staging/'contract.json')
    eligible=(output/'best.json').exists()
    bundle=resolve(output/('best.json' if eligible else 'latest.json'));state=verify(bundle)
    if state['contract']!=contract:raise ValueError('Candidate contract mismatch')
    parent=load_baseline(staging,'test')
    meta=read_json(bundle/'inference/model.safetensors.json')
    model=TransformerLM(TransformerConfig(**meta['recipe']['model']))
    model.load_weights(str(bundle/'inference/model.safetensors'),strict=True);mx.eval(model.parameters())
    with reference_inference(model):
        result=evaluate(model,Tokenizer(staging/'tokenizer'),staging/'corpus',plan,state['step'],output/'assessment',staging/'retention',split='test')
    check=gate(result,parent,split='test')
    raw=read_json(output/'assessment/final_test'/f'eval_{state["step"]:07d}.json')
    old=[s for s in parent['samples'] if s['passed'] is not None];new=[s for s in raw['samples'] if s['passed'] is not None]
    if len(old)!=len(new) or any(a['messages']!=b['messages'] for a,b in zip(old,new)):raise ValueError('Assessment cases changed')
    paired=dict(gained=sum(not a['passed'] and b['passed'] for a,b in zip(old,new)),lost=sum(a['passed'] and not b['passed'] for a,b in zip(old,new)))
    from .lora_search_common import SHARED
    from .lora_fresh_eval import evaluate as fresh_evaluate,identity as fresh_identity
    fresh_parent=read_json(SHARED/'fresh_parent.json')
    if fresh_parent['identity']!=fresh_identity():raise ValueError('Fresh parent evaluation changed')
    fresh=fresh_evaluate(model,Tokenizer(staging/'tokenizer'),output/'assessment/fresh_test.json')
    fresh_pass=(fresh['correct']>=fresh_parent['correct'] and fresh['natural_stop']+1e-12>=fresh_parent['natural_stop'] and fresh['repetition_rate']<=fresh_parent['repetition_rate']+1e-12)
    atomic_json(output/'assessment/summary.json',dict(parent=compact(parent),candidate=result,checkpoint=str(bundle),development_eligible=eligible,
        private_pass=eligible and check['passed'] and fresh_pass,test_gate=check,paired=paired,fresh=fresh,fresh_parent=compact(fresh_parent),fresh_pass=fresh_pass,
        scope='Same historical 704 checked cases plus 40 qualitative responses. Reused evaluation, not an untouched generalization estimate.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.output)
