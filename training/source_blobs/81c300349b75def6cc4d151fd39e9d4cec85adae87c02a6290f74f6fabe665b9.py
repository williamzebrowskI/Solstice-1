"""A separately worded and populated 256-case transfer check."""
import json
from collections import defaultdict
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v1.common import atomic_json,read_json,file_sha256
from sml_v1.model import TransformerLM,TransformerConfig
from sml_v1.tokenization import Tokenizer
from .lora_search_common import SHARED,REPORT,PARENT,ASSETS,data_receipt
from .lora_checkpoint import verify
from .text_eval import reference_inference
from .evaluate import generate
from .text_checks import answer_pass
from .quality_gate import looping

def identity():
    return dict(parent=file_sha256(PARENT/'manifest.json'),data=file_sha256(SHARED/'fresh_data/fresh_test.jsonl'),
                evaluator=file_sha256(SHARED/'runtime/sft/lora_fresh_eval.py'))

def evaluate(model,tokenizer,output):
    if any(v.dtype!=mx.float32 for _,v in tree_flatten(model.parameters())):raise ValueError('Fresh evaluation requires FP32')
    model.eval();rows=[json.loads(l) for l in (SHARED/'fresh_data/fresh_test.jsonl').open()];samples=[];domains=defaultdict(list)
    try:
        with reference_inference(model):
            for i,row in enumerate(rows,1):
                value=generate(model,tokenizer,row['messages'][:-1],limit=160)
                natural=value['stop_reason'] in ('turn_end','document_end');repeat=looping(value['text']);matched=answer_pass(value['text'],row['verification'])
                passed=matched and natural and not repeat and bool(value['text'].strip());domains[row['component']].append(passed)
                samples.append(dict(group=row['group'],component=row['component'],reference=row['messages'][-1]['content'],messages=row['messages'][:-1],**value,matched=matched,passed=passed,repeated=repeat))
                if i%32==0:print(f'[fresh-eval] {i}/{len(rows)}',flush=True)
        value=dict(correct=sum(s['passed'] for s in samples),total=len(samples),counts={k:sum(v) for k,v in domains.items()},
                   natural_stop=sum(s['stop_reason'] in ('turn_end','document_end') for s in samples)/len(samples),repetition_rate=sum(s['repeated'] for s in samples)/len(samples),samples=samples,identity=identity())
        atomic_json(output,value)
        return {k:v for k,v in value.items() if k!='samples'}
    finally:model.train()

def parent():
    data_receipt();verify(PARENT)
    model=TransformerLM(TransformerConfig(**read_json(PARENT/'inference/model.safetensors.json')['recipe']['model']))
    model.load_weights(str(PARENT/'inference/model.safetensors'),strict=True);mx.eval(model.parameters())
    result=evaluate(model,Tokenizer(ASSETS/'tokenizer'),SHARED/'fresh_parent.json')
    atomic_json(REPORT/'fresh_parent_summary.json',result)

if __name__=='__main__':parent()
