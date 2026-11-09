"""Compare the pretrained parent and selected candidate on untouched test cases."""
import argparse
from pathlib import Path
import mlx.core as mx
from sml_v1.common import read_json, atomic_json, file_sha256
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.tokenization import Tokenizer
from .text_corpus import verify
from .text_eval import evaluate,reference_inference


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=read_json(a.input/'plan.json');contract=read_json(a.input/'contract.json');verify(a.input/'corpus',plan)
    tokenizer=Tokenizer(a.input/'tokenizer');parent=a.input/'source'
    if file_sha256(parent/'masters/model.safetensors')!=contract['source_masters']:raise ValueError('Parent masters changed')
    candidates=[('parent',parent,0)]
    if plan.get('comparison_checkpoint'):
        comparison=a.input/'comparison'
        for name,digest in contract['comparison_files'].items():
            if file_sha256(comparison/name)!=digest:raise ValueError('Comparison checkpoint changed')
        candidates.append(('reference_417',comparison,417))
    pointer=read_json(a.output/'best.json');name=pointer['bundle']
    if Path(name).name!=name or not name.startswith('step_'):raise ValueError('Invalid selected checkpoint')
    selected=a.output/name;meta=read_json(selected/'model.safetensors.json')
    if meta['sft_contract']!=contract:raise ValueError('Selected candidate contract changed')
    if meta['step']>0:candidates.append(('selected',selected,meta['step']))
    latest_pointer=read_json(a.output/'latest.json');latest_name=latest_pointer['bundle']
    if Path(latest_name).name!=latest_name or not latest_name.startswith('step_'):raise ValueError('Invalid latest checkpoint')
    latest=a.output/latest_name
    latest_meta=read_json(latest/'model.safetensors.json')
    if latest_meta['sft_contract']!=contract:raise ValueError('Latest candidate contract changed')
    if latest_meta['step']>0 and latest!=selected:
        candidates.append(('latest_unselected',latest,latest_meta['step']))
    results={}
    for label,path,step in candidates:
        metadata=read_json(path/'model.safetensors.json');model=TransformerLM(TransformerConfig(**metadata['recipe']['model']))
        if label=='parent':weights=mx.load(str(path/'masters/model.safetensors'))
        elif label=='reference_417':weights=mx.load(str(path/'model.safetensors'))
        else:
            manifest=read_json(path/'manifest.json');name='model.safetensors.optimizer.safetensors'
            if file_sha256(path/name)!=manifest['files'][name]['sha256']:raise ValueError('Candidate state checksum changed')
            state=mx.load(str(path/name));weights={n[:-7]:v for n,v in state.items() if n.endswith('.master')}
            del state
        model.load_weights(list(weights.items()),strict=True);mx.eval(model.parameters());del weights
        with reference_inference(model):results[label]=evaluate(model,tokenizer,a.input/'corpus',plan,step,a.output/'assessment'/label,a.input/'retention',split='test')
        del model;mx.clear_cache()
    atomic_json(a.output/'assessment/summary.json',dict(results=results,selected=str(selected),
        status='candidate-needs-benchmark-review' if meta['step']>0 else 'no-eligible-improvement',
        note='Untouched grouped test set; does not replace the full ARC/PIQA/IFEval suite.'))
    print('[assessment] '+str(a.output/'assessment/summary.json'),flush=True)


if __name__=='__main__':main()
