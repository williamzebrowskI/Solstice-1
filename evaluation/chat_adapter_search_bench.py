"""Matched full FP32 choice/IFEval/chat evaluation for a merged adapter export."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'pretraining'))
from sml_v1.common import read_json, atomic_json, file_sha256

BASELINE = ROOT/'reports/benchmarks/deep44-reliable-chat-167-fp32-20261106'


def verify_scoring():
    receipt = read_json(BASELINE/'readiness.json')
    names = ['pretraining/scripts/benchmark_best.py', 'pretraining/scripts/benchmark_ifeval.py',
             'pretraining/sml_v1/model.py', 'pretraining/sml_v1/tokenization.py',
             'pretraining/sft/data.py', 'pretraining/sft/evaluate.py',
             'pretraining/chat_benchmark/panel.py', 'pretraining/chat_benchmark/scoring.py']
    for name in names:
        if file_sha256(ROOT/name) != receipt['files'][name]:
            raise ValueError('Scoring no longer matches 167: '+name)
    return {n:receipt['files'][n] for n in names}


def chat(snapshot, output, training_data):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from sml_v1.model import TransformerLM, TransformerConfig
    from sml_v1.tokenization import Tokenizer
    from sft.evaluate import generate
    from chat_benchmark.panel import build
    from chat_benchmark.scoring import score, summarize, normalize
    cases = build()
    expected = [json.loads(l) for l in (BASELINE/'chat/cases.jsonl').read_text().splitlines()]
    if cases != expected: raise ValueError('Private chat cases differ from 167')
    prompts = {normalize(m['content']) for l in Path(training_data).read_text().splitlines()
               for m in json.loads(l)['messages'] if m['role']=='user'}
    if any(normalize(c['messages'][-1]['content']) in prompts for c in cases):
        raise ValueError('Private chat prompt overlaps training')
    meta = read_json(snapshot/'model.safetensors.json')
    model = TransformerLM(TransformerConfig(**dict(meta['recipe']['model'], attention_impl='vanilla',
                                                    ffn_impl='reference', ce_impl='reference')))
    model.load_weights(str(snapshot/'model.safetensors'), strict=True)
    mx.eval(model.parameters()); model.eval()
    if any(v.dtype != mx.float32 for _,v in tree_flatten(model.parameters())):
        raise ValueError('Chat evaluation requires FP32 weights')
    tok = Tokenizer(snapshot/'tokenizer')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'responses.jsonl'
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    if len(rows)>len(cases) or any(r['case']!=c for r,c in zip(rows,cases)):
        raise ValueError('Saved chat responses differ')
    with path.open('a') as stream:
        for case in cases[len(rows):]:
            value=generate(model,tok,case['messages'],limit=case['max_tokens'],temperature=0)
            row=dict(case=case,response=value,scores=score(case,value));rows.append(row)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            atomic_json(output/'progress.json',dict(completed=len(rows),total=len(cases)))
    result=summarize(rows)
    atomic_json(output/'results.json',result)
    atomic_json(output/'protocol.json',dict(scoring=verify_scoring(),cases=file_sha256(BASELINE/'chat/cases.jsonl'),
        precision='FP32 base masters plus FP32 trained deltas',model_sha256=file_sha256(snapshot/'model.safetensors'),
        weight_provenance=meta['inference_export'],temperature=0,repetition_penalty=1,
        attention='vanilla',ffn='reference',system_prompt=None,prompt_truncation=False))
    return result


def run(bundle, staging, output):
    verify_scoring()
    bundle,staging,output=map(Path,(bundle,staging,output))
    output.mkdir(parents=True,exist_ok=True)
    snapshot=output/'choices/checkpoint'
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        source=bundle/'inference'
        manifest=read_json(source/'manifest.json')
        for name in ('model.safetensors','model.safetensors.json'):
            if file_sha256(source/name)!=manifest['files'][name]['sha256']:
                raise ValueError('Merged export checksum mismatch')
            shutil.copyfile(source/name,snapshot/name)
        shutil.copytree(staging/'tokenizer',snapshot/'tokenizer')
        meta=read_json(snapshot/'model.safetensors.json')
        atomic_json(snapshot/'manifest.json',dict(format='sml-benchmark-snapshot-v1',source_bundle=bundle.name,
            step=meta['step'],tokens=meta['tokens'],phase='SFT',files=manifest['files']))
    cfg=read_json(ROOT/'configs/benchmark_reliable_chat_167_fp32_20261106.json')
    cfg.update(run_dir=str(bundle.parent),output_dir=str(output/'choices'),explicit_checkpoint=str(bundle/'inference'))
    config=output/'config.json';atomic_json(config,cfg)
    if not (output/'choices/datasets').exists():
        shutil.copytree(BASELINE/'choices/datasets',output/'choices/datasets')
    (output/'ifeval').mkdir(exist_ok=True)
    for name in ('inputs.json','inputs.sha256.json'):
        target=output/'ifeval'/name
        if not target.exists():shutil.copyfile(BASELINE/'ifeval'/name,target)
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false')
    commands=[('choices-prepare',[sys.executable,str(ROOT/'pretraining/scripts/benchmark_best.py'),'--config',str(config),'--prepare']),
              ('choices',[sys.executable,'-u',str(ROOT/'pretraining/scripts/benchmark_best.py'),'--config',str(config)]),
              ('ifeval',[sys.executable,'-u',str(ROOT/'pretraining/scripts/benchmark_ifeval.py'),'--checkpoint',str(snapshot),
                         '--tokenizer',str(snapshot/'tokenizer'),'--output',str(output/'ifeval'),'--max-tokens','1024'])]
    for label,command in commands:
        with (output/(label+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    choices=read_json(output/'choices/results.json');ifeval=read_json(output/'ifeval/results.json')
    if choices['status']!='complete' or not ifeval['complete'] or ifeval['evaluated']!=541:
        raise ValueError('Full benchmark did not complete')
    for name,spec in cfg['tasks'].items():
        if not choices['results'][name]['complete'] or choices['results'][name]['evaluated']!=spec['expected_examples']:
            raise ValueError('Incomplete choice task')
    old=read_json(BASELINE/'ifeval/protocol.json');new=ifeval['protocol']
    for key in ('dataset','dataset_revision','google_research_commit','evaluator_files','runner_sha256',
                'temperature','repetition_penalty','max_tokens','system_prompt','serialization',
                'prompt_truncation','context_limit','compute_precision','model_implementation','mlx_enable_tf32'):
        if old[key]!=new[key]:raise ValueError('IFEval differs from 167: '+key)
    result=dict(choices=choices['results'],ifeval=ifeval['metrics'],
                chat=chat(snapshot,output/'chat',staging/'corpus/train.jsonl'),
                complete=True,model_sha256=file_sha256(snapshot/'model.safetensors'))
    atomic_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit('bundle staging output required')
    run(*sys.argv[1:])
