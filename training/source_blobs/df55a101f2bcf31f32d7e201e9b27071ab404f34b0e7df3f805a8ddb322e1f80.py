"""Always assess the actual fresh candidate against the frozen Part 1 control."""
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint
from sml_v1.checkpoint_bundle import resolve_bundle
from sml_v1.model import TransformerConfig,TransformerLM
from sml_v1.tokenization import Tokenizer
from .grounded_corpus import verify
from .balanced_eval import evaluate
from .quality_gate import eligibility


def assess(plan):
    run=Path(plan['output']);summary=run/'final_assessment.json'
    if summary.exists():return read_json(summary)
    jobs=sorted((run/'jobs').glob('*/result.json'))
    if not jobs or read_json(jobs[-1])['status'] not in ('complete','quality-stopped'):raise ValueError('Final test is reserved until completion')
    verify(plan['prepared'],plan)
    latest=resolve_bundle(run/'latest.json');metadata=read_json(latest+'.json');contract=metadata['sft_contract']
    if contract['plan']!=fingerprint(plan):raise ValueError('Assessment plan differs')
    reference=Path(plan['reference_checkpoint']);source=Path(plan['source_checkpoint'])
    for path,expected in ((reference/'model.safetensors',contract['reference']['weights']),
        (reference/'model.safetensors.json',contract['reference']['metadata']),
        (Path(plan['reference_master_weights']),contract['reference']['masters']),
        (source/'model.safetensors',contract['source_weights']),
        (source/'model.safetensors.json',contract['source_metadata'])):
        if file_sha256(path)!=expected:raise ValueError('Assessment source changed: '+str(path))
    selected=resolve_bundle(run/'best.json') if (run/'best.json').exists() else None
    config=dict(read_json(reference/'model.safetensors.json')['recipe']['model'])
    config.update(attention_impl='vanilla',ffn_impl='reference',ce_impl='reference')
    model=TransformerLM(TransformerConfig(**config));tokenizer=Tokenizer(plan['tokenizer']);mx.set_cache_limit(int(plan.get('cache_limit_gib',8)*2**30))
    results={}
    candidates=[('part1_reference',None),('latest_candidate',latest)]
    if selected and selected!=latest:candidates.append(('selected_candidate',selected))
    for label,weights in candidates:
        if weights is None:
            model.load_weights(plan['reference_master_weights']);step=read_json(reference/'model.safetensors.json')['step']
        else:
            saved=read_json(weights+'.json')
            if saved['sft_contract']!=contract:raise ValueError('Candidate contracts differ')
            state=mx.load(weights+'.optimizer.safetensors')
            model.load_weights([(n[:-7],v) for n,v in state.items() if n.endswith('.master')]);del state
            step=saved['step']
        if any(v.dtype!=mx.float32 for _,v in tree_flatten(model.parameters())):raise ValueError('Expected original FP32 master parameters')
        mx.eval(model.parameters())
        results[label]=evaluate(model,tokenizer,plan['prepared'],dict(plan,eval_examples=320),step,run/'final_assessment'/label,plan['retention'],split='test')
    if selected==latest:results['selected_candidate']=results['latest_candidate']
    checks={label:eligibility(result,results['part1_reference'],plan) for label,result in results.items() if label!='part1_reference'}
    receipt=dict(format='opensml-fresh-curriculum-final-test-v1',compute_precision='float32',
        source_checkpoint=str(source),reference_checkpoint=str(reference),latest_checkpoint=str(Path(latest).parent),
        selected_checkpoint=str(Path(selected).parent) if selected else None,
        promoted=bool(selected and checks['selected_candidate']['eligible']),
        results=results,quality_checks=checks,
        scope='Actual latest candidate is always tested, including when no candidate qualified. Comparison uses protected private cases, a development knowledge guard and original prose retention; full public benchmarks remain separate.')
    atomic_json(summary,receipt);print(receipt,flush=True);return receipt
