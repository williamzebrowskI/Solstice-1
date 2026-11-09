"""Final test of the selected candidate versus its exact source checkpoint."""
import argparse
from pathlib import Path
import mlx.core as mx
from sml_v1.common import atomic_json,read_json,fingerprint,file_sha256
from sml_v1.checkpoint_bundle import resolve_bundle
from sml_v1.model import TransformerConfig,TransformerLM
from sml_v1.tokenization import Tokenizer
from .grounded_corpus import verify
from .grounded_eval import evaluate
from .quality_gate import eligibility


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    plan=read_json(a.config);run=Path(plan['output']);summary=run/'final_assessment.json'
    if plan['format']=='opensml-deep44-curriculum-sft-v4':
        from .curriculum_assess import assess
        assess(plan);return
    evaluate_run=evaluate
    stage2=plan['format']=='opensml-deep44-balanced-sft-v3'
    if stage2:
        from .balanced_eval import evaluate as evaluate_run
    if summary.exists():
        print(read_json(summary));return
    jobs=sorted((run/'jobs').glob('*/result.json'))
    if not jobs or read_json(jobs[-1])['status'] not in ('complete','quality-stopped'):
        raise ValueError('Final test is reserved until training finishes')
    verify(plan['prepared'],plan)
    selected=resolve_bundle(run/'best.json');metadata=read_json(selected+'.json')
    if metadata['sft_contract']['plan']!=fingerprint(plan):raise ValueError('Selected SFT recipe differs')
    source=Path(plan['source_checkpoint']);parent=read_json(source/'model.safetensors.json')
    if file_sha256(source/'model.safetensors')!=metadata['sft_contract']['source_weights']:
        raise ValueError('Source baseline changed')
    tokenizer=Tokenizer(plan['tokenizer']);cfg=dict(parent['recipe']['model'])
    if stage2:cfg.update(attention_impl='vanilla',ffn_impl='reference',ce_impl='reference')
    model=TransformerLM(TransformerConfig(**cfg))
    mx.set_cache_limit(int(plan.get('cache_limit_gib',8)*2**30))
    results={}
    parent_label='parent_sft' if stage2 else 'pretrained'
    for label,weights,step in ((parent_label,str(source/'model.safetensors'),0),('selected_sft',selected,metadata['step'])):
        if stage2:
            if label==parent_label:
                masters=Path(plan['source_master_weights'])
                if file_sha256(masters)!=metadata['sft_contract']['source_masters']:raise ValueError('Source FP32 masters changed')
                model.load_weights(str(masters))
            else:
                state=mx.load(weights+'.optimizer.safetensors')
                model.load_weights([(name[:-7],value) for name,value in state.items() if name.endswith('.master')])
                del state
            from mlx.utils import tree_flatten
            if any(v.dtype!=mx.float32 for _,v in tree_flatten(model.parameters())):raise ValueError('Expected FP32 final assessment')
        else:model.load_weights(weights)
        mx.eval(model.parameters())
        test_plan=dict(plan,generated_eval_per_component=32,eval_examples=320)
        results[label]=evaluate_run(model,tokenizer,Path(plan['prepared']),test_plan,step,run/'final_assessment'/label,Path(plan['retention']),split='test')
    checks=eligibility(results['selected_sft'],results[parent_label],plan) if stage2 else None
    receipt=dict(format='opensml-balanced-sft-final-test-v3' if stage2 else 'opensml-grounded-sft-final-test-v1',selected_checkpoint=str(Path(selected).parent),
        source_checkpoint=str(source),results=results,promoted=bool(metadata['step']>0 and (checks is None or checks['eligible'])),
        compute_precision='float32' if stage2 else 'bfloat16',
        dev_quality_baseline=metadata.get('selection_baseline'),
        final_quality_checks=checks,
        delta_generated_macro_pass=results['selected_sft']['generated_macro_pass']-results[parent_label]['generated_macro_pass'],
        delta_retention_loss=results['selected_sft']['retention']['val_loss']-results[parent_label]['retention']['val_loss'],
        scope=('Separately authored task combinations and held-out natural behavior, plus development knowledge and prose retention; final-test failures flag the candidate for review. This does not certify all assistant capabilities.' if stage2 else 'Grouped held-out passage/short-answer and authored-template tests; not a broad assistant capability certificate.'))
    atomic_json(summary,receipt);print(receipt,flush=True)


if __name__=='__main__':main()
