#!/usr/bin/env python3
"""Fresh five-Mac research pretraining; plan only unless --run is explicit."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.common import ROOT, file_sha256, fingerprint, load_corpus, lock, read_json
from sml_v1.recipe import learning_rate, tokens_per_update, validate
from sml_v1.sample_prompts import read_prompts
from sml_v1.lr_scheduler import adaptive_recipe, restore_scheduler
from sml_v1.continuation import extend_recipe


def resume_recipe(recipe, metadata, *, plateau_config=None):
    """Enable one explicit controller transition, or retain its saved history."""
    if metadata is None:
        if plateau_config is not None or 'plateau_scheduler' in recipe:
            raise ValueError('Plateau scheduler requires an explicit resume checkpoint')
        return recipe
    saved = metadata['recipe']
    if plateau_config is not None or 'plateau_scheduler' in saved:
        base = copy.deepcopy(saved)
        base.pop('plateau_scheduler', None)
        base.pop('scheduler_transition', None)
        if recipe != saved and recipe != base:
            raise ValueError('Scheduler resume must preserve all checkpoint training settings')
        if plateau_config is not None:
            return adaptive_recipe(metadata, plateau_config)
        # Reusing the original config must never silently remove the controller.
        return validate(copy.deepcopy(saved))
    if 'plateau_scheduler' in recipe or 'scheduler_transition' in recipe:
        raise ValueError('Enable the scheduler explicitly with --plateau-config')
    return recipe


def budget_recipe(recipe, metadata, target_tokens):
    """A prepared extension may change only the budget and its derived schedule."""
    if metadata is None:
        raise ValueError('Budget extension requires an explicit resume checkpoint')
    expected = extend_recipe(metadata, target_tokens)
    if recipe != expected:
        raise ValueError('Prepared config differs from the exact checkpoint budget extension')
    return expected


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT.parent/'configs/pretrain_20b.json')
    p.add_argument('--corpus', type=Path, default=ROOT.parent/'configs/corpus_20b.json')
    p.add_argument('--data', type=Path, help='Verified frozen token pools for a curated-data recipe; coordinator only')
    p.add_argument('--tokenizer', type=Path,
                   default=ROOT.parent/'reference/opensml-150m/tokenizer/bytebpe32k_v1')
    p.add_argument('--sample-prompts-file', type=Path,
                   help='Live file on Mac 1; overrides the recipe-relative prompt path')
    p.add_argument('--output', type=Path, default=ROOT/'runs/research_20b_176m_b44222')
    p.add_argument('--network', type=Path, help='Verified current five-Mac network plan')
    p.add_argument('--resume', type=Path, help='Explicit checkpoint from this output; otherwise fresh')
    p.add_argument('--extend-target-tokens', type=int,
                   help='Explicit TOTAL token budget; requires a prepared derived config and resume checkpoint')
    p.add_argument('--change-batches', type=int, nargs=5, help='Explicit resume-only five-rank microbatch change')
    p.add_argument('--plateau-config', type=Path,
                   help='Explicit resume-only validation scheduler settings; saved state persists on later resumes')
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--stop-after-steps', type=int, default=0,
                   help='Bounded smoke/pilot run; 0 means the full token target')
    p.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    if a.stop_after_steps < 0:
        p.error('--stop-after-steps must be nonnegative')
    if a.run and a.network is None:
        p.error('--run requires --network with fresh network/correctness verification')
    if a.plateau_config and a.change_batches:
        p.error('Change the scheduler and batches in separate explicit resumes')
    if a.extend_target_tokens is not None and (not a.resume or a.plateau_config or a.change_batches):
        p.error('Budget extension requires --resume without scheduler or batch changes')
    return a


def main(argv=None):
    a = arguments(argv)
    config, output = a.config.resolve(), a.output.resolve()
    recipe = validate(read_json(config))
    metadata = None
    if a.resume:
        from sml_v1.checkpoint_metadata import verify_bundle_metadata
        metadata = verify_bundle_metadata(a.resume)
    if a.change_batches:
        if not a.resume:
            raise ValueError('--change-batches requires an explicit resume checkpoint')
        from sml_v1.batch_transition import change_batches
        if 'plateau_scheduler' in metadata['recipe']:
            raise ValueError('Use a separate reviewed transition to change batches after enabling the scheduler')
        recipe = change_batches(metadata, a.change_batches)
    if a.extend_target_tokens is not None:
        recipe = budget_recipe(recipe, metadata, a.extend_target_tokens)
    else:
        recipe = resume_recipe(recipe, metadata,
                               plateau_config=read_json(a.plateau_config) if a.plateau_config else None)
    scheduler = restore_scheduler(recipe, metadata)
    if any(k in recipe for k in ('batch_transition', 'batch_rebalance_transition')) and not a.resume:
        raise ValueError('Batch transition requires explicit resume')
    streaming = recipe.get('data_mode') == 'hf_stream'
    if len(recipe['batches']) != 5 or not (streaming or recipe.get('data_policy')):
        raise ValueError('Research launcher requires five ranks and HF streaming or explicitly pinned curated pools')
    if streaming and a.data is not None:
        raise ValueError('--data is incompatible with HF streaming')
    if not streaming and a.data is None:
        raise ValueError('Curated training requires --data with the completed frozen corpus')
    if 'continuation' in recipe and a.extend_target_tokens is None:
        raise ValueError('Budget continuation requires explicit --extend-target-tokens')
    if any(k in recipe for k in ('stage_b_transition', 'topology_transition',
                                 'lr_trial_transition')):
        raise ValueError('Use a fresh research recipe, not a historical transition')
    corpus = load_corpus(a.corpus)
    prompts_file = (a.sample_prompts_file or config.parent/recipe['sample_prompts_file']).resolve()
    prompts, _ = read_prompts(prompts_file)
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('Research outputs must stay inside pretraining/runs')
    if (output/'latest.json').exists() and a.resume is None:
        raise ValueError('Existing output requires explicit --resume; fresh weights never overwrite it')
    initial_extension = (a.extend_target_tokens is not None and metadata is not None
                         and 'continuation' not in metadata['recipe'])
    if a.resume and not a.resume.resolve().is_relative_to(output) and not initial_extension:
        raise ValueError('Resume must come from this research output')
    if initial_extension:
        parent_bundle = a.resume.resolve()
        if parent_bundle.suffix == '.json':
            parent_bundle = parent_bundle.parent/read_json(parent_bundle)['bundle']
        parent_run = parent_bundle.parent.resolve()
        if output == parent_run or output.is_relative_to(parent_run):
            raise ValueError('Use a separate continuation output; preserve the parent run')
    if (output/'latest.json').exists() and a.resume.resolve() != (output/'latest.json').resolve():
        raise ValueError('Existing output must resume its own latest.json')
    if metadata is not None and metadata['tokens'] >= recipe['target_tokens']:
        raise ValueError('Token target already reached; refusing additional updates')
    # Validate the full resume contract during planning too, without loading a model.
    from sml_v1.tokenization import Tokenizer
    from sml_v1.stream import stream_manifest
    tokenizer = Tokenizer(a.tokenizer)
    if tokenizer.vocab_size != recipe['model']['vocab_size']:
        raise ValueError('Frozen tokenizer vocabulary differs from model')
    if streaming:
        manifest = stream_manifest(corpus, tokenizer, recipe['stream'])
    else:
        from sml_v1.data import verify_data
        from sml_v1.common import quotas
        manifest = verify_data(a.data, tokenizer, corpus=corpus, policy=recipe['data_policy'])
        targets = quotas(recipe['target_tokens'], corpus['sources'])
        for record in manifest['files']:
            minimum = (targets[record['source']] + 2*tokens_per_update(recipe) if record['split'] == 'train'
                       else recipe['eval_batches_per_source']*recipe['eval_batch_size']*recipe['model']['max_seq_len']+1)
            if record['tokens'] < minimum:
                raise ValueError(f'Insufficient prepared tokens: {record["file"]}')
    modules = {p.name:file_sha256(p) for p in (ROOT/'sml_v1').glob('*.py')}
    transition = None
    if metadata is not None:
        from sml_v1.continuation import check_resume
        contract = dict(recipe=fingerprint(recipe), tokenizer=tokenizer.fingerprint,
                        data=fingerprint(manifest), code=fingerprint(modules))
        transition = check_resume(metadata, contract, recipe, manifest)
    print(json.dumps(dict(mode='run' if a.run else 'plan only', initialization='resume' if a.resume else 'fresh',
        config=str(config), output=str(output), target_tokens=recipe['target_tokens'], model=recipe['model'],
        source=str(a.resume) if a.resume else None, resume_transition=transition,
        start_step=metadata['step'] if metadata else 0,
        start_tokens=metadata['tokens'] if metadata else 0,
        remaining_updates=(recipe['target_tokens']-(metadata['tokens'] if metadata else 0)
                           +tokens_per_update(recipe)-1)//tokens_per_update(recipe),
        eval_every_tokens=recipe['eval_every_tokens'],
        validation_fingerprint=metadata.get('validation_fingerprint') if metadata else None,
        physical_mac_order=[1,2,3,5,4], batches=recipe['batches'], grad_accum=recipe['grad_accum'],
        cache_gib_by_mac=recipe.get('cache_gib_by_mac', [4] * 5),
        tokens_per_update=tokens_per_update(recipe), peak_lr=recipe['peak_lr'],
        warmup_tokens=int(recipe['target_tokens']*recipe['warmup_fraction']),
        decay_start_tokens=int(recipe['target_tokens']*recipe['decay_start_fraction']),
        final_lr=learning_rate(recipe['target_tokens'],recipe),
        plateau_scheduler=recipe.get('plateau_scheduler'),
        scheduler_state=scheduler.state_dict() if scheduler else None,
        next_learning_rate=(scheduler.rate(learning_rate(metadata['tokens'] + tokens_per_update(recipe),recipe))
                            if scheduler else None),
        corpus_shares={s['label']:s['weight'] for s in corpus['sources']},
        data_mode=recipe['data_mode'], data=str(a.data.resolve()) if a.data else None,
        data_manifest_sha256=recipe.get('data_policy', {}).get('manifest_sha256'),
        tokenizer=str(a.tokenizer.resolve()), sample_prompts_file=str(prompts_file),
        sample_prompt_count=len(prompts), prompt_reload='each sampling event on coordinator',
        validation='fixed held-out data; editable prompts are qualitative samples only',
        stop_after_steps=a.stop_after_steps), indent=2), flush=True)
    if not a.run:
        print('No downloads, SSH, model initialization or training. Add --run only after preflight.')
        return
    proof = read_json(a.network/'five_rank_correctness.json')
    receipts = proof.get('receipts', [])
    if (proof.get('modules') != modules or len(receipts) != 5 or
            {r.get('rank') for r in receipts} != set(range(5)) or
            any(r.get('status') != 'passed' for r in receipts)):
        raise ValueError('Run five-rank correctness checks against the current worker code first')
    from five_mac_cluster import Cluster, PYTHON
    from sml_v1.launch import stage
    cluster = Cluster(a.network)
    with lock(a.network/'.training-launch.lock'), lock(output/'.launcher.lock'):
        if (output/'STOP').exists() and not a.clear_stop:
            raise ValueError('STOP exists; use --clear-stop explicitly to resume')
        job = output/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster.preflight(job)
        source = stage(job, recipe, a.tokenizer, a.resume, corpus)
        cluster.stage(source)
        if a.clear_stop:
            (output/'STOP').unlink(missing_ok=True)
        worker = ['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v1.pretrain',
            '--config',str(source/'recipe.json'),'--tokenizer',str(source/'tokenizer'),
            '--corpus',str(source/'corpus.json'),'--save-dir',str(output),'--job-dir',str(job),
            '--ring-control','native','--stop-after-steps',str(a.stop_after_steps),
            '--sample-prompts-file',str(prompts_file)]
        if not streaming:
            # Rank 0 reads the immutable pools and distributes disjoint batches.
            # The other Macs receive the pinned recipe/corpus/tokenizer/code.
            worker += ['--data', str(a.data.resolve())]
        if a.resume:
            worker += ['--resume',str(source/'checkpoint')]
        cluster.launch(job, source, worker, timeout=None, stop_dir=output)
        result = read_json(job/'result.json')
        if result.get('status') not in ('stopped','complete'):
            raise RuntimeError('Missing clean worker completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}', flush=True)


if __name__ == '__main__':
    main()
