"""Freeze the original pretraining validation bytes for SFT retention checks."""
import argparse
from pathlib import Path
import shutil
import uuid
import numpy as np
from sml_v1.common import atomic_json, file_sha256, load_corpus, read_json
from sml_v1.tokenization import Tokenizer
from sml_v1.stream import stream_manifest, validation_batches


def verify(path, expected=None):
    path=Path(path);receipt=read_json(path/'manifest.json')
    if receipt['format']!='opensml-prose-sft-retention-v1' or (expected and receipt['validation_fingerprint']!=expected):
        raise ValueError('Pretraining retention panel changed')
    if file_sha256(path/'panel.npz')!=receipt['sha256']:
        raise ValueError('Pretraining retention panel checksum failed')
    return receipt


def prepare(config):
    plan=read_json(config);dest=Path(plan['retention'])
    source=read_json(Path(plan['source_checkpoint'])/'model.safetensors.json')
    if dest.exists():return verify(dest,source['validation_fingerprint'])
    project=Path(__file__).resolve().parents[2]
    tokenizer=Tokenizer(plan['tokenizer'])
    corpus=load_corpus(project/'configs/corpus_prose_stream_20b_20261022.json')
    manifest=stream_manifest(corpus,tokenizer,source['recipe']['stream'])
    batches,digest=validation_batches(manifest,tokenizer,source['recipe']['model']['max_seq_len'],
        source['recipe']['eval_batches_per_source'],source['recipe']['eval_batch_size'])
    if digest!=source['validation_fingerprint']:
        raise ValueError('Frozen original validation fingerprint does not reproduce')
    stage=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex);stage.mkdir(parents=True)
    try:
        arrays={source+'_'+axis:np.stack([b[i] for b in rows])
                for source,rows in batches.items() for i,axis in enumerate(('x','y'))}
        np.savez_compressed(stage/'panel.npz',**arrays)
        receipt=dict(format='opensml-prose-sft-retention-v1',validation_fingerprint=digest,
            sha256=file_sha256(stage/'panel.npz'),weights=manifest['weights'],tokenizer=tokenizer.fingerprint)
        atomic_json(stage/'manifest.json',receipt);stage.rename(dest);return receipt
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();print(prepare(a.config),flush=True)
