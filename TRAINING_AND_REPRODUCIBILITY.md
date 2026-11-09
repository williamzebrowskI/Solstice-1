# Training and reproducibility

## What is reproducible from this package

The selected FP32 inference weights, tokenizer, architecture, native serialization and runner are supplied. The release verification records loading, example output parity, token/control boundaries and package integrity checks. The existing complete public evaluations identify these exact weight bytes.

The package includes frozen source snapshots for every recorded pretraining job in the inherited lineage and for the three selected SFT stages. Identical files are stored once under `training/source_blobs`; snapshot manifests map their original module paths to these blobs and record original versus sanitized hashes. Use:

```bash
python scripts/materialize_training_snapshot.py --list
python scripts/materialize_training_snapshot.py --snapshot sft3 --output /tmp/solstice-sft3-source
```

The materializer creates a new directory and never overwrites an existing one. Read the resulting source and the corresponding `configs/` plan before attempting a training launch. Historical entry points include `sml_v1.pretrain`, `sft.lora_train`, `sft.lora_search_train`, and `sft.lora_repair_train`; they are archival training programs with dependencies on original run assets, rather than a verified one-command retraining interface.

## Required external inputs for exact retraining

- Pinned upstream datasets and the same filtering, splits, row identities and deterministic order.
- The original pretrained initialization and stream history for a full from-scratch reproduction.
- Exact ancestor base masters and adapters for a stage-only reproduction.
- Original prepared corpora, prose-replay pools, data receipts, comparison assets and optimizer/resume state where the selected stage imports them.
- A compatible distributed MLX/JACCL environment for pretraining. The public inference wheel does not itself establish five-worker training parity.

Raw source corpora, prepared training pools, optimizer binaries and all intermediate checkpoint weights are not redistributed. Full selected-pool file hashes, dataset revisions, plans, counters and later-stage selected batch indices are included. These establish provenance and permit auditing when the original inputs are available; they do not make unavailable corpora reconstructible from a hash.

## Stage boundaries and counters

The selected base contains 42,001,367,040 pretraining positions. SFT stages add 2,000, 256 and 128 updates respectively. The final stage was planned for 256 updates, but its later updates are not inherited. `provenance/SELECTED_PREFIX.json` distinguishes complete pools from the actually selected prefix and distinguishes unique rows from repeated presentations.

The original training runtime used MLX `0.32.3.dev20260920+7241f12`. Packaged inference uses the pinned stock runtime listed in requirements and verification. A complete fresh training rerun was not performed to validate numerical equivalence between these runtimes.

## Source sanitization and integrity

Absolute workspace/user paths become `${WORKSPACE}` / `${USER_HOME}` in archived configuration and source. Local network addresses are removed where present. Snapshot manifests retain original hashes and packaged hashes separately. Replacing placeholders and satisfying historical file guards is an explicit reproduction prerequisite. Private SSH configurations, network inventories, credentials and raw operational logs are not included.

The release source does not pretend that every imported helper recipe was a model ancestor. Only the phase/configuration and checkpoint lineage records identify selected training exposure. In particular, 167, 1200, 417, previous full-weight SFT attempts and DPO experiments are not additional ancestors of Solstice-1.

## Evaluations

`evaluation/` contains the recorded native scorer entry points and their dependencies/provenance. Frozen source hashes identify the historical implementation. Public metric JSON files include dataset revisions, count checks, precision and inference settings. New runs should use a fresh output directory and preserve generated responses. Package verification is deliberately reported separately from a full benchmark rerun.
