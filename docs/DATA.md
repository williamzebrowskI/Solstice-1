# Solstice-1 data provenance

## Pretraining

Configured token shares are 70/20/10 web/DCLM/wiki until 36,801,216,512 positions, then 56/16/8/15/5 web/DCLM/wiki/Python/SQL through 42,001,367,040. These shares describe the mixture controller; they are not a separately audited count of all unique content delivered. The SQL configuration includes a SynSQL mirror continuation source. Configuration establishes eligibility, not proof that every configured downstream shard was consumed. Streaming cursor/validation records are not represented as a global deduplicated data manifest.

The frozen stream uses bounded accepted-document shuffling and a 100,000-document recent-deduplication window. Content identity controls the fixed train/validation partition. English prose and code filters are archived in each source snapshot. DCLM and code sources require their configured educational score at least 3. Text size bounds are 200 characters through 200,000 UTF-8 bytes where applicable. Null characters, corruption/replacement patterns and repetitive content are filtered by the source-specific code. These rules are heuristic; they do not guarantee every retained example is correct, licensed identically, or uncontaminated.

Python and SQL full-text mirrors retain upstream repository/license fields. The recorded filter sets `require_explicit_license=false`; it must not be described as an explicit-license-only corpus. No raw training corpus is redistributed with this release.

## Dataset identities

The following pins and terms come from the frozen configurations and corpus manifests. They are source provenance, not a blanket grant of rights to all underlying content.

| Source | Pinned revision | Recorded terms / role |
| --- | --- | --- |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/3ba9d605774198c5868892d7a8deda78031a781f) | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-BY; preserve underlying source notices |
| [HuggingFaceTB/dclm-edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/tree/dbad8ad71224482740cd9c9d353591adbf62fe04) | `dbad8ad71224482740cd9c9d353591adbf62fe04` | CC-BY-4.0 |
| [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki/tree/8bd13e72e6a002407649b3e898535f42ceb1aeb9) | `8bd13e72e6a002407649b3e898535f42ceb1aeb9` | CC-BY-SA-4.0 / GFDL |
| [hongliu9903/stack_edu_python](https://huggingface.co/datasets/hongliu9903/stack_edu_python/tree/76eeb3eb59bbad948e4fbfae9f4e6159d6f8f376) | `76eeb3eb59bbad948e4fbfae9f4e6159d6f8f376` | Original source repository licenses; preserve provenance and notices |
| [hongliu9903/stack_edu_sql](https://huggingface.co/datasets/hongliu9903/stack_edu_sql/tree/079ce8b93c0e83a767ee61af20b9de9931a7e07c) | `079ce8b93c0e83a767ee61af20b9de9931a7e07c` | Original source repository licenses; preserve provenance and notices |
| [iNeil77/SynSQL-2.5M](https://huggingface.co/datasets/iNeil77/SynSQL-2.5M/tree/56fed46215366defbbc3d7d13064349bbd990a8f) | `56fed46215366defbbc3d7d13064349bbd990a8f` | See upstream source terms |
| [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/tree/f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc) | `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` | apache-2.0 |
| [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/tree/5feaf2fd3ffca7c237fc38d1861bc30365d48ffa) | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | see upstream component license |
| [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2/tree/3ffb306f725f7d2ce8394bc1873b24868140c412) | `3ffb306f725f7d2ce8394bc1873b24868140c412` | ['cc-by-sa-4.0'] |
| [allenai/sciq](https://huggingface.co/datasets/allenai/sciq/tree/2c94ad3e1aafab77146f384e23536f97a4849815) | `2c94ad3e1aafab77146f384e23536f97a4849815` | cc-by-nc-3.0 |

## Stage-1 conversations

The 48,000-conversation pool contains 6,200 constraint examples, 8,000 dialogue examples, 4,800 general responses, 11,000 grounded examples, 4,000 knowledge examples, 1,000 rewrites, 12,000 skills examples and 1,000 social examples. Public sources are Smol-SmolTalk, SmolTalk everyday conversations, SQuAD v2 TRAIN and SciQ TRAIN. Computed/authored tasks supply much of the skills, correction/dialogue and constraint coverage. Two complete passes produce 96,000 conversation presentations; prose replay contributes another 32,000 512-token sequences.

Public answers were filtered and sample-reviewed; not every statement received independent factual verification. SQuAD span checks verify annotation consistency, not every annotation's truth. The frozen data manifests preserve rejection rules, token statistics and file hashes.

## Later selected prefixes

Stage 2 draws 4,096 distinct new tasks from an 8,000-task authored pool, plus 4,096 distinct replay conversations. Stage 3 draws from a new 20,480-task pool; the released prefix inherits 2,048 fresh tasks and 2,048 replay conversations. Stage-3 fresh counts are reasoning 1,230; grounding 324; dialogue 306; abstention 188. The full five-experiment search did not all become training exposure for this model.

The replay pool contains prior stage-1 material, including SciQ and SQuAD. Later plan files contain inherited source declarations such as HelpSteer2/OASST/Smol-Magpie; those declarations are **not evidence those sources were fetched for these stages**. Actual selected rows, subsets and source-file hashes in [SELECTED_PREFIX.json](../provenance/SELECTED_PREFIX.json) are authoritative. The release's dataset metadata lists the actual inherited sources rather than those stale plan declarations.

## Filtering and evaluation separation

Conversations are retained whole and serialized using the frozen tokenizer. The selected text-focused SFT excludes dedicated Python, SQL, JSON and tool tasks. Exact prompt exclusions and grouped splits were checked. The new authored panel separates entity values and wording between train/development/test, but shares task families. All 21,780 newly authored train/dev/test rows were independently parsed and target-checked during preparation. Its 1,000-case test was unopened until one finalist was frozen.

Historical 320/704 panels and public benchmarks had prior research exposure. Their use does not provide a fresh blind evaluation of the entire research program. Hash-based decontamination does not establish absence of paraphrases or source-document overlap in pretraining.

## Redistribution

Author-controlled code and selected model weights use Apache-2.0 by the author's release decision. Dataset terms remain distinct. In particular, recorded SQuAD terms are CC-BY-SA-4.0 and SciQ terms are CC-BY-NC-3.0; Python/SQL repositories can have heterogeneous terms. The release does not relicense or redistribute the raw source datasets. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for attribution and the exact upstream links.
