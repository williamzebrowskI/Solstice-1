# Third-party notices

## Runtime and source

MLX is an Apple open-source project: [ml-explore/mlx](https://github.com/ml-explore/mlx). NumPy and Hugging Face Tokenizers are dependencies distributed under their own terms. Their packages are installed separately and are not relicensed by Solstice-1.

Archived training code includes SQLGlot source and its retained license where required by the historical code filter. Snapshot manifests preserve the license file mapping. Google IFEval evaluator source and its Apache-2.0 attribution are preserved with the evaluation archive. Dataset content is not made Apache-2.0 by being used in training or evaluation.

## Training sources

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

SciQ is recorded as CC-BY-NC-3.0 and SQuAD v2 as CC-BY-SA-4.0. SmolTalk-family synthetic responses retain their upstream provenance. FineWeb-related database terms and underlying webpage rights are separate. The Python/SQL mirrors carry heterogeneous original repository notices and did not require an explicit license on every retained record. These statements describe the frozen records, not a newly completed item-by-item rights audit.

## Evaluation sources

[ARC](https://huggingface.co/datasets/allenai/ai2_arc), [PIQA](https://huggingface.co/datasets/baber/piqa), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), and [IFEval](https://huggingface.co/datasets/google/IFEval) are external evaluation datasets. Their versions and source code hashes are in the public result records. The repository distributes aggregate scores, authored checks and model-generated answers as documented; full public benchmark datasets are obtained separately under upstream terms.

## Prior OpenSML references

[OpenSML-150M](https://huggingface.co/OpenSML/OpenSML-150M) and [OpenSML-JSONLite-175M](https://huggingface.co/OpenSML/OpenSML-JSONLite-175M) are earlier releases by the same author. Their weights are not ancestors of Solstice-1. Prior documentation layouts informed this package; their task-specific results are not silently assigned to this model.
