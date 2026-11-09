---
language:
- en
license: apache-2.0
pipeline_tag: text-generation
library_name: mlx
tags:
- small-language-model
- from-scratch
- text-generation
- conversational
- apple-silicon
- research
datasets:
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/dclm-edu
- HuggingFaceFW/finewiki
- hongliu9903/stack_edu_python
- hongliu9903/stack_edu_sql
- iNeil77/SynSQL-2.5M
- HuggingFaceTB/smol-smoltalk
- HuggingFaceTB/smoltalk
- rajpurkar/squad_v2
- allenai/sciq
---

# Solstice-1

**An OpenSML 175M-parameter English chat research model trained from scratch on Apple Silicon.**

Author: William Zebrowski · License: Apache-2.0 · Status: research preview

[Model weights](https://huggingface.co/OpenSML/Solstice-1) · [GitHub source](https://github.com/williamzebrowskI/Solstice-1) · [Run guide](RUN_GUIDE.md) · [Technical report](docs/TECHNICAL_REPORT.md) · [Evaluation](docs/EVALUATION.md) · [Training data](docs/DATA.md) · [Results](results/public/summary.json)

Solstice-1 starts a model family under OpenSML. This first release is a 175M chat model with a 2,048-token context. Its selected weights come from **42.001B pretraining tokens and three adapter fine-tuning stages**. The package provides merged FP32 weights, the original tokenizer, a native MLX runner, training-source snapshots, and recorded evaluation evidence.

The model is intended for research into small English language models, short conversations, reading supplied text, and instruction following. It can produce incorrect answers, repeat itself, decline answerable questions, and invent information that was not provided. The measurements below do not establish dependable factual or production behavior.

## Model specification

| Setting | Value |
| --- | --- |
| Public size designation | 175M |
| Exact parameter count | 174,202,432 |
| Transformer | 44 layers; width 576; SwiGLU intermediate width 1,536 |
| Attention | 9 query heads / 3 KV heads; head dimension 64 |
| Components | RoPE 10,000; RMSNorm; Q/K normalization; tied embeddings |
| Vocabulary / shared context | 32,000 byte-level BPE tokens / 2,048 tokens |
| Pretraining exposure | 42,001,367,040 processed token positions |
| Released weights | Unquantized merged FP32 safetensors |
| Supported inference | Supplied native MLX runner on Apple Silicon |

This is a different architecture and lineage from OpenSML-150M and OpenSML-JSONLite-175M. Those releases are comparison references, not weight ancestors. The frozen tokenizer is reused from the earlier OpenSML project.

## Run locally

Download the entire model repository, including its tokenizer and native source. While the repository is private, run `hf auth login` after installing `huggingface_hub`, using an account with access to OpenSML/Solstice-1:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download OpenSML/Solstice-1 --local-dir Solstice-1
cd Solstice-1
python -m pip install -r requirements.txt
python inference.py --prompt "What is the main function of the heart?" --max-new-tokens 96
python inference.py --interactive
```

The runner uses the model's native structural-token chat format. It checks the weight hash and tokenizer before loading, defaults to greedy FP32 inference, and rejects context overflow without truncation. `/reset` clears an interactive conversation; `/quit` exits. See [recorded package verification](PACKAGING_VERIFICATION.json) for the environment and checks actually completed.

**Use the supplied runner.** This package does not provide a verified Transformers AutoModel, MLX-LM auto-loader, GGUF, or CUDA implementation. The context limit includes previous turns, role delimiters, the current prompt, and the requested output budget.

## Measured results

Full-split, zero-shot likelihood evaluations use native MLX FP32 scoring:

| Benchmark | Examples | Accuracy | Normalized accuracy |
| --- | --- | --- | --- |
| ARC Easy | 2376 | 62.58% | 62.33% |
| ARC Challenge | 1172 | 29.35% | 30.89% |
| PIQA | 1838 | 66.49% | 66.21% |
| HellaSwag | 10042 | 32.85% | 39.40% |

`acc_norm` divides each candidate's summed log-likelihood by its Unicode-character count. It is distinct from raw accuracy. The evaluator follows pinned task conventions but is not an installed lm-evaluation-harness run.

| IFEval metric | Score | Passes / checks |
| --- | --- | --- |
| Strict prompts | 17.93% | 97/541 |
| Strict instructions | 30.10% | 251/834 |
| Loose prompts | 19.22% | 104/541 |
| Loose instructions | 32.49% | 271/834 |

IFEval uses all 541 prompts / 834 instructions, greedy native-chat generation, no system message or repetition penalty, and a 1,024-token output cap. **130/541 responses reach that cap.** All responses are scored as produced. These numbers do not measure general factual accuracy.

On a separate 192-response authored chat check, 107/180 automatically checked answers match the accepted answers; 191/192 responses stop naturally, and three have repetition flags. Automatic answer matching is conservative, and repetition detection is heuristic. These checks are not a broad human preference study.

![Recorded pretraining history](docs/figures/pretraining-overview.png)

See the [evaluation report](docs/EVALUATION.md) for baseline comparisons, dataset pins, protocol differences, and private-test scope.

## Training lineage

```text
Fresh Deep44 initialization → streamed prose pretraining
  → add Python/SQL at 36.801B → 40.001B → 2.000B-token cooldown
  → 42.001B pretrained base
  → adapter SFT 1, selected update 2000
  → adapter SFT 2, selected update 256
  → adapter SFT 3, selected update 128 → Solstice-1
```

Only updates through 128 of the final 256-update experiment are inherited. Earlier experimental branches and checkpoint 167 are not additional stages in this lineage. [Checkpoint lineage and hashes](provenance/LINEAGE.json) · [Selected data exposure](provenance/SELECTED_PREFIX.json).

## Known limitations

- The model sometimes gives fluent but incorrect factual explanations.
- On the new authored evaluation, 111/200 answerable extraction questions triggered unnecessary-abstention signals. On a separate set with different wording, it failed all 15 missing-information cases. These are distinct tests, not contradictory measures of the same inputs.
- Instruction following remains incomplete; only 97/541 IFEval prompts satisfy every strict requirement.
- Repetition and token-limit stops remain possible, especially on longer requested outputs.
- JSON extraction, tool use, robust coding, long documents, multilingual use, safety behavior, and prompt-injection resistance are not validated release capabilities.
- Public and historical benchmarks informed earlier experiments. The newly reserved 1,000-case panel was evaluated after this finalist was selected, but shares authored task families with training. No comprehensive semantic contamination audit or independent human evaluation is claimed.

## Repository contents and licensing

The weights and author-controlled code are released under [Apache-2.0](LICENSE). Upstream datasets and third-party code retain their own terms; the model license does not relicense those materials. [Source notice](SOURCE_NOTICE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md).

Technical details, data provenance, exact selected training prefixes, source snapshots, and reproduction prerequisites are included. Full raw corpora, optimizer binaries, and all ancestor weights are not distributed. The historical training archive is not a verified one-command retraining package. [Training and reproducibility](TRAINING_AND_REPRODUCIBILITY.md).
