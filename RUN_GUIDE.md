# Run Solstice-1

Use Python 3.11 on Apple Silicon macOS with the pinned requirements. The tested public MLX wheel is 0.32.3; it requires a compatible macOS version. See PACKAGING_VERIFICATION.json for the actual tested system. Inference is offline after downloading the package and installing dependencies.

While the model repository is private, run `hf auth login` after installing `huggingface_hub` below. Use a Hugging Face account with read access to OpenSML/Solstice-1. Tokens belong in the terminal sign-in prompt, never in a shared command, notebook, or repository file.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download OpenSML/Solstice-1 --local-dir Solstice-1
cd Solstice-1
python -m pip install -r requirements.txt
python scripts/verify_package.py
python inference.py --prompt "Name the largest continent." --max-new-tokens 64
python inference.py --interactive
```

For the GitHub source checkout, download the matching `model.safetensors` from the Hugging Face repository into its root. That large weight file is deliberately excluded from Git history.

## Previous messages

Interactive mode retains complete previous turns. `/reset` starts a new conversation and `/quit` exits. The program never silently truncates your history. When history plus output budget would exceed 2,048 tokens, reset it or shorten the request.

For scripted conversations, create a JSON file containing message objects:

```json
[
  {"role":"user","content":"My name is Rowan."},
  {"role":"assistant","content":"Hello, Rowan."},
  {"role":"user","content":"What is my name?"}
]
```

```bash
python inference.py --messages-file conversation.json --max-new-tokens 64 --json
```

That file describes the input conversation; it does not ask the model to produce JSON. An optional system message is accepted only as the first message. No system message is inserted by default.

## Generation settings and outputs

Greedy temperature zero is the default and matches benchmark decoding. `--temperature 0.6 --seed 2026` enables sampling; sampled outputs are not the reported benchmark protocol. The supplied runner uses no repetition penalty, output repair, external model, retrieval, or constrained decoder. `--json` wraps the unmodified generated answer with token IDs, lengths, timing and stop reason for inspection.

Stop reasons are `turn_end`, `document_end`, `token_limit`, or `unexpected_structure`. A natural stop says the model completed a turn; it does not establish answer correctness. Generation defaults to 128 new tokens and can be changed within the context budget. In the native playground, additional controls and history trimming may differ from this standalone CLI; use the CLI's recorded protocol for reproducibility.

## Verification and troubleshooting

Run `python scripts/verify_package.py` to check the recorded file hashes. The runner independently verifies its weight checksum and tokenizer manifest. A missing or mismatched weight file is rejected. Unknown role sequences and context overflow are rejected before generation.

Use the supplied `inference.py`; Transformers AutoModel, MLX-LM automatic loading and GGUF conversion are not supported by this release. Do not repair a hash mismatch by editing the expected hash. Redownload the intended package instead.
