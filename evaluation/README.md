# Evaluation source and recorded evidence

The complete saved choice and IFEval results are in `../results/public/`. Choice files retain example indices, summed likelihoods, predictions, gold labels and truncation counts. IFEval files retain generated responses and strict/loose verifier decisions; `dataset_row_index` maps back to the pinned Google dataset. Upstream prompt fields are omitted from these copies. None of these files represents a newly rerun suite during packaging.

`benchmark_best.py`, `benchmark_ifeval.py` and `chat_adapter_search_bench.py` are historical entry points. Their paths and imports expect the original research workspace. Materialize the matching `sft3` source snapshot to audit their imported training modules. Running the historical suite additionally requires prepared dataset caches, original checkpoint metadata and the pinned IFEval runtime. They are not advertised as standalone commands for this smaller inference package.

The Google evaluator source, original headers and license are retained under `vendor/ifeval`; dependency versions and the source commit are recorded there. ARC/PIQA/HellaSwag dataset pins and evaluation settings are in the saved choice result JSON. The authored chat panel/scorer source is in `chat_benchmark/`.

For a standalone package check that does work directly in this release, run:

```bash
python scripts/verify_inference.py --output /tmp/solstice-inference-check.json
```

It runs 27 position-selected examples across seven chat categories, compares exact text/stop/token counts against saved FP32 answers, checks context/serialization guards, and checks fixed-seed sampling. Both correct and incorrect saved answers are included. This is package parity verification, not a substitute for the full benchmark suite or a new accuracy estimate.
