# Experiment data retained for the ICLR manuscript

This directory is a deliberately small migration bundle. It retains the records needed to check the numbers currently reported in the manuscript without copying checkpoints, partial files, logs, or duplicate historical runs from the full `output/` tree.

## Main full-release result

- `main_full/source_summary.json` is the summary produced by the selected 1,055-problem Full Dual-Loop run.
- `main_full/source_traces.tar.gz` contains its `traces.json`. It is retained because the SAS transitions, property-clause counts, trace evidence, and case study can be recomputed from this file.
- `main_full/heldout_per_problem.csv` contains the private held-out re-evaluation of the saved final programs.
- `main_full/heldout_summary.json` and `main_full/heldout_run_manifest.json` describe that re-evaluation with local paths anonymized.

The source run records 402 public-feedback passes. The paper's Final Pass criterion requires a program to pass both public feedback and private held-out tests, yielding 401 complete-suite passes. Four programs fail public feedback but pass the private evaluation; they are not counted as Final Passes. One public-feedback pass fails the private evaluation.

## Other retained evidence

- `main_baselines/`: summaries for the Direct, Decomposition Only, Self-Refine-style, and Reflexion-style rows used in the full-release comparison.
- `external_baselines/`: aggregate SpecFix-BM and LPW-adapted results, protocol configuration, and paper-table rows. Checkpoints and full traces are omitted.
- `cross_setting/results.csv`: the three rows needed for each Qwen, Llama, DeepSeek, and HumanEval+ setting in the robustness table.
- `component_ablation/results.csv`: the four rows in the controlled SAL/IRL component ablation.
- `intermediate_representation/`: the 50-problem Direct, plan, pseudocode, and specification probe.
- `sal_sensitivity/`: experiment design, sampled problem IDs, aggregate results, and per-problem sensitivity data. Per-configuration traces and partial files are omitted.
- `mechanism/`: requirement-heavy subset results and the reviewed 36-sample trace audit.
- `historical_nonfrozen/`: a compressed legacy trace retained only because the manuscript discusses reopening SAL after execution failure. It is not a primary comparison row.

The HumanEval+ source run uses the legacy tag `full_adaptive_sal`; in the manuscript it is reported as the ordinary Full configuration rather than as a separate adaptive method.

## Restoring the main trace

From the repository root:

```bash
mkdir -p output/dual_loop/full_Qwen2.5-Coder-7B-Instruct_full_dual_loop_20260529_005015_876463
tar -xzf paper/ICLR/experiment_data/main_full/source_traces.tar.gz \
  -C output/dual_loop/full_Qwen2.5-Coder-7B-Instruct_full_dual_loop_20260529_005015_876463
cp paper/ICLR/experiment_data/main_full/source_summary.json \
  output/dual_loop/full_Qwen2.5-Coder-7B-Instruct_full_dual_loop_20260529_005015_876463/summary.json
```

All JSON paths in this bundle are relative or anonymized. No model weights, benchmark installation, API credentials, machine logs, or private user paths are included.

