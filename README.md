# MP1 Final

This repository provides the final model, frozen checkpoints, original benchmark data, and scripts for local training and complete evaluation. The final method combines **RoPE, LayerNorm, squared ReLU, same-trajectory checkpoint averaging and a causal adaptive cache**.

The supplied `checkpoints/final.pt` achieves **1.5553096072435364 test BPB** and **25.82342986323253 token perplexity** using CPU FP32. Lower BPB is better. The evaluation protocol is `7506-mp1-wt2-v2`.

The [English Word report](REPORT.docx) is **6 pages**, within the 10-page limit, and covers methods, comparisons, ablations and critical analysis. Its [Markdown version](REPORT.md) uses figures in `report_assets/`.

## 1. Select a workflow

| Goal | Instructions |
|---|---|
| Evaluate the supplied model without training | Complete Sections 3–4, then Section 5 |
| Train the entire final recipe from scratch | Complete Sections 3–4, then Section 6 |
| Run training stages individually | Follow Section 7 instead of Section 6 |
| Run with PyCharm's green Run button | Set up the environment, then follow Section 8 |
| Review earlier experiments | Read Section 9 and the report |

This is a local command-line research project. No server, database or cloud account is required. The data and evaluation checkpoints are included. Dependency installation needs internet access; subsequent training and evaluation use local files.

## 2. Repository layout

```text
FINAL_SUBMISSION/
  README.md                 Complete local guide
  REPORT.docx               English report, 6 pages
  REPORT.md                 Corresponding Markdown report
  report_assets/            English report figures
  SUBMISSION_FILES.md       Submission checklist
  requirements.txt         Pinned dependencies
  PACKAGE_MANIFEST.json     File sizes and SHA256 hashes
  verify_package.py         Package integrity check
  student.py                Final model
  model.py                  Original GPT and required base classes
  common.py                 Original data and runtime utilities
  evaluate.py               Unchanged official scorer
  measure_eval.py           Windows whole-process peak RAM wrapper
  train.py                  Backbone training
  average.py                Checkpoint averaging and selection
  train_gate.py             Gate fitting with frozen backbone
  run_pipeline.py           Train, average, fit gate, freeze and test
  configs/final.json        Final backbone training configuration
  configs/baseline.json     Original baseline configuration
  data/                    Original text splits, tokenizer and manifest
  checkpoints/final.pt      Frozen submitted checkpoint
  checkpoints/baseline.pt   Baseline checkpoint for resource comparison
  tests/                   Correctness tests
  evidence/                Training, selection and evaluation records
```

Keep all five data files: `manifest.json`, `tokenizer.json`, `wikitext_train.txt`, `wikitext_validation.txt` and `wikitext_test.txt`. Do not edit the text or retrain the tokenizer; the loader checks hashes.

The evaluator reads the inference configuration from the checkpoint. `configs/final.json` uses `variant=residual_dropout` for backbone training; the final checkpoint uses `variant=gated_cache`. Its legacy `cache_lambda=0` field does not disable the learned gate. Do not remove `model.py` or copy `student.py` alone into another directory.

## 3. Set up Python and dependencies

### Original environment

The original experiment used **Python 3.12**, **PyTorch 2.7.1+cu126** and an **RTX 3060 Laptop GPU**. Backbone training used GPU BF16. All reported final evaluations used **CPU FP32, four threads**. Dependencies are `torch==2.7.1`, `numpy==2.5.3` and `tokenizers==0.21.4`.

A CUDA GPU is useful for training but unnecessary for evaluating the supplied model. CPU training is supported and substantially slower. Full training keeps multiple checkpoints; allow more disk space than the size of the submission archive.

### Windows PowerShell

Install 64-bit Python 3.12 if needed. Extract the project and open PowerShell in the folder containing `evaluate.py`. For the original local path:

```powershell
Set-Location 'E:\MP1_student_starter\MP1_student_starter\FINAL_SUBMISSION'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Replace the path if your extraction location differs. Create `.venv` once and reuse it. These instructions invoke its interpreter directly, so activation and PowerShell execution-policy changes are unnecessary.

Choose **one** torch installation option. For NVIDIA GPU training matching the original CUDA build:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU-only use:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The wheel indexes follow the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271). This project does not require torchvision or torchaudio. GPU operation requires a compatible NVIDIA driver.

Check the environment:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import sys, torch, numpy, tokenizers; print(sys.version); print('torch:', torch.__version__, 'numpy:', numpy.__version__, 'tokenizers:', tokenizers.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
```

For the matching GPU build, expect torch `2.7.1+cu126` and CUDA `True`. A CPU build is appropriate for CPU-only use. If BF16 is unsupported, use `--precision fp32` for training.

### Linux or macOS

Windows is the recorded reproduction environment. On another platform:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Linux, choose the CUDA or CPU torch command above using `python -m pip`, then install `requirements.txt`. On macOS, use `python -m pip install -r requirements.txt` and the CPU execution path. Replace `.\.venv\Scripts\python.exe` with `python` in subsequent commands. Cross-platform training equivalence has not been established. The RAM wrapper is Windows-only; use the official evaluator elsewhere and measure memory separately according to the course protocol.

## 4. Verify files and correctness

From the project root, after installing dependencies:

```powershell
.\.venv\Scripts\python.exe verify_package.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
Get-FileHash checkpoints/final.pt -Algorithm SHA256
```

The integrity check verifies registered package hashes without downloading or training. The test command checks model correctness, including causality and window independence; the retained submission checks report **12 passing tests**. These tests do not replace full-dataset scoring.

The final checkpoint SHA256 must be:

```text
7c53ebdcf91ea086c6916a082c3f01cdb3278747cf050ac3bba8cac0cb5998cf
```

New files in `runs/` or `results/` do not change registered file hashes. Intentional edits to registered code or documents make the integrity check report those changes. Preserve a clean copy before experimenting.

## 5. Run complete validation and test evaluation

### Score the supplied model without training

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/final.pt --split validation --device cpu --precision fp32 --threads 4 --output results/final_validation.json
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/final_test.json
Get-Content results/final_test.json
```

Each command evaluates the **complete split**, not a sample, and does not update parameters. Output directories are created automatically. Initial data loading may take about 20 seconds on the original machine.

| Metric | Recorded result |
|---|---:|
| Validation BPB | 1.5391035941086513 |
| Full-test BPB | 1.5553096072435364 |
| Full-test token PPL | 25.82342986323253 |
| Test prediction targets | 428405 |
| Test UTF-8 bytes | 1292013 |

Small numerical differences may arise across CPU implementations. Timing and RAM vary by machine and load. A GPU/BF16 score is not a substitute for the submission's CPU FP32 result.

### Measure evaluation resources on Windows

Run the two models **sequentially** on the same machine with identical settings:

```powershell
.\.venv\Scripts\python.exe measure_eval.py --checkpoint checkpoints/baseline.pt --split test --device cpu --precision fp32 --threads 4 --output results/baseline_resources_test.json
.\.venv\Scripts\python.exe measure_eval.py --checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/final_resources_test.json
```

Each run writes the score `.json`, peak-RAM `.resources.json` and per-window-loss `.window-nll.npy`. Scoring time is the score JSON's `seconds` field; the RAM measurement covers the full evaluator process including loading.

```powershell
$baselineResult = Get-Content results/baseline_resources_test.json -Raw | ConvertFrom-Json
$finalResult = Get-Content results/final_resources_test.json -Raw | ConvertFrom-Json
$finalResult.seconds / $baselineResult.seconds
Get-Content results/final_resources_test.resources.json
$assetBytes = (Get-Item checkpoints/final.pt,student.py,model.py | Measure-Object -Property Length -Sum).Sum
$assetBytes / 1MB
```

The time ratio must be **at most 5**, peak RAM **at most 4 GiB**, and uncompressed inference assets **at most 64 MiB**. The last expression gives MiB because PowerShell's `1MB` equals 1,048,576 bytes. Count the final checkpoint and model source, not compressed archive bytes.

The cleaned package recorded **59.448 s**, versus baseline **21.265 s**, ratio **2.796**, peak RAM **1.812 GiB** and inference assets **9.821 MiB**. On non-Windows systems, use `evaluate.py` and a separate appropriate RAM measurement.

## 6. Train and test with one command

Original GPU training path:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --out runs/reproduce-s17 --device cuda --precision bf16
```

CUDA without BF16 support:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --out runs/reproduce-fp32-s17 --device cuda --precision fp32
```

CPU-only training:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --out runs/reproduce-cpu-s17 --device cpu --precision fp32
```

Use a **new output directory** each time. The pipeline does not automatically resume an interrupted run. It performs:

1. Backbone training: 12,000 steps, seed 17, batch 32, LR 0.001 and weight decay 0.20, with validation and checkpoint saving every 600 steps.
2. Validation selection among averages of 1/3/5/7/9 same-trajectory checkpoints ending at the best single validation checkpoint.
3. Train-only fitting of constant and MLP cache gates with a frozen backbone; validation chooses the best MLP epoch out of at most ten.
4. Copying the winner to `final.pt` and writing source/checkpoint hashes to `FROZEN.json`.
5. Complete **CPU FP32 test scoring**, regardless of training device, followed by frozen-file hash checks.

The averaging and gate scripts use FP32. The pipeline's precision flag controls backbone training. The pipeline uses `evaluate.py` and therefore **does not automatically measure Windows peak RAM**. For that check, use Section 5's resource commands with `runs/reproduce-s17/final.pt` as the final checkpoint.

```text
runs/reproduce-s17/
  train/                    Backbone weights, metrics and source snapshots
  average/                  Averaged weights and selection.json
  gate/                     Gate candidates, checkpoint and selection.json
  final.pt                  Model selected by this new run
  FROZEN.json               Hashes recorded before testing
  test_cpu_fp32.json         Complete test score
  test_cpu_fp32.window-nll.npy
  train.log
  average.log
  train_gate.log
  evaluate.log
```

The terminal shows stage names; detailed progress goes to logs. In a second PowerShell window, after the log exists:

```powershell
Get-Content runs/reproduce-s17/train.log -Tail 20 -Wait
```

Ctrl+C stops this log viewer, not the separate training process. The recorded backbone process took about nine minutes on the RTX 3060 Laptop, plus averaging, gate fitting and scoring. This is not a timing guarantee. Retraining can select different steps/epochs or produce slightly different scores; the published result belongs to the supplied `checkpoints/final.pt`.

## 7. Train each stage manually

Use this section instead of the one-command pipeline. For CPU operation, change every `--device cuda` to `--device cpu` and use `--precision fp32` for backbone training. Use new output directories.

### Step 1 — Backbone training

```powershell
.\.venv\Scripts\python.exe train.py --config configs/final.json --run-dir runs/manual-train --device cuda --precision bf16 --threads 4 --seed 17 --steps 12000 --batch-size 32 --eval-every 600 --save-best --lr 0.001 --weight-decay 0.20
```

Architecture: four layers, width 192, four heads, FFN width 768, RoPE theta 10000, LayerNorm, squared ReLU, dropout zero. AdamW uses betas `(0.9,0.999)`, 100 warmup steps, cosine decay to 0.1 of initial LR and gradient clipping at 1. The backbone processes `12000 × 32 × 256 = 98,304,000` targets.

`checkpoint.pt` is the last training step; `best_validation.pt` is the best single checkpoint. Neither is the final gated model. Keep the full 12,000-step schedule: shortening it to 8400 changes the cosine learning-rate trajectory even though the original winner used checkpoints no later than that step.

### Step 2 — Checkpoint averaging

```powershell
.\.venv\Scripts\python.exe average.py --run-dir runs/manual-train --out runs/manual-average --device cuda
Get-Content runs/manual-average/selection.json
```

Read `selected.checkpoint`. The original run selected steps 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800 and 8400. Average compatible checkpoints from one trajectory only. A new run may select a different average size.

### Step 3 — Gate training

These PowerShell commands automatically use the selected average:

```powershell
$averageSelection = Get-Content runs/manual-average/selection.json -Raw | ConvertFrom-Json
$selectedParent = $averageSelection.selected.checkpoint
.\.venv\Scripts\python.exe train_gate.py --parent "$selectedParent" --out runs/manual-gate --device cuda --epochs 10 --lr 0.001
```

The parent must be an **ungated averaged backbone**, not `checkpoints/final.pt`. The script freezes the backbone and fits 161 MLP gate parameters on training text. Gate weight decay is 0.01, batch size 4096, cache theta 10 and window size 256. A learned constant is retained as a control. The selected MLP is saved as `runs/manual-gate/checkpoint.pt`. The original winner was epoch 9; do not blindly choose the last epoch.

### Step 4 — Validate, freeze and fully test

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/manual-gate/checkpoint.pt --split validation --device cpu --precision fp32 --threads 4 --output runs/manual-gate/validation_cpu.json
```

Choose candidates using validation only. Record hashes before inspecting test results:

```powershell
Get-FileHash student.py,model.py,common.py,evaluate.py,train.py,average.py,train_gate.py,configs/final.json,runs/manual-gate/checkpoint.pt -Algorithm SHA256 | Select-Object Path,Hash | ConvertTo-Json | Set-Content runs/manual-gate/FROZEN_HASHES.json
.\.venv\Scripts\python.exe measure_eval.py --checkpoint runs/manual-gate/checkpoint.pt --split test --device cpu --precision fp32 --threads 4 --output runs/manual-gate/test_cpu.json
```

The manual hash file is a record, not an automatic lock. On Linux/macOS, use `evaluate.py` for scoring and platform-appropriate hash tools; the one-command pipeline records hashes portably. Do not tune repeatedly against test feedback. Keep new weights under `runs/` rather than overwriting the frozen submission checkpoint.

## 8. PyCharm configuration

1. Open `FINAL_SUBMISSION` as the project root.
2. In the project's Python Interpreter settings, select the existing `<project-root>\.venv\Scripts\python.exe` on Windows, or `<project-root>/.venv/bin/python` elsewhere.
3. Under **Run → Edit Configurations**, add a Python configuration. Set its working directory to the project root, and choose the script and parameters below.
4. Run with the green button. Menu wording varies by PyCharm version; the necessary fields are interpreter, script, parameters and working directory.

| Purpose | Script | Parameters |
|---|---|---|
| Complete test | `evaluate.py` | `--checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/pycharm_test.json` |
| Validation | `evaluate.py` | `--checkpoint checkpoints/final.pt --split validation --device cpu --precision fp32 --threads 4 --output results/pycharm_validation.json` |
| Full training and test | `run_pipeline.py` | `--out runs/pycharm-reproduce --device cuda --precision bf16` |
| Windows resources | `measure_eval.py` | `--checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/pycharm_resources.json` |
| Backbone only | `train.py` | All arguments after `train.py` in Section 7, Step 1 |

For CPU training, change device and precision as in Section 6. Choose a fresh output directory for each run. `student.py` defines the model; it is **not** the training entry point. Create the local environment before selecting it in PyCharm; the submission does not include a portable prebuilt `.venv`.

## 9. Experiment history and evidence

| Completed historical stage | CPU FP32 full-test BPB |
|---|---:|
| Original classroom baseline | 2.101260129 |
| Squared ReLU, averaging and dynamic cache | 1.592946795 |
| Squared-ReLU recipe tuning | 1.589165536 |
| Independent RMSNorm variant | 1.580309452 |
| Independent RoPE variant | 1.566650584 |
| Final RoPE recipe tuning | **1.555309607** |

These stages are not one isolated ablation. The original baseline used a smaller width and one tenth of the backbone training targets. The report explains matched-target comparisons, cache/gate ablations, single-seed limitations and historical test exposure. Interrupted runs are not reported as completed test results.

| Record under `evidence/` | Contents |
|---|---|
| `final_training_metrics.json` | Final backbone configuration, losses and validation curve |
| `final_averaging.json` | Average candidates and selected steps |
| `final_gate_selection.json` | Fixed/constant/MLP cache comparisons and epoch selection |
| `architecture_comparison.json` | Matched backbone-target structure comparison |
| `tuning_comparison.json`, `tuning_records.csv` | Latest hyperparameter trials |
| `primer_ablation_comparison.json` | Earlier squared-ReLU and convolution experiments |
| `latest_search_costs.json`, `search_cost_audit.json` | Retained search costs and known interrupted runs |
| `reproduced_test_cpu_fp32.json` and `.resources.json` | Full test score and RAM reproduction |
| `reproduced_baseline_cpu_fp32.json` | Baseline score and timing |
| `cleanup_equivalence.json`, `package_checks.json` | Cleanup equivalence and retained verification results |

Absolute paths in historical JSON records are audit metadata, not dependencies for the supplied checkpoint. New runs generate their own logs. The latest search round processed 485,675,360 training targets across backbone and gate candidates; one final backbone run is not the entire search cost.

## 10. Troubleshooting

| Problem | Action |
|---|---|
| Python 3.12 or `py` not found | Install Python 3.12 or use its full executable path to create `.venv` |
| Missing dependency | Run pip through the same interpreter used to launch the project; check package versions |
| CUDA unavailable | Check torch build and NVIDIA driver, or use the CPU route |
| BF16 unsupported | Train with `--precision fp32`; evaluation stays CPU FP32 |
| CUDA out of memory | Close other GPU tasks or use CPU; changing batch/model size changes the reproduction recipe |
| Output directory exists | Use a new directory; the pipeline has no automatic interrupted-run resume |
| Pipeline raises `CalledProcessError` | Read the failing stage's log for the underlying error; retain logs before restarting in a new directory |
| `ModuleNotFoundError: common` | Run from the complete project root, with helper files present |
| Changed benchmark file | Restore original data; do not bypass checks |
| Missing/unexpected checkpoint keys | Use the matching final or baseline checkpoint; old experimental structures are not supported by the cleaned model |
| Package integrity mismatch | Compare with a fresh package; intentional source/document edits also change hashes |
| Quiet training console | Read the stage logs; the pipeline redirects detailed output there |

## 11. Attribution

WikiText-2 comes from [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext), with text contributed to Wikipedia. Upstream attribution identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and [GFDL](https://www.gnu.org/licenses/fdl-1.3.html). Original data revision: `b08601e04326c79dfdd32d625aee71d232d685c3`. Text was joined line-by-line as UTF-8; the tokenizer was fitted on training text only. Data hashes are in `data/manifest.json`. Preserve attribution and classroom files; no new license is assigned to the supplied course code. Method references are in [REPORT.md](REPORT.md).

## 12. AI usage disclosure

AI was used to research optimization methods, write and test code, compile and document experiment and test history, and assist with report writing.
