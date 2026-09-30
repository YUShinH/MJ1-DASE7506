# MP1 Language Model Report

RoPE, squared ReLU and a causal adaptive cache on WikiText-2

The final model achieves **1.555310 test bits per byte (BPB)** and **25.823430 token perplexity** under the supplied CPU FP32 evaluator. It combines a small Transformer with rotary position embeddings, squared ReLU, checkpoint averaging and a learned within-window cache gate. This report presents the method, matched-budget comparisons, component ablations and the limits of the evidence. All results are from retained experiments; no new training was performed for this report.

## 1 Method

### 1.1 Task and evaluation protocol

MP1 trains a next-token language model from scratch on the supplied WikiText-2 text using the fixed BPE-2048 tokenizer. Evaluation follows protocol `7506-mp1-wt2-v2` in independent, causal windows of 256 tokens. BPB is total negative log likelihood in nats divided by the natural logarithm of two and the number of evaluated UTF-8 bytes. Lower is better. Training text fits parameters; validation selects candidates and checkpoints. The official data, tokenizer and evaluator are unchanged.

### 1.2 Final backbone

The backbone has four pre-LayerNorm Transformer blocks, hidden width 192, four attention heads and FFN width 768. Input embeddings and the output projection share weights. Squared ReLU, implemented as `relu(x).square()`, replaces the FFN activation following the Primer direction [1]. RoPE rotates Q and K using theta 10,000 and replaces learned absolute position embeddings [2]; V is not rotated. Dropout is zero. The model retains LayerNorm, with no RMSNorm, SwiGLU or Q/K/V convolution in the submitted implementation.

![Final model architecture](report_assets/architecture.png)

*Figure 1. The final backbone and cache mixture. All cache state is reconstructed within each input window.*

### 1.3 Causal cache and adaptive gate

The cache adapts the continuous-cache idea [3]. At position t, it compares the final normalized hidden state with earlier hidden states i < t using cosine similarity. Similarities are multiplied by 10 before softmax; the weights are accumulated at the already observed input token at i + 1. Thus it predicts the next token without reading that target. The first position uses the backbone alone; no state persists across windows or examples.

An 8-to-16-to-1 MLP with a Tanh hidden activation produces a sigmoid gate capped at 0.5. The final distribution is a convex mixture of the backbone and cache distributions. Its eight causal inputs summarize entropy, confidence, agreement, support and position. Feature standardization uses training data only. The backbone is frozen during gate fitting. The gate adds 161 parameters to 2,173,056 backbone parameters. Pointer Sentinel [4] motivates adaptive mixing, but this implementation is not a reproduction of its full architecture.

<!-- pagebreak -->

### 1.4 Training and model selection

The main training run uses seed 17, batch size 32 and 12,000 steps, processing **98,304,000 next-token targets**. Training uses GPU BF16 on an RTX 3060 Laptop GPU. Reported final validation and test scores use CPU FP32 with four threads. The original environment was Python 3.12 and PyTorch 2.7.1+cu126.

| Component | Final setting |
|---|---|
| Backbone optimizer | AdamW [5]; learning rate 0.001; weight decay 0.20 |
| Schedule and stability | Betas (0.9, 0.999); 100 warmup steps; cosine decay to 0.1 of initial LR; gradient clip 1 |
| Validation frequency | Every 600 steps; best single checkpoint at step 8,400 |
| Checkpoint average | 9 checkpoints at steps 3,600 to 8,400 inclusive, spaced by 600 |
| Gate optimizer | AdamW; LR 0.001; weight decay 0.01; batch size 4,096 |
| Gate selection | At most 10 epochs; best MLP at epoch 9; train-only features |

![Training and validation curves](report_assets/training.png)

*Figure 2. Recorded training loss and validation BPB for the final backbone. They have different units and should not be compared numerically.*

Training loss continues to decline while validation performance flattens and slightly worsens late in training. The last checkpoint is therefore not the selected model. Candidates average the most recent 1, 3, 5, 7 or 9 checkpoints ending at the best validation checkpoint, within the same trajectory. Nine points give the lowest validation BPB. This is offline arithmetic weight averaging, related to but not the complete SWA training procedure [6].

The total schedule must remain 12,000 steps when reproducing the run: shortening it to 8,400 changes the cosine learning-rate trajectory. After averaging, only the gate is trained. Both a learned constant gate and an MLP are fitted for comparison; validation selects their epochs. The final code and checkpoint are then frozen before the final round's full test evaluation.

### 1.5 Why these changes may help

Squared ReLU changes the FFN response to positive activations. RoPE makes attention sensitive to relative position through Q/K rotations. Averaging reduces dependence on one training iterate. A local cache can raise probability for tokens associated with similar recent contexts, while the gate adapts how strongly it is used. These are plausible mechanisms, not independent causal proofs: architecture, optimization, averaging and cache compatibility interact, as the comparisons below show.

<!-- pagebreak -->

## 2 Comparisons

### 2.1 Baseline and matched backbone training budget

The original classroom baseline has width 128, four layers and four heads. It processes 9,830,400 targets in 1,200 steps and obtains **2.101260 test BPB**. The final score is 25.98% lower, but capacity and backbone training targets both increase. This comparison measures the complete recipe change, not the isolated effect of an architectural choice.

Table 1 supplies the stronger control. All three runs use width 192, four layers, seed 17, 12,000 steps, batch 32, LR 0.001 and weight decay 0.15: **98,304,000 backbone targets each**. They compare the same averaging candidates in this round and fit their own cache gates. This matches backbone data exposure, not parameter count, wall-clock compute or the complete model-selection cost.

| Backbone | Averaged validation BPB without cache | Gated validation BPB | CPU test BPB |
|---|---|---|---|
| LayerNorm + squared ReLU | 1.618499 | 1.573540 | 1.589166 |
| RMSNorm + squared ReLU | 1.610461 | 1.567194 | 1.580309 |
| LayerNorm + RoPE + squared ReLU | 1.609339 | 1.548937 | 1.566651 |

*Table 1. Matched backbone-target comparison. Gated validation and full-test columns use CPU FP32. Source: architecture_comparison.json and relu2_tuning_comparison.json.*

RoPE performs best after averaging and gate adaptation. However, its best single checkpoint is worse than the LayerNorm control (1.641246 versus 1.640013 validation BPB). The result supports RoPE within the combined recipe, rather than a universal advantage for the attention change alone. RMSNorm [7] improves prediction but did not accelerate the recorded local training run.

### 2.2 Hyperparameter selection on the RoPE backbone

The next round extends averaging candidates to nine checkpoints and compares the recipes below on validation. Each backbone still receives the same target budget; the control reuses its earlier training trajectory. All rows use gate LR 0.001.

| Backbone LR | Weight decay | Nine-point average validation BPB | Gated validation BPB |
|---|---|---|---|
| 0.0010 | 0.15 | 1.607561 | 1.545741 |
| 0.0008 | 0.15 | 1.605033 | 1.548585 |
| **0.0010** | **0.20** | **1.601673** | **1.539104** |

*Table 2. Validation-only recipe selection. These are development-device results; final CPU verification is in Section 4. Values are rounded to six decimals.*

Lowering backbone LR improves the averaged backbone slightly but worsens the gated result. Weight decay 0.20 wins as a complete recipe. On that parent, gate LR 0.0005 gives 1.540065 validation BPB, worse than 0.001. The final full-test score improves by 0.011341 BPB (0.72%) over the preceding RoPE model. There is no multi-seed evidence that this small gain is statistically stable.

Earlier completed experiments found 1.592947 test BPB for squared ReLU with averaging and dynamic cache, versus 1.593824 for the independently tested Q/K/V convolution direction. Their difference is small. The combined ReLU/convolution run was interrupted and supplies no complete final comparison. SwiGLU was also explored but not retained. These attempts do not establish that either family is generally inferior.

<!-- pagebreak -->

## 3 Ablations

### 3.1 Checkpoint averaging on one training trajectory

For the final RoPE backbone with weight decay 0.20, averaging 1, 3, 5, 7 and 9 checkpoints gives validation BPB of **1.633417, 1.611557, 1.605323, 1.602157 and 1.601673**, respectively. The nine-point model improves on the best single checkpoint by 0.031744 BPB without additional backbone training. Improvement from seven to nine points is only 0.000483; its robustness is untested.

![Averaging and cache ablations](report_assets/ablation.png)

*Figure 3. Validation ablations. Left: number of averaged checkpoints. Right: learned constant and dynamic gate candidates on the same frozen nine-point parent.*

### 3.2 Cache and gate ablation on a frozen parent

Table 3 holds the nine-point parent, cache similarity rule, theta and window size fixed. The first two variants fit no additional gate parameters. The learned constant and MLP each fit only their gate on training features, for at most ten epochs with the same batch size and optimizer recipe; epochs are chosen by validation.

| Variant | Validation BPB | Change from no cache |
|---|---|---|
| No cache | 1.601673 | Reference |
| Fixed cache weight 0.05 | 1.560950 | -0.040723 |
| Learned constant gate, best epoch 1 | 1.565585 | -0.036088 |
| Dynamic MLP gate, best epoch 9 | **1.539104** | **-0.062570** |

*Table 3. Key-mechanism ablation from final_gate_selection.json. Validation calculations use the same cached parent features. They are not additional full-test results.*

A cache already helps with a fixed weight. The MLP further improves over the fixed weight by 0.021847 BPB and over the learned constant by 0.026481. This supports input-dependent mixing under the tested training recipe. The trained constant is worse than the fixed 0.05 weight, so learning a gate is not automatically beneficial; its training optimum may transfer imperfectly to validation.

The selected MLP's mean cache weight is about 0.0548 and its median about 0.0134, showing that the mixture varies across positions. This is descriptive evidence, not proof that every large gate value is useful. Feature-removal ablations and multiple seeds would be needed to identify which gate inputs matter reliably.

These are sequential ablations, not a full factorial study. Their gains should not be extrapolated to other backbones or added to historical improvements as if each were independent. In particular, the record lacks a final-RoPE comparison that changes only squared ReLU back to GELU under the complete final recipe.

<!-- pagebreak -->

## 4 Critical analysis

### 4.1 Prediction quality and resource trade-offs

The cleaned submission was evaluated again using the official CPU FP32 scorer. Its predictions match the original frozen implementation, and the complete test score is unchanged. Twelve submission checks cover properties including causality, window independence, probability normalization and frozen-backbone gate training. These checks support implementation correctness; they do not establish statistical significance.

| Measurement | Final submission | Requirement or reference |
|---|---|---|
| Validation BPB | 1.539103594 | Lower is better |
| Full-test BPB / token PPL | 1.555309607 / 25.823429863 | 428,405 targets; 1,292,013 UTF-8 bytes |
| CPU scoring time | 59.448 s | Baseline 21.265 s; ratio 2.796; limit 5 |
| Peak process RAM | 1.812 GiB | Limit 4 GiB |
| Uncompressed inference assets | 9.821 MiB | Checkpoint plus model source; limit 64 MiB |

*Table 4. Clean-package resource reproduction. Sources: reproduced_test_cpu_fp32.json, its resources file, reproduced_baseline_cpu_fp32.json and package_checks.json. Timing depends on machine load.*

The cache improves predictive quality but adds similarity computation, vocabulary accumulation and gating during scoring. The final model remains inside all three measured limits, with less time headroom than the classroom baseline. RMSNorm's theoretical simplicity did not translate into measured speed gains here; efficiency claims must be checked on the actual implementation and target hardware.

### 4.2 Search cost and unsuccessful candidates

The selected backbone's recorded process time is 553.47 s; its averaging stage takes 25.21 s and its constant/MLP gate search 121.43 s. These figures describe stages, not the entire project cost. The latest complete tuning round takes about **1,977.73 s** and processes **485,675,360 training targets**, including candidates not selected.

That target count comprises two new backbone runs (196,608,000 targets) and four gate searches (289,067,360 targets). Each gate search fits two gate types for ten epochs over 3,613,342 targets per epoch. Gate-only examples do not have the same compute cost as full backbone updates. Reused control training is not counted as new training in this round. Earlier architecture search took about 1,791.60 s and the preceding ReLU tuning round about 1,730.98 s. The audit also retains earlier completed and known interrupted runs; these selected round totals are not an exhaustive project time ledger.

### 4.3 Limits of the conclusions

The main comparisons use one seed and one benchmark; no confidence intervals or seed-level variability are available. Backbone target counts are matched in Table 1, but model sizes, run times and downstream selection are not identical. The 25.98% baseline gain mixes capacity, training duration and method changes. Architecture and cache interactions further limit isolated attribution.

Several historical stages were fully tested. Although the final tuning round used validation for its candidate choices and froze before testing, the test set is not an untouched holdout across the whole project. Reported results are observed benchmark performance, not an unbiased estimate after a fresh evaluation. Future experiments should be specified in advance, selected on validation and assessed without repeated test-driven revision.

<!-- pagebreak -->

### 4.4 Improvements worth testing next

The first priority is repeating the fixed recipe with several seeds and reporting mean validation BPB and variability. Next, narrowly sweep weight decay around 0.20 and the learning rate while holding the number of backbone targets fixed. A gate feature-removal study and alternative averaging intervals can test which parts of the final recipe are necessary. Every candidate should record CPU time and full search cost. Joint RoPE/RMSNorm or convolution experiments remain hypotheses, not demonstrated improvements, and must be evaluated within the same resource limits.

### 4.5 Reproducibility and repository contents

The report accompanies the frozen checkpoint in the code repository. `student.py` defines the final model; `train.py`, `average.py` and `train_gate.py` implement its stages; `run_pipeline.py` runs the complete process. `configs/final.json` fixes the backbone configuration. The checkpoint stores the final inference configuration, and the official `evaluate.py`, `common.py`, `model.py`, tokenizer and data remain unchanged. Evidence JSON/CSV files are included under `evidence/`; figure assets are under `report_assets/`.

From the repository root, with the project Python environment active, score the supplied checkpoint without retraining:

```text
python evaluate.py --checkpoint checkpoints/final.pt --split test
  --device cpu --precision fp32 --threads 4 --output results/final_test.json
```

Join the two lines into one command. To train, average, fit the gate, freeze and evaluate a new run, use:

```text
python run_pipeline.py --out runs/reproduce-s17 --device cuda --precision bf16
```

Use a new output directory. The exact installation and manual stage instructions are in `README.md`. Retraining can vary with device and numerical behavior; the reported score belongs to the supplied frozen checkpoint. Its SHA256 is:

```text
7c53ebdcf91ea086c6916a082c3f01cdb3278747cf050ac3bba8cac0cb5998cf
```

### References

[1] So et al. (2021). *Primer: Searching for Efficient Transformers for Language Modeling*. https://arxiv.org/abs/2109.08668

[2] Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. https://arxiv.org/abs/2104.09864

[3] Grave et al. (2017). *Improving Neural Language Models with a Continuous Cache*. https://arxiv.org/abs/1612.04426

[4] Merity et al. (2016). *Pointer Sentinel Mixture Models*. https://arxiv.org/abs/1609.07843

[5] Loshchilov and Hutter (2017). *Decoupled Weight Decay Regularization*. https://arxiv.org/abs/1711.05101

[6] Izmailov et al. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization*. https://arxiv.org/abs/1803.05407

[7] Zhang and Sennrich (2019). *Root Mean Square Layer Normalization*. https://arxiv.org/abs/1910.07467

**AI disclosure:** Codex was used to assist with method research, code changes and report writing.
