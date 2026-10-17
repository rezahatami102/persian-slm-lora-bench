# Persian SLM LoRA Benchmark

A small, reproducible benchmark of parameter-efficient fine-tuning (QLoRA) of small open language models on Persian natural language inference (NLI). Everything runs on a single 8 GB consumer laptop GPU.

- **Task:** 3-class NLI (entailment / neutral / contradiction) on [FarsTail](https://arxiv.org/abs/2009.08820)
- **Hardware:** NVIDIA RTX 5050 Laptop GPU (8 GB), Windows 11

## Results

Setup: LoRA on all linear projections, 4-bit NF4 quantization, 3 epochs, effective batch size 32, seeds 42/43/44. "±" is the sample standard deviation across seeds. Test metrics are for the checkpoint with the best validation Macro-F1.

| Model | LoRA r | LR | Seeds | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Peak VRAM (MiB) |
|:--|--:|--:|--:|:--|:--|:--|--:|
| Qwen2.5-0.5B | 4 | 2e-4 | 3 | 0.7750 ± 0.0016 | 0.8052 ± 0.0030 | 0.8026 ± 0.0038 | 2373 |
| Qwen2.5-0.5B | 16 | 1e-4 | 1 | 0.7723 | 0.8133 | 0.8119 | 2475 |
| Qwen2.5-0.5B | 16 | 2e-4 | 3 | 0.7876 ± 0.0070 | 0.8148 ± 0.0061 | 0.8124 ± 0.0075 | 2475 |
| Qwen2.5-0.5B | 64 | 2e-4 | 3 | 0.7877 ± 0.0109 | 0.8039 ± 0.0071 | 0.8011 ± 0.0066 | 2910 |
| Qwen2.5-1.5B | 16 | 2e-4 | 3 | 0.8675 ± 0.0049 | 0.8659 ± 0.0060 | 0.8652 ± 0.0059 | 3545 |
| Gemma-3-1B (pt) | 16 | 2e-4 | 3 | 0.8902 ± 0.0076 | 0.8943 ± 0.0024 | **0.8935 ± 0.0024** | 2473 |
| ParsBERT (bert-fa-base-uncased), full FT | – | 2e-5 | 3 | 0.7975 ± 0.0047 | 0.8084 ± 0.0109 | 0.8071 ± 0.0103 | 4074 |

## Low-data learning curve

LoRA r=16, lr 2e-4, 3 seeds per row. Each subset is a stratified random sample of the training set (about 727 / 1,816 / 3,632 / 7,266 examples); epochs were set to 12 / 6 / 4 / 3 so that smaller subsets are trained longer. Validation and test sets are always complete.

| Model | Train fraction | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|:--|--:|:--|:--|:--|
| Qwen2.5-0.5B | 10% | 0.6947 ± 0.0012 | 0.7283 ± 0.0119 | 0.7277 ± 0.0158 |
| Qwen2.5-0.5B | 25% | 0.7379 ± 0.0118 | 0.7696 ± 0.0060 | 0.7703 ± 0.0056 |
| Qwen2.5-0.5B | 50% | 0.7637 ± 0.0084 | 0.7922 ± 0.0073 | 0.7903 ± 0.0063 |
| Qwen2.5-0.5B | 100% | 0.7876 ± 0.0070 | 0.8148 ± 0.0061 | 0.8124 ± 0.0075 |
| Gemma-3-1B | 10% | 0.8063 ± 0.0212 | 0.8097 ± 0.0103 | 0.8104 ± 0.0110 |
| Gemma-3-1B | 25% | 0.8387 ± 0.0121 | 0.8455 ± 0.0098 | 0.8445 ± 0.0106 |
| Gemma-3-1B | 50% | 0.8674 ± 0.0053 | 0.8783 ± 0.0013 | 0.8779 ± 0.0013 |
| Gemma-3-1B | 100% | 0.8902 ± 0.0076 | 0.8943 ± 0.0024 | 0.8935 ± 0.0024 |

For reference, ParsBERT (full fine-tuning, 100% of the data) reaches 0.8071 ± 0.0103 test Macro-F1 and Qwen2.5-1.5B (100%) reaches 0.8652 ± 0.0059.

## Easy vs. hard test examples

The official FarsTail easy/hard partition is not distributed with the `farstail` package, so we reconstruct the idea of the original paper: a hypothesis-only classifier (character n-gram TF-IDF + logistic regression, trained on the training set; 52.1% test accuracy vs. 33% chance) is run on the test set. Examples it classifies correctly are "easy" (solvable from superficial hypothesis cues, n=814); the rest are "hard" (n=750). This is our own split, not the official one.

Accuracy, mean ± std over 3 seeds:

| Model | Easy acc | Hard acc | Gap (easy - hard) |
|:--|:--|:--|:--|
| ParsBERT (full FT) | 0.8501 ± 0.0154 | 0.7631 ± 0.0060 | 0.0870 ± 0.0095 |
| Qwen2.5-0.5B, r=16 | 0.8587 ± 0.0061 | 0.7631 ± 0.0056 | 0.0956 ± 0.0072 |
| Qwen2.5-0.5B, r=64 | 0.8632 ± 0.0047 | 0.7409 ± 0.0107 | 0.1223 ± 0.0073 |
| Qwen2.5-1.5B, r=16 | 0.8907 ± 0.0061 | 0.8360 ± 0.0114 | 0.0547 ± 0.0175 |
| Gemma-3-1B, r=16 | 0.9189 ± 0.0021 | 0.8667 ± 0.0027 | 0.0523 ± 0.0013 |

Full table: `results/difficulty_summary.md`.

ParsBERT is fully fine-tuned in bf16 without quantization, so its VRAM figure is not directly comparable to the QLoRA rows.

Peak VRAM is `torch.cuda.max_memory_allocated`, not total process memory.

## Findings

1. **Model family mattered more than size.** Gemma-3-1B (0.8935) > Qwen2.5-1.5B (0.8652) > Qwen2.5-0.5B (0.8124).
2. **LoRA rank 4 / 16 / 64 shows no clear effect** on Qwen2.5-0.5B. Differences (about 1 point) are comparable to seed noise. r=64 costs about 430 MiB more VRAM with no gain.
3. **Learning rate:** 2e-4 beat 1e-4 on validation Macro-F1 for Qwen2.5-0.5B (0.7876 over 3 seeds vs 0.7723 for a single seed). 2e-4 was then used for all models.
4. **Tokenizer efficiency** on the training set (prompt template included):

| Tokenizer | Mean tokens / example | Chars / token |
|:--|--:|--:|
| Qwen2.5 | 155.2 | 1.86 |
| Gemma-3 | 95.2 | 3.03 |

Gemma needs about 39% fewer tokens for the same Persian text. This is consistent with tokenizer efficiency contributing to its advantage, but this study does not isolate it: the model families also differ in pre-training data and architecture.
5. **Encoder baseline.** A fully fine-tuned ParsBERT (0.8071) is statistically indistinguishable from Qwen2.5-0.5B + LoRA (0.8124), and is 5.8 points below Qwen2.5-1.5B and 8.6 points below Gemma-3-1B. ParsBERT used common default hyperparameters (lr 2e-5, 4 epochs) and was not tuned.
6. **Larger models gain mostly on hard examples.** From Qwen2.5-0.5B to Gemma-3-1B, accuracy rises by 6.0 points on easy and 10.4 points on hard examples, and the easy-hard gap shrinks from about 0.10 to 0.05. This suggests they rely less on superficial hypothesis cues (with the caveat that the split is our own reconstruction and the label mix differs between groups: the easy group contains more "neutral" examples).
7. **Exploratory:** LoRA rank 64 had the lowest hard-example accuracy (0.741 vs 0.763 for r=16) and the largest gap, although overall accuracy differences between ranks were within noise. With 3 seeds and a post-hoc comparison, this should be treated as a hypothesis, not a result.
8. **Low-data regime.** Gemma-3-1B reaches 0.810 test Macro-F1 with 10% of the training data (about 727 examples), on par with Qwen2.5-0.5B (0.812) and fully fine-tuned ParsBERT (0.807), both trained on 100% of the data. With 50% of the data (0.878) it exceeds Qwen2.5-1.5B trained on 100% (0.865). The Gemma-vs-Qwen-0.5B gap stays at about 7-9 points at every data fraction, and each doubling of data yields less than the previous one (Gemma: +3.4, +3.3, +1.6 points).

## Method

- Classification head (`AutoModelForSequenceClassification`) + LoRA (`alpha = 2r`, dropout 0.05) on q/k/v/o/gate/up/down projections; the classification head is trainable.
- Input template: `Premise: {premise}\nHypothesis: {hypothesis}\nRelation:`, max length 448 tokens (no truncation in practice).
- 3 epochs, cosine schedule, 3% warmup, bf16, gradient checkpointing. Effective batch 32 (16x2 for 0.5B, 8x4 for 1.5B and Gemma).
- Metrics: accuracy and Macro-F1 (classes are balanced).

## Reproduce

Python `3.11.16`. Install PyTorch with CUDA support from pytorch.org (tested: torch 2.14.1, CUDA 13.0), then:

```bash
pip install -r requirements.txt
python scripts/cache_data.py            # downloads FarsTail once into data/
python scripts/train.py --smoke         # quick sanity check, results not saved
python scripts/train.py --model_id Qwen/Qwen2.5-0.5B --lora_r 16 --lr 2e-4 --seed 42
python scripts/run_queue.py             # batch of runs defined in JOBS
python scripts/aggregate.py             # mean ± std table -> results/summary.md
python scripts/make_difficulty_split.py
python scripts/predict.py
python scripts/analyze_difficulty.py
```

Gemma models are gated: create a free Hugging Face account, accept the model license on its page, and run `hf auth login`.

## Limitations

- Only 3 seeds per configuration; no significance tests.
- Runs are not bit-exact reproducible even with a fixed seed: an accidental duplicate of Qwen2.5-1.5B seed 44 gave test Macro-F1 0.8719 and 0.8655 (the first run is the one reported).
- One dataset; learning rate was tuned on Qwen2.5-0.5B only and reused for all models.
- Gemma was tested at a single size.
The encoder baseline (ParsBERT) uses untuned default hyperparameters.
- Easy vs. hard test subsets have not been analyzed yet.
- Test predictions for the difficulty analysis were regenerated from saved checkpoints; recomputed Macro-F1 matched the logged values within 0.005 for all runs. One Qwen2.5-1.5B seed-44 checkpoint belongs to the accidental duplicate run (0.8655), not the first run (0.8719) used in the main table.
- The easy/hard split is not the official one (see above).
- Low-data protocol: stratified random subsets whose composition depends on the seed (so the reported std includes subset variability); epochs were set to 12/6/4/3 for 10/25/50/100% of the data, giving about 276/342/456/684 optimizer steps, so data amount and number of updates covary; model selection used the full validation set, which is optimistic for a true low-data setting.
- Peak VRAM for the Qwen2.5-0.5B low-data rows used micro-batch 8x4 instead of 16x2, so it is not comparable to the full-data rows.

## Roadmap

- [x] Encoder baseline (ParsBERT / XLM-R)
- [x] Error analysis on easy vs. hard test subsets
- [x] Low-data learning curves
- [ ] Synthetic-data augmentation ablation
- [ ] Second task (question answering)

## References

- FarsTail: A Persian Natural Language Inference Dataset (arXiv:2009.08820)
- LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023)
- Qwen2.5 and Gemma 3 model families