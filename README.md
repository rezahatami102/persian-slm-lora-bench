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
```

Gemma models are gated: create a free Hugging Face account, accept the model license on its page, and run `hf auth login`.

## Limitations

- Only 3 seeds per configuration; no significance tests.
- Runs are not bit-exact reproducible even with a fixed seed: an accidental duplicate of Qwen2.5-1.5B seed 44 gave test Macro-F1 0.8719 and 0.8655 (the first run is the one reported).
- One dataset; learning rate was tuned on Qwen2.5-0.5B only and reused for all models.
- Gemma was tested at a single size.
The encoder baseline (ParsBERT) uses untuned default hyperparameters.
- Easy vs. hard test subsets have not been analyzed yet.

## Roadmap

- [x] Encoder baseline (ParsBERT / XLM-R)
- [ ] Error analysis on easy vs. hard test subsets
- [ ] Synthetic-data augmentation ablation
- [ ] Second task (question answering)

## References

- FarsTail: A Persian Natural Language Inference Dataset (arXiv:2009.08820)
- LoRA (Hu et al., 2021) and QLoRA (Dettmers et al., 2023)
- Qwen2.5 and Gemma 3 model families