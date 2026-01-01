# Evaluate Tab

The Evaluate tab lets you systematically benchmark your fine-tuned models using standardized tests. Instead of just chatting with your model to see if it "feels right," you can get objective scores that show exactly how well it performs.

## Why Evaluate?

After training a model, you might wonder: "Did my training actually improve the model?" The Evaluate tab answers this question with numbers.

**The workflow:**
1. **Train** → Teach your model with your data
2. **Inference** → Chat with it to see if it feels right
3. **Evaluate** → Get objective benchmark scores
4. **Publish** → Share with confidence

## What You'll See

![Evaluate Tab Screenshot](../../img/new/ff_evaluate.png)

The Evaluate tab has four sections:

### 1. Model Selection

- **Training run to evaluate**: Pick from your completed training runs
- **Base model**: Auto-filled based on your training run
- **Compare with base model**: Toggle this to see if fine-tuning helped

### 2. Benchmark Configuration

Choose which benchmark to run:

| Benchmark | Category | What It Tests | Speed |
|-----------|----------|---------------|-------|
| ⚡ HellaSwag | Quick | Commonsense reasoning | Fast |
| ⚡ TruthfulQA MC2 | Quick | Truthfulness and factual accuracy | Fast |
| ⚡ ARC Easy | Quick | Elementary science questions | Fast |
| ⚡ Winogrande | Quick | Pronoun resolution | Fast |
| ⚡ BoolQ | Quick | Yes/no questions | Fast |
| ARC Challenge | Full | Harder science questions | Medium |
| MMLU | Full | 57 different knowledge tasks | Slow |
| MMLU-PRO | Full | Enhanced MMLU with 10 choices | Slow |
| GSM8K | Full | Grade school math word problems | Medium |
| IFEval | Advanced | Instruction following | Medium |
| BBH | Advanced | 23 challenging reasoning tasks | Slow |
| GPQA | Advanced | PhD-level questions | Medium |
| MuSR | Advanced | Multistep soft reasoning | Medium |
| HumanEval | Advanced | Python code generation | Medium |

**Configuration options:**
- **Max samples**: How many test questions to run (default: 100). Lower = faster, higher = more accurate.
- **Batch size**: How many questions to process at once (default: 4). Lower if you run out of GPU memory.

### 3. Run Evaluation

Click **Run Evaluation** to start. You'll see:
- Progress indicator showing which benchmark is running
- Status messages as the model loads and processes questions

Click **Stop** to cancel a running evaluation.

### 4. Evaluation Results

After completion, you'll see:

**Metrics Table:**
| Metric | Score |
|--------|-------|
| Acc | 69.00% |
| Acc Norm | 72.50% |

**Visual Bars:**
- Green bar for Accuracy
- Gold bar for Normalized Accuracy (when available)

**With Comparison Mode enabled:**
| Metric | Fine-tuned | Base Model | Δ Change |
|--------|------------|------------|----------|
| Acc | 69.00% | 65.00% | +4.00% |

Green Δ = improvement, Red Δ = regression

## Step-by-Step Guide

### Quick Evaluation (5 minutes)

1. Go to **Evaluate** tab
2. Select your training run from the dropdown
3. Keep benchmark as "⚡ HellaSwag (quick)"
4. Keep max samples at 100
5. Click **Run Evaluation**
6. Wait for results (~2-3 minutes)

### Full Evaluation with Comparison (15+ minutes)

1. Go to **Evaluate** tab
2. Select your training run
3. Check "Also evaluate base model for comparison"
4. Select a benchmark (e.g., "TruthfulQA MC2")
5. Set max samples to 200-500 for more accurate results
6. Click **Run Evaluation**
7. Compare fine-tuned vs base model scores

## Understanding the Metrics

### Accuracy (acc)
The percentage of questions answered correctly. Simple and straightforward.

### Normalized Accuracy (acc_norm)
Accuracy adjusted for answer length bias. Some models prefer shorter or longer answers regardless of correctness. Normalized accuracy corrects for this.

**Which to use?**
- For HellaSwag, ARC, MMLU: Use **acc_norm** (standard practice)
- For TruthfulQA: Use **acc** (MC2 format)
- When in doubt: Report both

### Delta (Δ)
The difference between your fine-tuned model and the base model:
- **+5.00%**: Your fine-tuning improved performance by 5 percentage points
- **-2.00%**: Your fine-tuning decreased performance by 2 percentage points
- **0.00%**: No change

## Tips & Best Practices

### Start Small
- Use **quick benchmarks** (⚡) first
- Start with **100 samples** to get fast feedback
- Increase samples for final evaluation before publishing

### Memory Management
- FineFoundry automatically clears GPU memory between evaluations
- If you get out-of-memory errors, lower the **batch size**
- Close other GPU-intensive applications

### Choosing Benchmarks
- **General capability**: HellaSwag, MMLU
- **Truthfulness**: TruthfulQA
- **Reasoning**: ARC, BBH
- **Math**: GSM8K
- **Code**: HumanEval
- **Instruction following**: IFEval

### Interpreting Results

**Good signs:**
- Positive Δ on benchmarks related to your training data
- Maintained or improved scores on general benchmarks

**Warning signs:**
- Large negative Δ on general benchmarks (catastrophic forgetting)
- Very low scores overall (model may have overfit to training data)

## Troubleshooting

### "CUDA out of memory"
- Lower the batch size to 2 or 1
- Reduce max samples
- Close other applications using GPU

### Evaluation takes too long
- Use quick benchmarks (⚡)
- Reduce max samples to 50-100
- Avoid full MMLU (57 tasks) for quick tests

### Results seem wrong
- Ensure you selected the correct training run
- Check that the adapter path exists
- Try a different benchmark to verify

### Comparison mode not working
- Base model evaluation requires loading the model twice
- Ensure you have enough GPU memory
- Try without comparison mode first

## Technical Details

FineFoundry uses [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) under the hood. This is the same framework used by:
- HuggingFace Open LLM Leaderboard
- Academic research papers
- Industry model evaluations

The benchmarks are standardized, so your results are comparable to published model scores.

## Related

- [Training Tab](training-tab.md) — Train your model first
- [Inference Tab](inference-tab.md) — Chat with your model
- [Troubleshooting](troubleshooting.md) — General help
