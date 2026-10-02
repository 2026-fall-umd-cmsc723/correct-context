# Do Past Mistakes Haunt Transformers? Investigating and Intervening on Erroneous KV Cache Entries in Mathematical Reasoning

**Authors:** Amey Varhade, Despoina [Last Name], Dileesha [Last Name], Joshua [Last Name], Alexander Medeiros

---

## 1. Introduction

Autoregressive language models generate text one token at a time, and every generated token remains permanently visible to the model through the key–value (KV) cache. This design choice has a subtle but important consequence: when a model produces an erroneous intermediate step during chain-of-thought reasoning, the faulty tokens stay in the cache and continue to influence all subsequent generation. Even if the model later "self-corrects" at the text level (for instance, by writing "wait, that was wrong"), the erroneous key and value vectors remain in the cache and are still attended to during every future forward pass.

This raises a natural question: **does retaining erroneous KV cache entries actually harm downstream accuracy, or are models robust to their own mistakes?**

If the answer is yes—if stale errors poison future computation—then there is a strong motivation for developing mechanisms that actively manage the model's working memory. Recent work on context management has explored this space from several angles. Fan et al. (2025) show that training small transformers to erase completed sub-computations from context can dramatically improve length generalization on combinatorial problems. Yan et al. (2026) demonstrate that compressing finished reasoning segments into summaries preserves performance while bounding context length. Meanwhile, Stoehr et al. (2025) show that KV cache entries can be surgically edited at inference time to steer frozen model behavior.

Our project takes a complementary, empirically-driven approach. Rather than proposing a new context management architecture, we focus on *measuring* the causal effect that erroneous KV entries have on downstream generation quality, and then testing whether simple, training-free cache interventions can mitigate those effects. We frame this around mathematical reasoning tasks, where (a) errors are frequent and clearly identifiable, (b) ground-truth solutions provide unambiguous correctness signals, and (c) intermediate steps have well-defined logical dependencies.

Our contributions are as follows:

1. A controlled experimental framework for injecting, localizing, and removing errors in the KV cache during chain-of-thought reasoning.
2. A causal analysis comparing model behavior across clean runs, error-injected runs where the model self-corrects, and error-injected runs where it does not.
3. An evaluation of training-free KV cache interventions—masking, removal, and replacement—and their effect on final-answer accuracy and downstream perplexity.

## 2. Related Work

### Context management and engineering

The recent survey by Agarwal and Khashabi (2025) categorizes context engineering into three families: context retrieval (RAG-style), context processing (in-model modification), and context management (external orchestration). Our work falls squarely into context processing: we modify the model's internal representations during inference rather than rewriting the textual prompt. Chen et al. (2025) take this idea further by training the language model itself to manage its own context, treating context modification as a learned capability rather than an external intervention.

### Reasoning trace compression

Several recent methods reduce the length of reasoning traces to improve efficiency and length generalization. PENCIL (Fan et al., 2025) trains transformers from scratch to use special reduction tokens that erase completed sub-computations, achieving polynomial context usage on problems that would otherwise require exponential space. InftyThink (Yan et al., 2026) fine-tunes models to summarize finished reasoning segments, replacing detailed intermediate steps with compressed summaries. LightThinker (Zhang et al., 2025) takes a similar iterative compression approach. These methods all modify the *text-level* context; our work instead operates directly on the *KV cache representations*, which lets us intervene without retraining.

### KV cache pruning and eviction

A separate line of work focuses on which KV entries to keep during long-context generation. Dynamic Context Pruning (Anagnostidis et al., 2025) retains only question-relevant key–value pairs, while Think Clearly (Zhou et al., 2025) proposes keeping only tokens that contribute to the final answer. Token Skipping (Chen et al., 2025) accelerates inference by selectively removing tokens from the cache. These methods optimize for efficiency; we focus instead on whether *error-specific* pruning improves correctness.

### KV cache steering and error correction

Stoehr et al. (2025) demonstrate that editing KV cache entries in frozen models can steer generation behavior, providing a mechanism for inference-time control without parameter updates. Concurrent work on inference-time error correction via KV cache steering (Zhang and Huang, 2025) directly manipulates cache entries to correct factual errors. MEMENTO (Guo et al., 2025) learns to manage memory over long reasoning horizons. Our project draws on these techniques but asks a more foundational question: does removing erroneous cache entries actually help, or are models already robust to the noise?

## 3. Our Approach

We structure our investigation around two research questions, each corresponding to a phase of experiments.

### Question 1: Are Past Mistakes Harmful?

We construct a controlled experimental setup with three conditions:

1. **Clean run.** The model generates a correct chain-of-thought solution from scratch.
2. **Error-injected, self-corrected.** We inject an erroneous intermediate step into the generation (either by prompting or by forcing specific tokens), then allow the model to self-correct in text (e.g., "wait, that step was wrong, let me redo it").
3. **Error-injected, uncorrected.** The same injection, but the model continues without textual self-correction.

For each condition, we run generation both with the original KV cache intact and with the erroneous entries masked or removed. This gives us a 3 × 3 experimental grid:

| **Cache**         | **Clean** | **Self-Corrected**      | **Uncorrected**         |
|-------------------|-----------|-------------------------|-------------------------|
| Original KV       | baseline  | error condition         | error condition         |
| Pruned KV         | control   | intervention condition  | intervention condition  |
| Random KV removal | control   | noise control           | noise control           |

*Table 1: Experimental grid. Each cell measures final-answer accuracy, downstream perplexity, and attention weight distribution over erroneous vs. correct cache entries.*

The random-removal condition is critical: if removing *any* span of tokens helps equally well, then the benefit is due to shorter context rather than error-specific cleanup.

We measure:

- **Final-answer accuracy** on the math problem.
- **Downstream perplexity** of tokens generated after the error region.
- **Attention analysis**: do later tokens still attend to the erroneous entries even after textual self-correction?

### Question 2: Can Training-Free Interventions Help?

If Question 1 reveals that errors are indeed harmful, we evaluate three training-free KV cache interventions:

1. **Hard removal.** Physically delete the KV entries corresponding to the erroneous span, analogous to "ripping out a page from a sketchbook."
2. **Attention masking.** Zero out attention weights to the erroneous entries without removing them, preserving positional information.
3. **Summary replacement.** Replace the erroneous KV entries with entries derived from a short textual summary of the error (e.g., "the previous attempt at step 3 was wrong"), inspired by the sketchbook metaphor from our brainstorming discussions.

We compare these interventions against the clean-run baseline and against each other, measuring both accuracy recovery and computational cost.

### Baselines

Our primary baselines are: (1) standard greedy decoding with no intervention (the default behavior), (2) majority-vote self-consistency, where we sample multiple completions and take the majority answer, and (3) the "just add 'wait'" baseline, where we append a pause token to encourage the model to reconsider without any cache manipulation. These baselines are trivial to implement and represent the simplest possible approaches to improving reasoning accuracy.

### Justification of compute

We will work with small open-weight models in the 1–7B parameter range (e.g., Qwen2.5-Coder-7B, Llama-3-8B) that fit on free-tier Colab or Kaggle GPUs. Our experiments do not require any fine-tuning or training: all interventions are inference-time cache manipulations. The most expensive operation is the prefill pass for each problem, which is well within the memory budget of a single A100 or T4 GPU. Several group members also have access to Apple Silicon machines with MPS support for local prototyping, as demonstrated in our initial KV cache pruning experiments.

### Schedule

We plan to work collaboratively on all milestones, with individual leads noted below.

1. **Weeks 1–2: Data and infrastructure** (Amey, Alex). Select math benchmarks (GSM8K, MATH subsets). Implement error injection pipeline: methods for forcing erroneous tokens mid-generation. Build KV cache extraction and manipulation utilities (building on our existing `example-pruning.py` prototype).
2. **Weeks 3–5: Question 1 experiments** (Dileesha, Despoina). Run the 3 × 3 experimental grid across at least two model families. Collect attention maps and perplexity measurements. Perform statistical analysis of whether error removal significantly differs from random removal.
3. **Weeks 5–7: Question 2 interventions** (Josh, Alex). Implement hard removal, attention masking, and summary replacement interventions. Benchmark against baselines on the same problem sets.
4. **Weeks 7–8: Analysis and error diagnosis** (All). Deep-dive into failure cases. Characterize which types of errors are most harmful and which interventions are most effective.
5. **Weeks 9–10: Report and presentation** (All). Write the final report. Prepare figures, tables, and the class presentation.

## 4. Data

We will use publicly available mathematical reasoning benchmarks:

- **GSM8K** (grade school math): 8,500 problems with step-by-step solutions. We will use the test split (1,319 problems) for evaluation and a subset of the training split for developing our error injection pipeline.
- **MATH** (competition-level): We will select subsets at difficulty levels 1–3 to ensure our small models can solve a meaningful fraction of problems cleanly, providing a sufficient number of "clean run" baselines.

No manual annotation is required. Error injection is performed programmatically by (a) corrupting numeric values in intermediate steps, (b) forcing the model to generate a known-wrong token at a specific position, or (c) prompting the model with a deliberately flawed partial solution. All datasets are freely available and commonly used in the reasoning evaluation literature.

## 5. Tools

We plan to use the following libraries:

- **Hugging Face Transformers**: Model loading, tokenization, and inference. We rely on the `DynamicCache` API for direct KV cache manipulation, as demonstrated in our prototype code.
- **PyTorch**: Tensor operations for KV cache slicing, masking, and replacement. Forward hooks for attention weight extraction.
- **BauKit / TransformerLens** (if needed): Higher-level tools for activation inspection and intervention, particularly for attention analysis.
- **Weights & Biases or TensorBoard**: Experiment tracking and metric logging across the experimental grid.

The KV cache manipulation utilities (pruning, masking, summary injection) will be implemented from scratch, building on our existing prototype. Model loading, tokenization, and standard inference will use off-the-shelf Hugging Face APIs. We do not plan to train any models; all experiments are inference-time interventions on frozen, pre-trained weights.

## References

- Agarwal, O. and Khashabi, D. (2025). "Context Engineering for Language Models: A Survey." *arXiv:2507.13334*.
- Anagnostidis, S., Pavllo, D., Biggio, L., Noci, L., Lucchi, A., and Hofmann, T. (2025). "Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers." *arXiv:2601.07994*.
- Chen, M., Zhou, C., Fan, A., and Levy, O. (2025). "Context Language Models: Learning to Self-manage Context." *arXiv:2609.37725*.
- Chen, Y., Zhang, J., Chen, A., Li, J., Guo, R., and Tan, X. (2025). "Token Skipping: Accelerating Inference for Long-context LLMs via Selective Token Removal." *ACM TIST*.
- Fan, Y., Gemp, I., Hu, L., De, A., Pacchiano, A., Singh, A., and Kakade, S. (2025). "PENCIL: Reasoning via Subcomputation Reduction with Erasable Context." *ICML 2025*.
- Guo, S., Akyurek, E., Schuurmans, D., and Andreas, J. (2025). "MEMENTO: Learning to Manage Memory for Long-horizon Reasoning." *ICLR 2025*.
- Stoehr, N., Filandrianos, G., Smolensky, P., and Gao, J. (2025). "KV Cache Steering for Controlling Frozen LLMs." *arXiv:2507.08799*.
- Yan, J., Ren, J., Li, Z., Lv, X., Hou, L., and Li, J. (2026). "Thinking with Bounded Context: Efficient Long-chain Reasoning with Segment-wise Summarization." *ICLR 2026*.
- Zhang, J., Chen, Y., Zheng, N., Zhang, X., and Chen, H. (2025). "LightThinker: Thinking Step-by-Step Compression." *arXiv:2502.15589*.
- Zhang, W. and Huang, K. (2025). "Inference-Time Error Correction via KV Cache Steering." *arXiv:2604.18567*.
- Zhou, Z., Zhao, B., Lyu, K., and Wang, Z. (2025). "Think Clearly, Speak Precisely: Reasoning Trace Pruning for Autoregressive Language Models." *arXiv:2507.08806*.
