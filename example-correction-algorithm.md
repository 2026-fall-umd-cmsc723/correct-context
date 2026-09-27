Here are both the complete PyTorch/Hugging Face data collection pipeline and the formal academic project proposal structure tailored for your course instructor.
💻 Part 1: Token Index & Alignment Data Collection Pipeline
This pipeline automates the process of identifying exactly where specific entities lie inside a tokenized sequence. It handles the critical challenge of token alignment (e.g., when a name tokenizes into multiple sub-word tokens or mismatched lengths) so your group can reliably swap the target slice of the inputs_embeds tensor.

```python
import torch
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional

class EntityTokenPipeline:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3-8B"):
        """
        Initializes the pipeline with a specific model's tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Ensure a padding token exists for batch processing if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def find_entity_token_range(self, text: str, entity: str) -> Optional[Tuple[int, int]]:
        """
        Locates the exact start and end token indices of a substring entity 
        within the fully tokenized context sequence.
        """
        # 1. Tokenize both the full text and the standalone entity
        full_encoding = self.tokenizer(text, return_offsets_mapping=True)
        tokens = full_encoding.input_ids
        offset_mapping = full_encoding.offset_mapping  # Map of (char_start, char_end)

        # 2. Find the character bounds of the target entity in the raw text
        char_start = text.find(entity)
        if char_start == -1:
            print(f"Warning: Entity '{entity}' not found in raw text.")
            return None
        char_end = char_start + len(entity)

        # 3. Align character offsets with token indices
        token_start_idx = None
        token_end_idx = None

        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Skip special tokens (like BOS) that map to (0,0) but aren't text
            if tok_start == 0 and tok_end == 0 and idx == 0:
                continue
                
            # Catch the token where the entity begins
            if token_start_idx is None and tok_start  Dict[str, torch.Tensor]:
        """
        Extracts token indices and prepares the structural mapping required 
        for an in-memory input embedding layer vector swap.
        """
        # Tokenize main sequence
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Locate indices for Entity A (to be replaced)
        indices_a = self.find_entity_token_range(text, entity_a)
        
        # Tokenize Entity B standalone to fetch its target replacement vectors
        inputs_b = self.tokenizer(entity_b, return_tensors="pt", add_special_tokens=False)
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "entity_a_slice": indices_a,
            "entity_b_ids": inputs_b.input_ids
        }

# ==========================================
# Example Usage Execution
# ==========================================
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EntityTokenPipeline("meta-llama/Meta-Llama-3-8B")
    
    # Sample execution context
    sample_context = "Draft an update for Jonathan regarding the project timeline."
    old_contact = "Jonathan"
    new_contact = "David"
    
    # Process sequence
    pipeline_data = pipeline.prepare_swap_tensors(sample_context, old_contact, new_contact)
    
    print(f"Context text: \"{sample_context}\"")
    print(f"Tokenized Sequence IDs: {pipeline_data['input_ids'][0].tolist()}")
    print(f"Target Swap Range for '{old_contact}': Token indices {pipeline_data['entity_a_slice']}")
    print(f"Replacement Token IDs for '{new_contact}': {pipeline_data['entity_b_ids'][0].tolist()}")
```

📝 Part 2: Academic Course Project Proposal Structure

# Course Project Proposal: NLP Graduate Research Seminars
**Project Title:** An Empirical Evaluation of Inference-Time Activation Steering vs. Parametric Model Editing for Dynamic Context and Entity Correction
**Group Members:** [Your Names]
**Target Model Family:** Open-Source Autoregressive Transformers (Llama-3-8B / Mistral-7B)

---

## 1. Introduction & Research Problem
Large Language Models (LLMs) often struggle with real-time factual updates or mid-context user corrections within runtime environments. Conventional strategies rely on expanding the text context window by prefixing corrections (e.g., "Correction: Send to X instead of Y"). However, this approach increases token overhead, computational costs, and relies heavily on the model's variable attention mechanism to override stale historical data successfully.

This project investigates **Dynamic Context Swapping** at the structural layers of the network. We explore whether programmatically intercepting and replacing token representation vectors at inference time allows a model to correctly handle changed entity relationships without requiring full historical reprocessing or prompt rewriting.

## 2. Core Methodology & Technical Strategy
Our team will build a modular evaluation framework to benchmark two primary methods of vector-space context manipulation:

### Method A: Inference-Time Activation Steering
Using PyTorch forward hooks, we will intercept the model's `inputs_embeds` layer. Using a token-alignment data pipeline, we will programmatically isolate the tensor index coordinates corresponding to an outdated entity and slice-swap them with the vector weights of a newly requested entity. We will evaluate execution efficiency when this modification is performed at the base embedding layer versus intermediate hidden states.

### Method B: Parametric Model Editing (ROME / MEMIT)
As a baseline control, we will leverage rank-one model editing algorithms (ROME) to execute permanent parameter weight shifts inside the model's Multi-Layer Perceptron (MLP) modules. This allows us to compare ephemeral runtime "steering" against structural modification of the underlying factual association base.

## 3. Evaluation Suite & Dataset Design
To test robustness, we will construct a programmatic testing dataset containing multi-turn dialogue scenarios focused on entity relationships (e.g., manager, teammate, vendor configurations). We will evaluate model success across three metrics:
1. **Downstream Relationship Resolution:** Does the model seamlessly transition to using accurate attributes and details of the newly injected entity?
2. **Grammatical/Pronoun Alignment:** Does swapping a masculine entity vector with a feminine entity vector trigger accurate token generation updates downstream (e.g., *he/him* to *she/her*) without explicit text guidance?
3. **Semantic Drift & Perplexity:** Does hard vector swapping inside a local slice break tensor stability or negatively impact surrounding contextual coherence?

## 4. Project Milestones & Timeline
* **Milestone 1:** Finalize data extraction pipeline and token alignment layer script.
* **Milestone 2:** Implement PyTorch activation interception hooks using Hugging Face transformers.
* **Milestone 3:** Run baseline generation evaluations on un-modified text sequences vs. hard-swapped hidden state layers.
* **Milestone 4:** Run comparative benchmarks against ROME parameter edits; aggregate performance analytics.
* **Milestone 5:** Compile comprehensive research data results, finalize documentation charts, and prepare class presentation templates.


