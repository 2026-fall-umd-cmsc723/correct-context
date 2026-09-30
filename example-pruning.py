import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Using a code-specialized or highly capable local model
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)

# --- Define a Mock HumanEval Coding Task with a Context Dead End ---
problem_desc = "Write a python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome."
dead_end_context = "\n# DEAD END: Unrelated boilerplate data structures\n" + "\n".join([f"class DataNode{i}:\n    def __init__(self):\n        self.val = {i}" for i in range(25)]) 
code_prefix = "\n\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Check if s is a palindrome.\"\"\"\n"

# Full text sequence
full_prompt = problem_desc + dead_end_context + code_prefix

# Track exact token spans to know what to prune
tokens_desc = tokenizer(problem_desc, return_tensors="pt")["input_ids"].shape[1]
tokens_dead = tokenizer(dead_end_context, return_tensors="pt")["input_ids"].shape[1]

# Index calculations for pruning
start_prune_idx = tokens_desc
end_prune_idx = tokens_desc + tokens_dead

inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

def prune_kv_cache(past_key_values, start_idx, end_idx):
    pruned_kv = []
    for layer in past_key_values:
        k_state, v_state = layer
        # Dim 2 is the sequence length dimension in HuggingFace models
        k_pruned = torch.cat([k_state[:, :, :start_idx, :], k_state[:, :, end_idx:, :]], dim=2)
        v_pruned = torch.cat([v_state[:, :, :start_idx, :], v_state[:, :, end_idx:, :]], dim=2)
        pruned_kv.append((k_pruned, v_pruned))
    return tuple(pruned_kv)

def run_coding_benchmark(prune=False):
    # Prefill Phase
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    if prune:
        past_key_values = prune_kv_cache(past_key_values, start_prune_idx, end_prune_idx)
    
    # Generation Phase (Decode)
    current_input_ids = torch.tensor([[tokenizer.eos_token_id]], device=device) 
    generated_tokens = []
    
    start_time = time.time()
    for _ in range(50): # Generate up to 50 tokens of code
        with torch.no_grad():
            outputs = model(current_input_ids, past_key_values=past_key_values, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            # Stop if model outputs stop token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            current_input_ids = next_token
            past_key_values = outputs.past_key_values
            
    elapsed = time.time() - start_time
    tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
    code_output = tokenizer.decode(generated_tokens)
    
    return elapsed, tokens_per_sec, code_output

# Run both cases
time_base, speed_base, code_base = run_coding_benchmark(prune=False)
time_prune, speed_prune, code_prune = run_coding_benchmark(prune=True)

print(f"=== Baseline ===\nTime: {time_base:.2f}s | Speed: {speed_base:.2f} tok/s\nGenerated Code:\n{code_base}\n")
print(f"=== Pruned ===\nTime: {time_prune:.2f}s | Speed: {speed_prune:.2f} tok/s\nGenerated Code:\n{code_prune}\n")
