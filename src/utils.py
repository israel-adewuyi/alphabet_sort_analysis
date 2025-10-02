import os
import json
import torch
import logging
from torch import Tensor
from jaxtyping import Float

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset() -> Dataset:
    dataset = load_dataset("israel-adewuyi/eval_data_alphabet_sort", split="train")

    return list(dataset["prompt"])[:64]

def compute_perplexity(
    logits: Float[Tensor, "batch seq_len vocab_size"], 
    labels: Float[Tensor, "batch seq_len"]
) -> float:
    """
    Computes the perplexity of a language model given logits and corresponding labels.
    
    Args:
        logits: Model output logits of shape (batch_size, sequence_length, vocab_size).
        labels: Target token indices of shape (batch_size, sequence_length).
    
    Returns:
        The computed perplexity as a float.
    """
    # Shift logits and labels for causal language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute log probabilities using logsumexp for numerical stability
    gathered_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    log_probs = gathered_logits - torch.logsumexp(shift_logits, dim=-1)
    
    logger.debug(f"Log probs shape: {log_probs.shape}")
    
    # Average negative log likelihood (cross-entropy loss)
    avg_nll = -torch.mean(log_probs)
    
    # Exponentiate to obtain perplexity
    perplexity = torch.exp(avg_nll)
    
    return perplexity.item()

def load_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)

def load_model(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return model

def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: list
) -> None:
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    logger.debug(f"Logits shape: {logits.shape}")
    
    perplexity = compute_perplexity(logits, labels)
    logger.info(f"Perplexity: {perplexity}")
    
    return perplexity

def save_perplexities(perplexities: list, model_names: list) -> None:
    os.makedirs("artefacts", exist_ok=True)
    
    data = [
        {"model_name": model_name, "perplexity": perplexity}
        for model_name, perplexity in zip(model_names, perplexities)
    ]
    
    with open("artefacts/perplexities.json", "w") as f:
        json.dump(data, f, indent=4)

"""
- Load model
- Load dataset
- run inference
- Get logits
- compute perplexity
- save to json
"""

# dataset = load_hf_dataset()
# model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
# run_inference(model, tokenizer, dataset)