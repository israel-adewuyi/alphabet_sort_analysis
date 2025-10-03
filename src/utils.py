import os
import json
import torch
import logging
from torch import Tensor
from jaxtyping import Float, Int
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset() -> Dataset:
    dataset = load_dataset("israel-adewuyi/eval_data_alphabet_sort", split="train")

    return list(dataset["prompt"])[:32]

def compute_entropy(
    logits: Float[Tensor, "batch seq_len vocab_size"],
    labels: Int[Tensor, "batch seq_len"]
) -> float:
    """
    Computes the entropy across the unembedding of a LM
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    valid_mask = (shift_labels != -100)
    
    log_softmax = F.log_softmax(shift_logits, dim=-1)
    softmax_probs = torch.exp(log_softmax)
    
    # Per-position entropy
    per_position_entropy = -(softmax_probs * log_softmax).sum(dim=-1)
    
    # Mask and average
    num_valid_tokens = valid_mask.sum().item()
    masked_entropy = per_position_entropy * valid_mask.float()
    avg_entropy = torch.sum(masked_entropy) / num_valid_tokens
    
    return avg_entropy.item()

def compute_perplexity(
    logits: Float[Tensor, "batch seq_len vocab_size"], 
    labels: Int[Tensor, "batch seq_len"]
) -> float:
    """
    Computes the perplexity of a language model given logits and corresponding labels.
    
    Args:
        logits: Model output logits of shape (batch_size, sequence_length, vocab_size).
        labels: Target token indices of shape (batch_size, sequence_length).
    
    Returns:
        The computed perplexity as a float.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    valid_mask = (shift_labels != -100)
    
    # Clamp invalid labels to 0 to prevent out-of-bounds in gather
    shift_labels_clamped = torch.where(valid_mask, shift_labels, torch.tensor(0, device=shift_labels.device, dtype=shift_labels.dtype))
    
    # Compute log probs
    gathered_logits = shift_logits.gather(dim=-1, index=shift_labels_clamped.unsqueeze(-1)).squeeze(-1)
    log_probs = gathered_logits - torch.logsumexp(shift_logits, dim=-1)
    
    masked_log_probs = log_probs * valid_mask.float()
    num_valid_tokens = valid_mask.sum().item()
    
    if num_valid_tokens == 0:
        raise ValueError("No valid tokens found after masking.")
    
    avg_nll = -torch.sum(masked_log_probs) / num_valid_tokens
    perplexity = torch.exp(avg_nll)
    logger.info(f"Log probs shape: {log_probs.shape}, Valid tokens: {num_valid_tokens}")
    
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
) -> float:
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to("cuda")

    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    # Move other inputs to CUDA
    attention_mask = tokenized.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask} if attention_mask is not None else {"input_ids": input_ids}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    del inputs
    del input_ids
    
    logits = outputs.logits
    logger.debug(f"Logits shape: {logits.shape}")
    
    perplexity = compute_perplexity(logits, labels)
    logger.debug(f"Perplexity: {perplexity}")
    
    entropy = compute_entropy(logits, labels)
    
    return perplexity, entropy

def save_perplexities(perplexities: list, model_names: list) -> None:
    os.makedirs("artefacts", exist_ok=True)
    
    data = [
        {"model_name": model_name, "perplexity": perplexity}
        for model_name, perplexity in zip(model_names, perplexities)
    ]
    
    with open("artefacts/perplexities.json", "w") as f:
        json.dump(data, f, indent=4)
