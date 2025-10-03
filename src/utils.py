import os
import json
import torch
import logging

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Int
from typing import List, Tuple
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset() -> Dataset:
    dataset = load_dataset("israel-adewuyi/eval_data_alphabet_sort", split="train")
    return list(dataset["prompt"])


def evaluate_full_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[str],
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluates perplexity and entropy over the full dataset in batches.
    """
    if batch_size is None:
        batch_size = len(dataset)
    
    total_sum_log_probs = 0.0
    total_sum_entropy = 0.0
    total_valid = 0
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        logits, labels = run_inference(model, tokenizer, batch)
        
        sum_log_probs, sum_entropy, n_valid = compute_partial_metrics(logits, labels)
        
        total_sum_log_probs += sum_log_probs
        total_sum_entropy += sum_entropy
        total_valid += n_valid
    
    # Global averages
    avg_nll = -total_sum_log_probs / total_valid
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    entropy = total_sum_entropy / total_valid
    
    logger.info(f"Full dataset: Perplexity {perplexity}, Entropy {entropy}, Total valid tokens {total_valid}")
    
    return perplexity, entropy


def compute_partial_metrics(
    logits: Float[Tensor, "batch seq_len vocab_size"], 
    labels: Int[Tensor, "batch seq_len"]
) -> Tuple[float, float, int]:
    """
    Computes partial contributions for global perplexity and entropy.
    """
    # Shift for causal language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Validity mask
    valid_mask = (shift_labels != -100)
    num_valid_tokens = valid_mask.sum().item()
    
    # Clamp for safe gathering
    shift_labels_clamped = torch.where(valid_mask, shift_labels, torch.tensor(0, device=shift_labels.device, dtype=shift_labels.dtype))
    
    # Log probs for perplexity
    gathered_logits = shift_logits.gather(dim=-1, index=shift_labels_clamped.unsqueeze(-1)).squeeze(-1)
    log_probs = gathered_logits - torch.logsumexp(shift_logits, dim=-1)
    masked_log_probs = log_probs * valid_mask.float()
    sum_masked_log_probs = torch.sum(masked_log_probs).item()
    
    # Entropy computation
    log_softmax = F.log_softmax(shift_logits, dim=-1)
    softmax_probs = torch.exp(log_softmax)
    per_position_entropy = -(softmax_probs * log_softmax).sum(dim=-1)
    masked_entropy = per_position_entropy * valid_mask.float()
    sum_masked_entropy = torch.sum(masked_entropy).item()
    
    logger.debug(f"Batch: Log probs shape {log_probs.shape}, Valid tokens: {num_valid_tokens}")
    
    return sum_masked_log_probs, sum_masked_entropy, num_valid_tokens


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
    
    return outputs.logits, labels


def save_perplexities(perplexities: list, model_names: list) -> None:
    os.makedirs("artefacts", exist_ok=True)
    
    data = [
        {"model_name": model_name, "perplexity": perplexity}
        for model_name, perplexity in zip(model_names, perplexities)
    ]
    
    with open("artefacts/perplexities.json", "w") as f:
        json.dump(data, f, indent=4)
