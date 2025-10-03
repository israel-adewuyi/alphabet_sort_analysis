import os
import json
import torch
import logging
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from jaxtyping import Float, Int
from typing import List, Tuple, Optional
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_hf_dataset() -> Dataset:
    dataset = load_dataset("israel-adewuyi/eval_data_alphabet_sort", split="train")
    return list(dataset["prompt"])[0:2048:4]


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


def save_base_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[str],
    batch_size: int = 32
) -> None:
    """
    Save base model logits to disk batch by batch for memory efficiency.
    """
    base_logits_dir = "artefacts/base_logits"
    os.makedirs(base_logits_dir, exist_ok=True)
    
    logger.info("Saving base model logits to disk...")
    
    batch_count = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc="Saving base logits"):
        batch_file = os.path.join(base_logits_dir, f"batch_{batch_count}.pt")
        batch_count += 1
        
        # Check if batch file already exists
        if os.path.exists(batch_file):
            logger.debug(f"Batch {batch_count} already exists, skipping...")
            continue
        
        batch = dataset[i:i + batch_size]
        logits, labels = run_inference(model, tokenizer, batch)
        
        # Save logits and labels for this batch
        batch_data = {
            'logits': logits.cpu(),  # Move to CPU for storage
            'labels': labels.cpu()
        }
        
        torch.save(batch_data, batch_file)
        
        # Free GPU memory
        del logits, labels
    
    # Save metadata
    metadata = {
        'batch_size': batch_size,
        'total_batches': batch_count,
        'dataset_size': len(dataset)
    }
    
    with open(os.path.join(base_logits_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Saved {batch_count} batches of base logits to {base_logits_dir}")


def load_base_logits_batch(batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load base model logits for a specific batch.
    """
    base_logits_dir = "artefacts/base_logits"
    batch_file = os.path.join(base_logits_dir, f"batch_{batch_idx}.pt")
    
    if not os.path.exists(batch_file):
        raise FileNotFoundError(f"Base logits batch {batch_idx} not found at {batch_file}")
    
    batch_data = torch.load(batch_file, map_location='cuda')
    return batch_data['logits'].to('cuda'), batch_data['labels'].to('cuda')


def compute_partial_kl_divergence(
    base_logits: Float[Tensor, "batch seq_len vocab_size"],
    checkpoint_logits: Float[Tensor, "batch seq_len vocab_size"], 
    labels: Int[Tensor, "batch seq_len"]
) -> float:
    """
    Compute partial KL divergence contribution for a batch.
    KL(base || checkpoint) = sum(P * log(P/Q)) where P=base, Q=checkpoint
    """
    # Shift for causal language modeling
    base_shift_logits = base_logits[:, :-1, :].contiguous()
    checkpoint_shift_logits = checkpoint_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Validity mask
    valid_mask = (shift_labels != -100)
    
    # Convert to log probabilities for numerical stability
    base_log_probs = F.log_softmax(base_shift_logits, dim=-1)
    checkpoint_log_probs = F.log_softmax(checkpoint_shift_logits, dim=-1)
    
    # Convert base to probabilities
    base_probs = torch.exp(base_log_probs)
    
    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    kl_per_token = base_probs * (base_log_probs - checkpoint_log_probs)
    kl_per_position = kl_per_token.sum(dim=-1)  # Sum over vocabulary
    
    # Apply validity mask and sum
    masked_kl = kl_per_position * valid_mask.float()
    sum_masked_kl = torch.sum(masked_kl).item()
    
    return sum_masked_kl


def evaluate_with_kl_divergence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[str],
    batch_size: int = 32
) -> Tuple[float, float, float]:
    """
    Evaluates perplexity, entropy, and KL divergence over the full dataset in batches.
    """
    if batch_size is None:
        batch_size = len(dataset)
    
    total_sum_log_probs = 0.0
    total_sum_entropy = 0.0
    total_sum_kl = 0.0
    total_valid = 0
    
    batch_idx = 0
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating with KL"):
        batch = dataset[i:i + batch_size]
        
        # Get checkpoint model logits
        checkpoint_logits, labels = run_inference(model, tokenizer, batch)
        
        # Load corresponding base logits
        base_logits, base_labels = load_base_logits_batch(batch_idx)
        
        # Compute standard metrics
        sum_log_probs, sum_entropy, n_valid = compute_partial_metrics(checkpoint_logits, labels)
        
        # Compute KL divergence
        sum_kl = compute_partial_kl_divergence(base_logits, checkpoint_logits, labels)
        
        # Accumulate
        total_sum_log_probs += sum_log_probs
        total_sum_entropy += sum_entropy
        total_sum_kl += sum_kl
        total_valid += n_valid
        
        # Free memory
        del base_logits, checkpoint_logits, base_labels
        
        batch_idx += 1
    
    # Global averages
    avg_nll = -total_sum_log_probs / total_valid
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    entropy = total_sum_entropy / total_valid
    kl_divergence = total_sum_kl / total_valid
    
    logger.info(f"Full dataset: Perplexity {perplexity}, Entropy {entropy}, KL Divergence {kl_divergence}, Total valid tokens {total_valid}")
    
    return perplexity, entropy, kl_divergence


def save_metrics(
    perplexities: list,
    entropies: list,
    kl_divergences: list,
    model_names: list,
    name: str = "metrics"
) -> None:
    os.makedirs("artefacts", exist_ok=True)
    
    data = [
        {
            "model_name": model_name, 
            "perplexity": perplexity, 
            "entropy": entropy,
            "kl_divergence": kl_divergence
        }
        for model_name, perplexity, entropy, kl_divergence in zip(model_names, perplexities, entropies, kl_divergences)
    ]
    
    with open(f"artefacts/{name}.json", "w") as f:
        json.dump(data, f, indent=4)
