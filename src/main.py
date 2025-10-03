import logging
from utils import (
    load_hf_dataset, 
    load_model, 
    load_tokenizer, 
    run_inference,
    save_metrics,
    evaluate_full_dataset,
    save_base_logits,
    evaluate_with_kl_divergence
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "Qwen/Qwen2.5-0.5B-Instruct", 
    "israel-adewuyi/alphabet_sort_0.5B_s100",
    "israel-adewuyi/alphabet_sort_0.5B_s200",
    "israel-adewuyi/alphabet_sort_0.5B_s300",
    "israel-adewuyi/alphabet_sort_0.5B_s400",
    "israel-adewuyi/alphabet_sort_0.5B_s500",
    "israel-adewuyi/alphabet_sort_0.5B_s600",
]


def main():
    tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    dataset = load_hf_dataset()
    batch_size = 16
    
    # First pass: Save base model logits
    base_model_name = MODEL_NAMES[0]  # "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Loading base model: {base_model_name}")
    base_model = load_model(base_model_name)
    save_base_logits(base_model, tokenizer, dataset, batch_size=batch_size)
    del base_model
    
    # Second pass: Evaluate all models with KL divergence
    perplexities, entropies, kl_divergences = [], [], []
    
    for model_name in MODEL_NAMES:
        logger.info(f"Evaluating model: {model_name}")
        model = load_model(model_name)

        if model_name == base_model_name:
            # For base model, KL divergence with itself is 0
            perplexity, entropy = evaluate_full_dataset(model, tokenizer, dataset, batch_size=batch_size)
            kl_divergence = 0.0
        else:
            # For checkpoint models, compute KL divergence against base
            perplexity, entropy, kl_divergence = evaluate_with_kl_divergence(model, tokenizer, dataset, batch_size=batch_size)
        
        perplexities.append(perplexity)
        entropies.append(entropy)
        kl_divergences.append(kl_divergence)
        
        logger.info(f"Model: {model_name}, Perplexity: {perplexity}, Entropy: {entropy}, KL Divergence: {kl_divergence}")
        
        del model
    
    save_metrics(perplexities, entropies, kl_divergences, MODEL_NAMES, "updated_metrics")

if __name__ == "__main__":
    main()