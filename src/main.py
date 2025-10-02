import logging
from utils import (
    load_hf_dataset, 
    load_model, 
    load_tokenizer, 
    run_inference,
    save_perplexities
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
    perplexities = []
    for model_name in MODEL_NAMES:
        model = load_model(model_name)
        dataset = load_hf_dataset()
        perplexity = run_inference(model, tokenizer, dataset)
        perplexities.append(perplexity)
        logger.info(f"Model: {model_name}, Perplexity: {perplexity}")
        del model
        del dataset
    
    save_perplexities(perplexities, MODEL_NAMES)

if __name__ == "__main__":
    main()