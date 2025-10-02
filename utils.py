from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_dataset() -> Dataset:
    dataset = load_dataset("israel-adewuyi/eval_data_alphabet_sort", split="train")

    return list(dataset["prompt"])

# def compute_perplexity()

def load_model(model_name: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    inputs: list
) -> None:
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    print(type(logits))
    print(logits.shape)




"""
- Load model
- Load dataset
- run inference
- Get logits
- compute perplexity
- save to json
"""

dataset = load_hf_dataset()
model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
run_inference(model, tokenizer, dataset)