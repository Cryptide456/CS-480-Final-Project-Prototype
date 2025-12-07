import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """
    Load the open-source Code Llama Instruct model.
    No API keys required.
    """
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model


def generate_review(tokenizer, model, code_snippet):
    """
    Generate a code review-style explanation from Code Llama.
    """
    prompt = (
        "You are an expert software engineer performing a code review.\n"
        "Provide clear, constructive feedback and suggestions for improvement.\n\n"
        f"Code:\n{code_snippet}\n\n"
        "Review Comments:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    sample_code = """
def add(a,b):
  return a+b
"""

    print("Loading model... (this may take a moment)")
    tokenizer, model = load_model()

    print("\n=== CODE REVIEW OUTPUT ===\n")
    review = generate_review(tokenizer, model, sample_code)
    print(review)


if __name__ == "__main__":
    main()
