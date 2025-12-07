from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    model_name = "bigcode/tiny_starcoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return tokenizer, model


def generate_review(tokenizer, model, code_snippet):
    prompt = (
        "You are an expert software engineer performing a code review.\n"
        "Provide clear, constructive feedback.\n\n"
        f"Code:\n{code_snippet}\n\n"
        "Review:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.4,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    sample_code = """
def add(a, b):
    return a + b
"""
    print("Loading model...")
    tokenizer, model = load_model()

    print("\n=== CODE REVIEW OUTPUT ===\n")
    review = generate_review(tokenizer, model, sample_code)
    print(review)


if __name__ == "__main__":
    main()
