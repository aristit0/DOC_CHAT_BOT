from transformers import pipeline

# You can replace with a more optimized or fine-tuned model
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)

def generate_response(prompt, max_tokens=256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].split("Answer:")[-1].strip()