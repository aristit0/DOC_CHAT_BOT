from transformers import pipeline

# Use any compatible GPU-based LLM (adjust as needed)
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0)

def generate_response(prompt, max_tokens=256):
    response = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("Answer:")[-1].strip()