import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define your Hugging Face API token
api_token = "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"  # Replace this with your actual token

# Option 1: Set API token as an environment variable (recommended)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Option 2: Use Hugging Face's login functionality
# This will log you in and create a token configuration on your machine
login(api_token)

model_name = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate some text as a test
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50)

# Print generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Example input
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(inputs.input_ids, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
