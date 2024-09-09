import os
import torch
import transformers
from huggingface_hub import login

# Define your Hugging Face API token
api_token = "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"  # Replace this with your actual token

# Option 1: Set API token as an environment variable (recommended)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Option 2: Use Hugging Face's login functionality
# This will log you in and create a token configuration on your machine
login(api_token)

# If model is not downloaded and cached, it will be downloaded
model_id = "meta-llama/Meta-Llama-3.1-8B"

# try:
#     os.path.exists(model_id)
#     local_dir = "./" + model_id
#     pipeline = transformers.pipeline("text-generation", model=local_dir, tokenizer=local_dir, device_map="auto")
# except:
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
pipeline.save_pretrained(model_id)

# Generate some text as a test
input_text = "The future of AI is"
outputs = pipeline(input_text)
print(outputs)
