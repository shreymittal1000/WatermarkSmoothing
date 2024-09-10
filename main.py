from transformers import (pipeline, AutoTokenizer, AutoModelForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from model_loader import load_model
from torch import Tensor

# All the models we will be using
model_small = "TinyLlama/TinyLlama_v1.1"
model_big = "meta-llama/Llama-2-7b-hf"

DEBUG = True

if __name__ == "__main__":
    # Load the LLMs
    pipeline_small = load_model(model_small)
    tokenizer_small = pipeline_small.tokenizer
    model_small = pipeline_small.model
    # pipeline_big = load_model(model_big)
    # tokenizer_big = pipeline_big.tokenizer
    # model_big = pipeline_big.model
    
    if DEBUG: # Checks if the models work
        input_text = "Who is the current president of the United States?"
        print("--------------------")
        input_1 = tokenizer_small(input_text, return_tensors="pt", max_length=50, truncation=True)
        output_1: CausalLMOutputWithPast = model_small(**input_1)
        print(output_1.logits.shape)
        print(tokenizer_small.decode(output_1.logits.argmax(dim=-1)[0]))
        # print("--------------------")
        # input_2 = tokenizer_big(input_text, return_tensors="pt", max_length=50, truncation=True)
        # output_2 = model_big(**input_2)
        # print(output_2)