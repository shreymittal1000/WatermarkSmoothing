from transformers import (pipeline, AutoTokenizer, AutoModelForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import Tensor
from src.model_loader import load_model
from src.tools import tensor_rank_positions, rank_difference, n_bigger, z_score
from src.watermark_tools_context_independent import (
    generate_soft_greenlist_watermark_context_independent, watermark_checker, predict_greenlist_confidence
)

# All the models we will be using
model_small = "TinyLlama/TinyLlama_v1.1"
model_big = "meta-llama/Llama-2-7b-hf"

if __name__ == "__main__":
    # Load the LLMs
    pipeline_small = load_model(model_small)
    tokenizer_small = pipeline_small.tokenizer
    model_small = pipeline_small.model
    pipeline_big = load_model(model_big)
    tokenizer_big = pipeline_big.tokenizer
    model_big = pipeline_big.model
    
    # Generate the watermark
    watermark = generate_soft_greenlist_watermark_context_independent(tokenizer_big.vocab_size, 0.5, 1.0)
    
    # Sample test
    input_text = "Who is the current president of the United States?"
    input_main = tokenizer_big(input_text, return_tensors="pt", max_length=50, truncation=True)
    output_main: CausalLMOutputWithPast = model_big.generate(**input_main)
    print(tokenizer_big.decode(output_main.logits.argmax(dim=-1)[0]))