from model_loader import load_model

# All the models we will be using
model_small = "TinyLlama/TinyLlama_v1.1"
model_big = "meta-llama/Llama-2-7b-hf"
pipeline_small = load_model(model_small)
pipeline_big = load_model(model_big)

if __name__ == "__main__":
    # Generate some text as a test
    input_text = "The future of AI is"
    output_1 = pipeline_small(input_text, max_length=250, truncation=True)[0]["generated_text"]
    print(output_1)
    output_2 = pipeline_big(input_text, max_length=250, truncation=True)[0]["generated_text"]
    print(output_2)