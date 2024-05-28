import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Realize o login programaticamente
access_token = ""
login(access_token)
# Load the Meta-Llama model and tokenizer
model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # model_kwargs={
    #   "torch_dtype": torch.bfloat16,
        # "quantization_config": quantization_config
    # },
    device_map="auto"
)

# Function to generate a response given user input
def generate_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = pipeline(inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to run the AI application
def run_ai_app():
    print("Welcome to the AI Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break
        response = generate_response(user_input)
        print("AI:", response)

if __name__ == "__main__":
    run_ai_app()