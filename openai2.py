import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def run_ai_app():
    
    print("Loading tokenizer and model...")
    model_id = "openai-community/gpt2"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)


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
    
    print("Welcome to the AI Chatbot!")

    # Function to run the AI application
    chat = [
    {"role": "system", "content": "You are a helpful and intelligent AI assistant who responds to user queries. You ever respond in a comedy way."},
    {"role": "user", "content": "Tell me how to create a AI with hugging face and python. give me a short code to do that."},
    ]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI: Goodbye!")
            break
        # response = generate_response(user_input)
        chat.append({"role": "user", "content": user_input})
        response = pipe(chat, max_new_tokens=100, num_return_sequences=1)
        print("AI: ", response[0]['generated_text'])

if __name__ == "__main__":
    print("Starting the AI application...")
    # Teste simples
    for i in range(5):
        print(f"Test {i}")
    run_ai_app()

# print(pipe(chat, max_new_tokens=100))