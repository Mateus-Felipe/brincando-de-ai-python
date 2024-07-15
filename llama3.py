import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config
    },
    device_map="auto"
)

chat = [
    {"role": "system", "content": "You are a helpful and intelligent AI assistant who responds to user queries. You ever respond in a comedy way."},
    {"role": "user", "content": "Tell me how to create a AI with hugging face and python. give me a short code to do that."}
]

print(pipe(chat, max_new_tokens=100))
# pipe("Hey how are you doing today?")

    
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")


# model("Hey how are you doing today?")

# hf_quantizer.validate_environment(
#  File "C:\Users\mateus.ramos\Documents\brincandocomai\testeLlama3\.env\Lib\site-packages\transformers\quantizers\quantizer_bnb_8bit.py", line 62, in validate_environment
#  raise RuntimeError("No GPU found. A GPU is needed for quantization.")
# RuntimeError: No GPU found. A GPU is needed for quantization.