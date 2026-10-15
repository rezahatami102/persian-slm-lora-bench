import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from farstail.datasets import farstail

# 1) Inspect the FarsTail dataset
train, val, test = farstail.load_original_data()
print("Types:", type(train), type(val), type(test))
print("Sizes:", len(train), len(val), len(test))
print("First train sample:", train[0] if not hasattr(train, "iloc") else train.iloc[0])

# 2) Load a small model in 4-bit (NF4) to verify QLoRA works on this GPU
model_id = "Qwen/Qwen2.5-0.5B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)

# 3) Run a short generation
inputs = tokenizer("سلام، حال شما چطور است؟", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(out[0], skip_special_tokens=True))
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")