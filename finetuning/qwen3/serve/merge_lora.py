"""
将 RAFT-v2 LoRA adapter 合并进 Qwen3-4B base model 并保存。

运行：
    cd /home/jhqy/IRF/finetuning/qwen3
    source .venv/bin/activate
    python serve/merge_lora.py
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE   = "/home/jhqy/IRF/finetuning/qwen3/Qwen3-4B"
LORA   = "/home/jhqy/IRF/finetuning/qwen3/outputs/qwen3-4b-lora-raft-v2"
OUTPUT = "/home/jhqy/IRF/finetuning/qwen3/outputs/qwen3-4b-raft-v2-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")

print("Applying LoRA...")
model = PeftModel.from_pretrained(model, LORA)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT} ...")
model.save_pretrained(OUTPUT)

tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUTPUT)

print("Done.")
