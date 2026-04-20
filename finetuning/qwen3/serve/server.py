"""
Qwen3-4B RAFT-v2 本地推理服务（OpenAI-compatible /v1/chat/completions）

启动：
    cd /home/jhqy/IRF/finetuning/qwen3
    source .venv/bin/activate
    uvicorn serve.server:app --host 0.0.0.0 --port 8001
"""
import os
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

MODEL_PATH = os.environ.get(
    "RAFT_MODEL_PATH",
    "/home/jhqy/IRF/finetuning/qwen3/outputs/qwen3-4b-raft-v2-merged",
)

print(f"Loading model from {MODEL_PATH} ...")
_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
_model.eval()
print("Model ready.")

app = FastAPI(title="RAFT Inference Server")


class _Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-raft"
    messages: List[_Message]
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = _tok(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
        )
    full_text = _tok.decode(out[0], skip_special_tokens=True)
    answer = full_text.split("assistant\n")[-1].strip() if "assistant\n" in full_text else full_text.strip()

    n_prompt = inputs["input_ids"].shape[1]
    n_new = out.shape[1] - n_prompt
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_new, "total_tokens": n_prompt + n_new},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
