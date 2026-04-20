import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/home/jhqy/IRF/finetuning/qwen3/Qwen3-4B"
device = "cuda"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def test_len(L):
    try:
        print(f"Testing seq_len={L} ... ", end="")
        inp = torch.randint(0, tokenizer.vocab_size, (2, L)).to(device)
        out = model(input_ids=inp, labels=inp)
        loss = out.loss
        loss.backward()
        print("OK")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM")
            torch.cuda.empty_cache()
            return False
        else:
            raise e

low, high = 256, 4096
best = 256

# binary search max length
while low <= high:
    mid = (low + high) // 2
    if test_len(mid):
        best = mid
        low = mid + 128
    else:
        high = mid - 128

print("\n===================================")
print(f"🔥 Maximum trainable seq length: {best}")
print("===================================")
