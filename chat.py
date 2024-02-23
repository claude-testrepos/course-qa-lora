"""Quick REPL to poke at the fine-tuned model."""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
adapter = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/course-qa-lora"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

while True:
    try:
        q = input("\n> ")
    except EOFError:
        break
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    model.generate(ids, max_new_tokens=400, do_sample=True, temperature=0.7, streamer=streamer)
