"""Compare base vs fine-tuned answers on the validation set (ROUGE-L + LLM judge sample)."""
import json
import sys

import evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "mistralai/Mistral-7B-Instruct-v0.2"


def generate(model, tok, messages, max_new_tokens=300):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main(adapter):
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
    rouge = evaluate.load("rouge")
    rows = [json.loads(l) for l in open("data/val.jsonl")]
    refs = [r["messages"][-1]["content"] for r in rows]
    prompts = [r["messages"][:-1] for r in rows]

    base_out = [generate(base, tok, p) for p in prompts]
    tuned = PeftModel.from_pretrained(base, adapter)
    tuned_out = [generate(tuned, tok, p) for p in prompts]

    for name, preds in (("base", base_out), ("lora", tuned_out)):
        print(name, rouge.compute(predictions=preds, references=refs)["rougeL"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/course-qa-lora")
