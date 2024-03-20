"""Merge the LoRA adapter into the base weights so it can be served without peft (vLLM, Ollama, ...)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "mistralai/Mistral-7B-Instruct-v0.2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="checkpoints/course-qa-lora")
    p.add_argument("--out", default="merged/course-qa-mistral")
    p.add_argument("--push", metavar="REPO_ID", help="also push to the Hugging Face Hub (private)")
    args = p.parse_args()

    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(BASE)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"saved merged model to {args.out}")
    if args.push:
        model.push_to_hub(args.push, private=True)
        tok.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    main()
