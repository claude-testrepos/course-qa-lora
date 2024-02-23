import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

BASE = "mistralai/Mistral-7B-Instruct-v0.2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--out", default="checkpoints/course-qa-lora")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=peft_cfg,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            bf16=True,
            gradient_checkpointing=True,
            report_to="wandb",
        ),
    )
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
