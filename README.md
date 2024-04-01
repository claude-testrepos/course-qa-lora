# Course Q&A LoRA

Directed-study project (Winter 2024): fine-tuning Mistral-7B-Instruct with QLoRA on anonymized Q&A from our algorithms course forum, so it answers in the TAs' style.

## Train

```
python prepare_data.py raw.jsonl
python train.py --epochs 2 --rank 16
```
Fits on a single 24GB GPU (RTX 3090/4090) in about 40 minutes.

## Try it

```
python chat.py                 # REPL with the adapter
python merge.py --out merged/  # standalone model
```

## Results

| model | ROUGE-L (val) |
|---|---|
| Mistral-7B-Instruct | 0.19 |
| + LoRA r=16 | 0.31 |

## Data

The raw forum data is not included for privacy reasons.
