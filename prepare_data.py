"""Convert scraped Q&A (forum posts + TA answers) into chat-format JSONL."""
import argparse
import json
import random
from pathlib import Path

SYSTEM = "You are a teaching assistant for an undergraduate algorithms course."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path, help="raw.jsonl with {question, answer} rows")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--val", type=float, default=0.05)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.raw.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if len(r["answer"]) > 40]  # drop "see lecture 3" answers
    random.Random(0).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val))
    args.out.mkdir(exist_ok=True)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with (args.out / f"{name}.jsonl").open("w") as fh:
            for r in part:
                msgs = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"].strip()},
                    {"role": "assistant", "content": r["answer"].strip()},
                ]
                fh.write(json.dumps({"messages": msgs}) + "\n")
        print(name, len(part))


if __name__ == "__main__":
    main()
