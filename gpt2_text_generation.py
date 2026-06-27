"""
GPT-2 Text Generation - ProDigy Infotech ML Internship Task 01
Fine-tune GPT-2 on custom dataset and generate text from prompts.
"""

import os
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# 1.  CUSTOM DATASET
# ──────────────────────────────────────────────

class TextDatasetCustom(Dataset):
    """Simple character/word dataset for fine-tuning GPT-2."""

    def __init__(self, tokenizer, file_path: str, block_size: int = 128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )["input_ids"][0]

        # Split into fixed-length blocks
        self.examples = [
            tokenized[i : i + block_size]
            for i in range(0, len(tokenized) - block_size + 1, block_size)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx].clone().detach()


# ──────────────────────────────────────────────
# 2.  FINE-TUNING
# ──────────────────────────────────────────────

def fine_tune_gpt2(
    train_file: str = "train_data.txt",
    output_dir: str = "./gpt2-finetuned",
    model_name: str = "gpt2",
    num_epochs: int = 3,
    batch_size: int = 4,
    block_size: int = 128,
    learning_rate: float = 5e-5,
):
    """
    Fine-tune GPT-2 on a custom text corpus.

    Parameters
    ----------
    train_file   : path to plain-text training file
    output_dir   : where to save the fine-tuned model
    model_name   : base GPT-2 variant ('gpt2', 'gpt2-medium', ...)
    num_epochs   : training epochs
    batch_size   : per-device batch size
    block_size   : token block length for chunking
    learning_rate: AdamW learning rate
    """
    print(f"\n{'='*55}")
    print("  GPT-2 Fine-Tuning — ProDigy Infotech Task 01")
    print(f"{'='*55}\n")

    # Load tokenizer & model
    print(f"[1/4] Loading base model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Build dataset
    print(f"[2/4] Building dataset from '{train_file}'")
    train_dataset = TextDatasetCustom(tokenizer, train_file, block_size)
    print(f"      → {len(train_dataset)} training blocks of {block_size} tokens")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,           # causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train
    print("[3/4] Training …")
    trainer.train()

    # Save
    print(f"[4/4] Saving fine-tuned model to '{output_dir}'")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✅ Fine-tuning complete!\n")
    return output_dir


# ──────────────────────────────────────────────
# 3.  TEXT GENERATION
# ──────────────────────────────────────────────

def generate_text(
    prompt: str,
    model_path: str = "gpt2",
    max_new_tokens: int = 200,
    num_return_sequences: int = 3,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
):
    """
    Generate text continuations for a given prompt.

    Parameters
    ----------
    prompt              : the input text to continue
    model_path          : path/name of the GPT-2 model to use
    max_new_tokens      : maximum tokens to generate
    num_return_sequences: how many independent completions to produce
    temperature         : sampling temperature (higher → more creative)
    top_k               : top-K sampling cutoff
    top_p               : nucleus (top-P) sampling threshold
    repetition_penalty  : penalise already-generated tokens (>1 reduces repetition)
    do_sample           : use sampling (True) or greedy/beam (False)
    """
    print(f"\n{'─'*55}")
    print(f"  Model : {model_path}")
    print(f"  Prompt: {prompt!r}")
    print(f"{'─'*55}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, output in enumerate(outputs, 1):
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
        print(f"\n[Generation {i}]\n{text}")

    return results


# ──────────────────────────────────────────────
# 4.  DECODING STRATEGY DEMO
# ──────────────────────────────────────────────

def compare_decoding_strategies(prompt: str, model_path: str = "gpt2"):
    """
    Demonstrate five common decoding strategies side-by-side.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    strategies = {
        "Greedy": dict(do_sample=False),
        "Beam Search (n=4)": dict(do_sample=False, num_beams=4, early_stopping=True),
        "Top-K (k=50)": dict(do_sample=True, top_k=50, temperature=1.0),
        "Top-P / Nucleus (p=0.95)": dict(do_sample=True, top_p=0.95, top_k=0, temperature=1.0),
        "Top-K + Top-P": dict(do_sample=True, top_k=50, top_p=0.95, temperature=0.9),
    }

    print(f"\n{'='*55}")
    print("  Decoding Strategy Comparison")
    print(f"  Prompt: {prompt!r}")
    print(f"{'='*55}")

    for name, kwargs in strategies.items():
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=80,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n▶ {name}\n{text}")


# ──────────────────────────────────────────────
# 5.  QUICK DEMO (no fine-tuning needed)
# ──────────────────────────────────────────────

def quick_demo():
    """Run text generation with the base GPT-2 model (no fine-tuning required)."""
    prompts = [
        "Artificial intelligence is transforming the world by",
        "The future of machine learning looks",
        "Once upon a time in a digital world,",
    ]

    print("\n" + "=" * 55)
    print("  GPT-2 Quick Demo — ProDigy Infotech Task 01")
    print("=" * 55)

    for prompt in prompts:
        generate_text(
            prompt=prompt,
            model_path="gpt2",
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.85,
        )

    compare_decoding_strategies(
        prompt="The most important thing about deep learning is",
        model_path="gpt2",
    )


# ──────────────────────────────────────────────
# 6.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-2 Text Generation — ProDigy Infotech")
    subparsers = parser.add_subparsers(dest="command")

    # demo command
    subparsers.add_parser("demo", help="Quick demo with base GPT-2 (no training needed)")

    # generate command
    gen_p = subparsers.add_parser("generate", help="Generate text from a prompt")
    gen_p.add_argument("--prompt", required=True)
    gen_p.add_argument("--model", default="gpt2")
    gen_p.add_argument("--max_tokens", type=int, default=200)
    gen_p.add_argument("--num_sequences", type=int, default=3)
    gen_p.add_argument("--temperature", type=float, default=0.9)

    # finetune command
    ft_p = subparsers.add_parser("finetune", help="Fine-tune GPT-2 on a custom dataset")
    ft_p.add_argument("--train_file", required=True)
    ft_p.add_argument("--output_dir", default="./gpt2-finetuned")
    ft_p.add_argument("--model", default="gpt2")
    ft_p.add_argument("--epochs", type=int, default=3)
    ft_p.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    if args.command == "demo" or args.command is None:
        quick_demo()

    elif args.command == "generate":
        generate_text(
            prompt=args.prompt,
            model_path=args.model,
            max_new_tokens=args.max_tokens,
            num_return_sequences=args.num_sequences,
            temperature=args.temperature,
        )

    elif args.command == "finetune":
        fine_tune_gpt2(
            train_file=args.train_file,
            output_dir=args.output_dir,
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )