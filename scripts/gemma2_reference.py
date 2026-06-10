#!/usr/bin/env python3
"""
Reference script: run gemma-2-2b-it through HuggingFace transformers
to compare output against sheplet's candle implementation.

Usage:
    python3 scripts/gemma2_reference.py

Requires: pip install transformers torch
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "downloaded-models/google--gemma-2-2b-it"

# Exact same prompt that sheplet produces (assemble_prompt_gemma with RAG context)
SYSTEM_PROMPT = "You are a helpful ancient history tutor. Answer questions accurately using course materials."
CONTEXT = (
    "The city of Rome was built on seven hills overlooking the Tiber River: the Palatine, "
    "Capitoline, Aventine, Caelian, Esquiline, Viminal, and Quirinal. The Palatine Hill is "
    "where the earliest settlement is thought to have begun, and the Capitoline Hill held "
    "the city's most important temples."
)
QUESTION = "On how many hills was the city of Rome built?"

PROMPT = (
    f"<start_of_turn>user\n"
    f"Instructions: {SYSTEM_PROMPT}\n\n"
    f"---\nContext from course materials:\n"
    f"[1] {CONTEXT} (Source: roman_founding.txt)\n"
    f"---\n\n"
    f"Question: {QUESTION}<end_of_turn>\n"
    f"<start_of_turn>model\n"
)

def main():
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16
    ).to("cpu")
    model.eval()

    print(f"\nPrompt:\n{PROMPT}\n")

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}")

    # Greedy decoding (temperature=0, no sampling)
    print("\n=== Greedy decoding (do_sample=False) ===")
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"Response: {response}")

    # Print per-token breakdown
    print("\nPer-token breakdown:")
    for i, token_id in enumerate(outputs[0][input_len:]):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  [{i}] id={token_id.item()} text={text!r}")

    # Show top-5 logits for first generated token
    print("\n=== First-token logit analysis ===")
    with torch.no_grad():
        out = model(inputs["input_ids"])
        logits = out.logits[0, -1, :].float()  # last position
        top5 = torch.topk(logits, 5)
        print("Top-5 logits:")
        for i in range(5):
            tid = top5.indices[i].item()
            val = top5.values[i].item()
            text = tokenizer.decode([tid], skip_special_tokens=False)
            print(f"  {i+1}: token {tid} ({text!r}) = {val:.4f}")

    # Also try with sampling matching sheplet defaults (temp=0.5, top_p=0.9)
    print("\n=== Sampling (temp=0.5, top_p=0.9) — 3 runs ===")
    for run in range(3):
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.05,
            )
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"  Run {run+1}: {response}")

if __name__ == "__main__":
    main()
