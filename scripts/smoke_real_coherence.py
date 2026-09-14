from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-prompt real-weight coherence smoke test")
    parser.add_argument("--model-path", required=True, help="Local model snapshot path")
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    inputs = tokenizer(args.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
