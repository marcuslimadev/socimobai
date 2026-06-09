import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = (
    "Voce e Teresa, assistente imobiliaria da Exclusiva Lar Imoveis. "
    "Responda em portugues do Brasil, com educacao, empatia e uma pergunta por vez."
)


def build_prompt(tokenizer, message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": message},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"SYSTEM: {SYSTEM}\nUSER: {message}\nASSISTANT:"


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste local do modelo SocimobAI fine-tuned.")
    parser.add_argument("--model-dir", default="models/socimobai-ft")
    parser.add_argument("--message", default="Quero alugar apartamento ate 1500 em Santa Luzia")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    prompt = build_prompt(tokenizer, args.message)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.55,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
