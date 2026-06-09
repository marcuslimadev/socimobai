import argparse
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn


SYSTEM = (
    "Voce e Teresa, assistente imobiliaria da Exclusiva Lar Imoveis. "
    "Responda em portugues do Brasil, com educacao, empatia e uma pergunta por vez. "
    "Reconheca dados ja informados antes de perguntar o proximo dado."
)


class ChatRequest(BaseModel):
    message: str = ""
    history: str = ""
    assistant_name: str = "Teresa"
    company_name: str = "Exclusiva Lar Imoveis"
    max_new_tokens: int = 140


def build_prompt(tokenizer, payload: ChatRequest) -> str:
    system = SYSTEM.replace("Teresa", payload.assistant_name).replace(
        "Exclusiva Lar Imoveis",
        payload.company_name,
    )
    content = payload.message.strip()
    if payload.history.strip():
        content = f"HISTORICO:\n{payload.history.strip()}\n\nMENSAGEM ATUAL:\n{content}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"SYSTEM: {system}\nUSER: {content}\nASSISTANT:"


def create_app(model_dir: Path, api_key: str = ""):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    app = FastAPI(title="SocimobAI Fine-tuned Local API")

    @app.get("/health")
    def health():
        return {
            "success": True,
            "provider": "socimobai-python-finetuned",
            "model_dir": str(model_dir),
        }

    @app.post("/chat")
    def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
        if api_key:
            token = (authorization or "").removeprefix("Bearer ").strip()
            if token != api_key:
                raise HTTPException(status_code=401, detail="Unauthorized")

        prompt = build_prompt(tokenizer, payload)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max(16, min(payload.max_new_tokens, 300)),
                do_sample=True,
                temperature=0.45,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "success": True,
            "content": text.strip(),
            "provider": "socimobai-python-finetuned",
            "model": str(model_dir),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="API local para o modelo SocimobAI fine-tuned.")
    parser.add_argument("--model-dir", default="models/socimobai-qwen-ft")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    app = create_app(Path(args.model_dir), args.api_key)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
