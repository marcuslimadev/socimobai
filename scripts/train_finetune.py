import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_SYSTEM = (
    "Voce e Teresa, assistente imobiliaria da Exclusiva Lar Imoveis. "
    "Atenda em portugues do Brasil com empatia, educacao e foco no proximo passo. "
    "Nao repita perguntas ja respondidas, faca no maximo uma pergunta por mensagem "
    "e nunca invente imoveis."
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Linha {line_number} invalida em {path}: {exc}") from exc
    return rows


def normalize_messages(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []

    clean = []
    has_system = False
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        has_system = has_system or role == "system"
        clean.append({"role": role, "content": content})

    if clean and not has_system:
        clean.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})

    return clean


def render_chat(tokenizer, messages: list[dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered = []
    for message in messages:
        role = message["role"].upper()
        rendered.append(f"{role}: {message['content']}")
    rendered.append("FIM")
    return "\n".join(rendered)


def build_dataset(tokenizer, dataset_paths: list[Path]) -> Dataset:
    examples = []
    for dataset_path in dataset_paths:
        for row in read_jsonl(dataset_path):
            messages = normalize_messages(row)
            if not messages:
                continue
            text = render_chat(tokenizer, messages)
            examples.append({"text": text})

    if not examples:
        raise RuntimeError("Nenhum exemplo valido encontrado nos datasets informados")

    return Dataset.from_list(examples)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def find_lora_targets(model) -> list[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    ]
    linear_names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    targets = [name for name in preferred if name in linear_names]
    if targets:
        return targets

    return sorted(linear_names)[:8]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning real do SocimobAI com LoRA e merge final.")
    parser.add_argument("--dataset", nargs="+", default=["data/treinamento_imobiliaria.jsonl"])
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", default="models/socimobai-ft")
    parser.add_argument("--epochs", type=float, default=18)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    dataset_paths = [Path(item) for item in args.dataset]
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir.with_name(output_dir.name + "-adapter")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer, dataset_paths)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.config.use_cache = False

    targets = find_lora_targets(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(adapter_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.base_model,
        "dataset": [str(path) for path in dataset_paths],
        "examples": len(dataset),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "lora_targets": targets,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "socimob_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
