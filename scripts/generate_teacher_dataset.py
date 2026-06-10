import argparse
import json
import os
import random
import time
import unicodedata
from pathlib import Path

import requests


STUDENT_SYSTEM = (
    "Voce e Teresa, assistente imobiliaria da Exclusiva Lar Imoveis. "
    "Responda em portugues do Brasil, com cordialidade e clareza. "
    "Faca no maximo uma pergunta por mensagem, nao repita perguntas ja respondidas "
    "e nunca invente imoveis ou caracteristicas nao fornecidas."
)

TEACHER_SYSTEM = (
    "Voce e um professor gerando respostas ideais para treinar a assistente Teresa, "
    "da imobiliaria Exclusiva Lar Imoveis. Responda apenas com a mensagem final da Teresa. "
    "Use portugues do Brasil, tom educado, acolhedor e consultivo. "
    "Faca no maximo uma pergunta. Nunca use lista de perguntas. "
    "Nao invente imoveis nem detalhes nao informados. "
    "Se nao houver imoveis reais enviados, nao diga que nao ha disponibilidade; apenas colete o proximo dado do funil. "
    "Se houver imovel real, cite apenas codigo, tipo, bairro, cidade, quartos, vagas e valor quando enviados. "
    "Se o cliente quer vender terreno, pergunte tamanho, topografia, documentacao ou valor desejado; nunca pergunte quartos de terreno. "
    "Se faltar dado, peca somente o proximo dado mais importante."
)


LOCATIONS = [
    "Santa Luzia",
    "Sao Joao Batista",
    "Chacaras Santa Ines",
    "Sao Benedito",
    "Pampulha",
    "Jaragua",
    "Santa Amelia",
    "Copacabana",
]

PROPERTY_TYPES = ["apartamento", "casa", "cobertura", "terreno"]
OBJECTIVES = ["aluguel", "compra", "venda"]
INCOME_TYPES = ["CLT", "autonomo", "CLT e autonomo", "PJ", "aposentado"]
DEADLINES = ["este mes", "em 15 dias", "19 de junho", "sem pressa", "ate o fim do mes"]


def lead_lines(lead: dict) -> str:
    if not lead:
        return "Nenhum dado estruturado ainda."
    return "\n".join(f"- {key}: {value}" for key, value in lead.items() if value not in (None, ""))


def property_lines(properties: list[dict]) -> str:
    if not properties:
        return "Nenhum imovel real enviado."
    lines = []
    for item in properties:
        line = [
            f"Codigo {item['codigo']}",
            item["tipo"],
            f"{item['bairro']}, {item['cidade']}",
            f"{item['quartos']} quartos",
            f"{item['vagas']} vagas",
            f"R$ {item['valor']}",
        ]
        lines.append(" | ".join(line))
    return "\n".join(f"- {line}" for line in lines)


def clean_text(value: str) -> str:
    replacements = {
        "vocę": "voce",
        "Vocę": "Voce",
        "Ă§": "c",
        "ĂŁ": "a",
        "Ă¡": "a",
        "Ă¢": "a",
        "Ă©": "e",
        "ĂŞ": "e",
        "Ă­": "i",
        "Ăł": "o",
        "Ă´": "o",
        "Ăş": "u",
        "Ă‡": "C",
        "ă": "a",
        "Ă": "A",
        "ę": "e",
    }
    text = str(value)
    for source, target in replacements.items():
        text = text.replace(source, target)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def student_user_prompt(history: str, lead: dict, properties: list[dict], message: str) -> str:
    return "\n".join([
        "HISTORICO:",
        history or "Sem historico anterior.",
        "",
        "DADOS DO LEAD:",
        lead_lines(lead),
        "",
        "IMOVEIS REAIS DISPONIVEIS:",
        property_lines(properties),
        "",
        "MENSAGEM ATUAL DO CLIENTE:",
        message,
        "",
        "Responda como Teresa:",
    ])


def make_property(index: int, objective: str, location: str, budget: int) -> dict:
    tipo = random.choice(PROPERTY_TYPES[:-1])
    quartos = random.choice([1, 2, 2, 3])
    vagas = random.choice([0, 1, 1, 2])
    return {
        "codigo": f"EXC-{2026000 + index}",
        "tipo": tipo,
        "bairro": random.choice([location, "Chacaras Santa Ines", "Sao Benedito"]),
        "cidade": "Santa Luzia" if location in ["Santa Luzia", "Sao Joao Batista", "Chacaras Santa Ines", "Sao Benedito"] else "Belo Horizonte",
        "quartos": quartos,
        "vagas": vagas,
        "valor": budget,
        "objetivo": objective,
    }


def build_scenario(index: int) -> dict:
    stage = random.choice([
        "inicio",
        "bairro_informado",
        "valor_informado",
        "quartos_informado",
        "prazo_informado",
        "renda_informada",
        "imovel_escolhido",
        "duvida",
        "venda",
    ])
    objective = "venda" if stage == "venda" else random.choice(["aluguel", "compra"])
    location = random.choice(LOCATIONS)
    budget = random.choice([1200, 1500, 1800, 2200, 2500]) if objective == "aluguel" else random.choice([250000, 300000, 400000, 550000])
    quartos = random.choice([None, 1, 2, "1 ou 2", 3])
    deadline = random.choice([None, *DEADLINES])
    income = random.choice([None, *INCOME_TYPES])

    lead = {}
    properties = []
    history = ""

    if objective != "venda":
        lead["objetivo_compra"] = objective
    if stage not in ["inicio"]:
        lead["preferencia_bairro"] = location
    if stage in ["valor_informado", "quartos_informado", "prazo_informado", "renda_informada", "imovel_escolhido", "duvida"]:
        lead["budget_max"] = budget
    if stage in ["quartos_informado", "prazo_informado", "renda_informada", "imovel_escolhido", "duvida"] and quartos:
        lead["quartos"] = quartos
    if stage in ["prazo_informado", "renda_informada", "imovel_escolhido", "duvida"] and deadline:
        lead["prazo_compra"] = deadline
    if stage in ["renda_informada", "imovel_escolhido", "duvida"] and income:
        lead["renda_tipo"] = income

    if stage in ["imovel_escolhido", "duvida"]:
        properties = [make_property(index, objective, location, budget)]
        history = f"Cliente procura {objective} em {location}, ate R$ {budget}, {quartos or 2} quartos."

    if stage == "inicio":
        message = random.choice([
            f"Ola, quero {objective} um imovel",
            "Ola, tenho interesse neste imovel que vi no Chaves na Mao",
            "Bom dia, estou procurando um apartamento",
        ])
    elif stage == "bairro_informado":
        message = f"Pode ser em {location}"
    elif stage == "valor_informado":
        message = f"Ate {budget}"
    elif stage == "quartos_informado":
        message = f"{quartos or 2} quartos"
    elif stage == "prazo_informado":
        message = deadline or "este mes"
    elif stage == "renda_informada":
        message = income or "CLT"
    elif stage == "imovel_escolhido":
        message = random.choice([
            f"Gostei do codigo {properties[0]['codigo']}",
            "Gostei desse, pode me explicar melhor?",
            "Quero visitar esse imovel",
        ])
    elif stage == "venda":
        lead = {"objetivo_compra": "venda", "preferencia_bairro": location}
        message = random.choice([
            f"Quero vender minha casa no {location}",
            f"Tenho um terreno para vender no {location}",
            "Quero anunciar meu imovel com voces",
        ])
    else:
        message = random.choice([
            "Nao tenho certeza ainda",
            "Tem mais opcoes nesse valor?",
            "Pode chamar um corretor?",
        ])

    return {
        "stage": stage,
        "history": history,
        "lead": lead,
        "properties": properties,
        "message": message,
    }


def is_bad_teacher_reply(content: str, scenario: dict) -> bool:
    text = clean_text(content).lower()
    no_properties = not scenario["properties"]
    if content.count("?") > 1:
        return True
    if no_properties and any(term in text for term in [
        "nao temos",
        "não temos",
        "nao ha",
        "não há",
        "nao encontrei",
        "não encontrei",
        "nenhum imovel disponivel",
        "nenhum imóvel disponível",
        "nao tenho imoveis",
        "não tenho imóveis",
        "infelizmente",
        "tenho este imovel",
        "tenho esse imovel",
        "tenho estas opcoes",
        "tenho essas opcoes",
        "imovel que mais se encaixa",
        "opcao para voce",
        "oportunidade para voce",
        "casa 123",
        "codigo exc",
        "codigo 9049",
    ]):
        return True
    if no_properties and any(term in text for term in ["1)", "2)", "3)", "1.", "2.", "3.", "1️⃣", "2️⃣", "3️⃣"]):
        return True
    lead = scenario.get("lead", {})
    if lead.get("budget_max") and any(term in text for term in ["valor maximo", "valor máximo", "orcamento", "orçamento"]):
        return True
    if lead.get("quartos") and any(term in text for term in ["quantos quartos", "numero de quartos", "número de quartos"]):
        return True
    if (lead.get("preferencia_bairro") or lead.get("localizacao")) and any(term in text for term in ["qual bairro", "regiao prefere", "região prefere"]):
        return True
    if lead.get("prazo_compra") and any(term in text for term in ["qual o prazo", "para quando", "quando pretende"]):
        return True
    if lead.get("renda_tipo") and any(term in text for term in ["sua renda", "renda vem", "tipo de renda"]):
        return True
    if "terreno" in scenario["message"].lower() and any(term in text for term in ["quartos", "dormitorios", "dormitórios", "vaga de garagem"]):
        return True
    if scenario["properties"] and any(term in text for term in ["comodidades", "mercados", "farmacias", "restaurantes", "transporte publico"]):
        return True
    if no_properties and "r$" in text and not lead.get("budget_max") and not lead.get("budget_min"):
        return True
    return False


def existing_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def call_teacher(token: str, model: str, scenario: dict, timeout: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0.45,
        "max_tokens": 220,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM},
            {
                "role": "user",
                "content": student_user_prompt(
                    scenario["history"],
                    scenario["lead"],
                    scenario["properties"],
                    scenario["message"],
                ),
            },
        ],
    }
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"Hugging Face {response.status_code}: {response.text[:500]}")
    response.encoding = "utf-8"
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintetico Socimob usando Hugging Face como professor.")
    parser.add_argument("--output", default="data/teacher_socimob_qwen7b.jsonl")
    parser.add_argument("--target-examples", type=int, default=300)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct:fastest")
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=25)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Defina HF_TOKEN no ambiente antes de gerar o dataset.")

    random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = existing_count(output)
    errors = 0

    with output.open("a", encoding="utf-8") as handle:
        while count < args.target_examples:
            scenario = build_scenario(count + 1)
            try:
                assistant = clean_text(call_teacher(token, args.model, scenario, args.timeout))
                if is_bad_teacher_reply(assistant, scenario):
                    raise RuntimeError(f"Resposta rejeitada pelo filtro de qualidade: {assistant[:220]}")
            except Exception as exc:
                errors += 1
                print(f"[erro {errors}] {exc}")
                if errors >= args.max_errors:
                    raise
                time.sleep(max(args.sleep, 2))
                continue

            user_prompt = student_user_prompt(
                scenario["history"],
                scenario["lead"],
                scenario["properties"],
                scenario["message"],
            )
            row = {
                "messages": [
                    {"role": "system", "content": STUDENT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "teacher_model": args.model,
                    "stage": scenario["stage"],
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            print(f"generated={count}/{args.target_examples} stage={scenario['stage']}")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
