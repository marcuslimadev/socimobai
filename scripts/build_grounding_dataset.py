import json
from pathlib import Path


SYSTEM = (
    "Voce e Teresa, assistente imobiliaria da Exclusiva Lar Imoveis. "
    "Nunca invente detalhes de imoveis, nunca confunda aluguel com compra, "
    "nao repita perguntas ja respondidas e faca no maximo uma pergunta por mensagem."
)


PROPERTIES = [
    {
        "code": "EXC-2026032",
        "kind": "apartamento",
        "neighborhood": "Chacaras Santa Ines",
        "city": "Santa Luzia",
        "rooms": "2",
        "parking": "1",
        "price": "1500",
        "goal": "aluguel",
        "price_text": "aluguel de R$ 1.500",
    },
    {
        "code": "9049",
        "kind": "apartamento",
        "neighborhood": "Chacaras Santa Ines",
        "city": "Santa Luzia",
        "rooms": "2",
        "parking": "1",
        "price": "1500",
        "goal": "aluguel",
        "price_text": "aluguel de R$ 1.500",
    },
    {
        "code": "EXC-2026101",
        "kind": "apartamento",
        "neighborhood": "Sao Benedito",
        "city": "Santa Luzia",
        "rooms": "1",
        "parking": "1",
        "price": "1800",
        "goal": "aluguel",
        "price_text": "aluguel de R$ 1.800",
    },
    {
        "code": "EXC-2026202",
        "kind": "apartamento",
        "neighborhood": "Pampulha",
        "city": "Belo Horizonte",
        "rooms": "2",
        "parking": "1",
        "price": "300000",
        "goal": "compra",
        "price_text": "valor de R$ 300.000 para compra",
    },
]


def chat(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def property_user(prop: dict, message: str) -> str:
    return f"""
HISTORICO:
Cliente procura {prop['goal']} em {prop['city']} ate R$ {prop['price']}, {prop['rooms']} quartos, mudanca em 15 dias.

DADOS DO LEAD:
- objetivo_compra: {prop['goal']}
- preferencia_bairro: {prop['city']}
- budget_max: {prop['price']}
- quartos: {prop['rooms']}
- prazo_compra: 15 dias

IMOVEIS REAIS DISPONIVEIS:
- Codigo {prop['code']} | {prop['kind']} | {prop['neighborhood']}, {prop['city']} | {prop['rooms']} quartos | {prop['parking']} vagas | R$ {prop['price']}

MENSAGEM ATUAL DO CLIENTE:
{message}

Responda como Teresa:
"""


def property_answer(prop: dict, prefix: str) -> str:
    return (
        f"{prefix} O {prop['code']} e um {prop['kind']} em {prop['neighborhood']}, "
        f"{prop['city']}, com {prop['rooms']} quarto{'s' if prop['rooms'] != '1' else ''}, "
        f"{prop['parking']} vaga{'s' if prop['parking'] != '1' else ''} e {prop['price_text']}. "
        "Posso chamar um corretor para confirmar os detalhes e seguir com voce."
    )


def qualification_user(fields: dict, message: str) -> str:
    lead_lines = "\n".join(f"- {key}: {value}" for key, value in fields.items())
    return f"""
HISTORICO:
Sem historico anterior.

DADOS DO LEAD:
{lead_lines}

IMOVEIS REAIS DISPONIVEIS:
Nenhum imovel real enviado.

MENSAGEM ATUAL DO CLIENTE:
{message}

Responda como Teresa:
"""


def main() -> None:
    rows = []

    interest_messages = [
        "Gostei desse, pode me explicar melhor?",
        "Quero visitar esse imovel",
        "Gostei do codigo {code}",
        "{code} gostei desse",
        "Pode chamar um corretor",
        "Esse ainda esta disponivel?",
    ]
    prefixes = [
        "Claro.",
        "Perfeito.",
        "Otima escolha.",
        "Que bom que gostou.",
        "Certo, vou registrar seu interesse.",
    ]

    for prop in PROPERTIES:
        for message in interest_messages:
            rendered_message = message.format(code=prop["code"])
            for prefix in prefixes:
                rows.append(chat(property_user(prop, rendered_message), property_answer(prop, prefix)))

    rows.extend(
        [
            chat(
                qualification_user(
                    {
                        "objetivo_compra": "aluguel",
                        "preferencia_bairro": "Santa Luzia",
                        "budget_max": "1500",
                    },
                    "Quero alugar apartamento ate 1500 em Santa Luzia",
                ),
                "Perfeito, vou considerar aluguel em Santa Luzia ate R$ 1.500. Quantos quartos voce precisa?",
            ),
            chat(
                qualification_user(
                    {
                        "objetivo_compra": "aluguel",
                        "preferencia_bairro": "Santa Luzia",
                        "budget_max": "1500",
                        "quartos": "2",
                    },
                    "19 de junho",
                ),
                "Perfeito, anotei o prazo para 19 de junho. Sua renda vem de CLT, autonomo, empresa/PJ ou outra fonte?",
            ),
            chat(
                qualification_user(
                    {
                        "objetivo_compra": "aluguel",
                        "preferencia_bairro": "Santa Luzia",
                        "budget_max": "1500",
                        "quartos": "2",
                        "prazo_compra": "19 de junho",
                    },
                    "CLT e autonomo",
                ),
                "Obrigado, anotei renda CLT e autonomo. Se algum imovel fizer sentido, me diga o codigo ou posso chamar um corretor para seguir com voce.",
            ),
            chat(
                qualification_user(
                    {
                        "objetivo_compra": "compra",
                        "tipo_imovel": "terreno",
                        "preferencia_bairro": "Lagoa Santa",
                        "budget_max": "180000",
                    },
                    "Quero um terreno perto da lagoa",
                ),
                "Entendi. Vou considerar compra de terreno em Lagoa Santa ate R$ 180.000. Qual tamanho de terreno voce procura?",
            ),
            chat(
                qualification_user(
                    {
                        "objetivo_compra": "compra",
                        "tipo_imovel": "terreno",
                        "preferencia_bairro": "Lagoa Santa",
                        "budget_max": "180000",
                        "area_minima": "360 m2",
                    },
                    "Pode ser 360 metros",
                ),
                "Perfeito, anotei terreno em Lagoa Santa, ate R$ 180.000, com cerca de 360 m2. Posso procurar opcoes compativeis para voce.",
            ),
        ]
    )

    output = Path("data/treinamento_socimob_grounding_amplified.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(rows)} exemplos gravados em {output}")


if __name__ == "__main__":
    main()
