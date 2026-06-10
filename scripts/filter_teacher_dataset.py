import argparse
import json
import unicodedata
from pathlib import Path


BAD_ANYWHERE = [
    "infelizmente",
    "area util",
    "área útil",
    "andar",
    "banheiro",
    "sala",
    "cozinha",
    "condominio",
    "condomínio",
    "lazer",
    "varanda",
    "piscina",
    "transporte",
    "documentacao necessaria",
    "documentação necessária",
    "extrato bancario",
    "extrato bancário",
    "bairro tranquilo",
    "regiao muito tranquila",
    "região muito tranquila",
    "charmosa",
    "muitas opcoes interessantes",
    "muitas opções interessantes",
]

BAD_WITHOUT_PROPERTIES = [
    "nao temos",
    "não temos",
    "nao ha",
    "não há",
    "nao encontrei",
    "não encontrei",
    "nenhum imovel",
    "nenhum imóvel",
    "tenho este imovel",
    "tenho esse imovel",
    "tenho estas opcoes",
    "tenho essas opções",
    "imovel que mais se encaixa",
    "imóvel que mais se encaixa",
    "opcao para voce",
    "opção para você",
    "oportunidade para voce",
    "casa 123",
    "codigo exc",
    "código exc",
]


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


def row_text(row: dict) -> str:
    return "\n".join(str(message.get("content", "")) for message in row.get("messages", []))


def assistant_text(row: dict) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def has_real_properties(row: dict) -> bool:
    text = row_text(row)
    return "IMOVEIS REAIS DISPONIVEIS:\n- Codigo" in text


def is_bad(row: dict) -> bool:
    assistant = normalize(assistant_text(row))
    full = normalize(row_text(row))

    if assistant.count("?") > 1:
        return True
    if any(term in assistant for term in map(normalize, BAD_ANYWHERE)):
        return True
    if "terreno" in full and any(term in assistant for term in ["quartos", "dormitorios", "vaga de garagem"]):
        return True
    if not has_real_properties(row):
        if any(term in assistant for term in map(normalize, BAD_WITHOUT_PROPERTIES)):
            return True
        if any(marker in assistant for marker in ["1)", "2)", "3)", "1.", "2.", "3."]):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtra datasets de professor para manter exemplos seguros.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    kept = []
    rejected = 0
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if is_bad(row):
            rejected += 1
            continue
        kept.append(row)

    Path(args.output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    print(json.dumps({"kept": len(kept), "rejected": rejected, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
