import re

INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
ATTRIBUTE_PATTERN = re.compile(r"\.([A-Za-z_]\w*)\b")
FILE_REFERENCE_PATTERN = re.compile(
    r"^[\w./-]+\.[A-Za-z0-9]+(?::\d+(?:-\d+)?)?$"
)


def extract_referenced_symbols(text: str) -> set[str]:
    """Extrai símbolos afirmados em trechos inline sem confundir arquivos."""

    symbols: set[str] = set()
    for match in INLINE_CODE_PATTERN.finditer(text):
        snippet = match.group(1).strip()
        if FILE_REFERENCE_PATTERN.fullmatch(snippet):
            continue
        if IDENTIFIER_PATTERN.fullmatch(snippet):
            symbols.add(snippet)
            continue
        symbols.update(CALL_PATTERN.findall(snippet))
        symbols.update(ATTRIBUTE_PATTERN.findall(snippet))
    return symbols


def symbol_exists(symbol: str, evidence: str) -> bool:
    """Confirma um identificador completo, evitando colisões por prefixo."""

    return re.search(rf"\b{re.escape(symbol)}\b", evidence) is not None
