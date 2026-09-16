from repo_agent_chat.faithfulness import extract_referenced_symbols, symbol_exists


def test_extract_referenced_symbols_de_chamada_composta() -> None:
    symbols = extract_referenced_symbols(
        "Usa `RagAssistant().generate_response(query)` para responder."
    )

    assert symbols == {"RagAssistant", "generate_response"}


def test_extract_referenced_symbols_ignora_arquivo_e_citacao() -> None:
    symbols = extract_referenced_symbols(
        "Veja `src/repo_agent_chat/rag.py` e `src/repo_agent_chat/rag.py:38`."
    )

    assert symbols == set()


def test_symbol_exists_nao_aceita_prefixo_de_outro_identificador() -> None:
    assert symbol_exists("embed_chunks", "def embed_chunks(...):") is True
    assert symbol_exists("embed_chunk", "def embed_chunks(...):") is False
