from unittest.mock import Mock

from repo_agent_chat.retrieval.chunking import CodeChunk
from repo_agent_chat.retrieval.rag import RagAssistant, format_retrieved_context
from repo_agent_chat.retrieval.vector_store import SearchResult


def test_format_retrieved_context_inclui_fontes_e_conteudo() -> None:
    results = [
        SearchResult(
            chunk=CodeChunk("auth.py", 10, 12, "def login():\n    pass"),
            score=0.9,
        ),
        SearchResult(
            chunk=CodeChunk("users.py", 20, 20, "class User: ..."),
            score=0.8,
        ),
    ]

    context = format_retrieved_context(results)

    assert "Fonte: auth.py:10-12" in context
    assert "def login():" in context
    assert "Fonte: users.py:20-20" in context


def test_rag_assistant_recupera_contexto_e_transmite_resposta() -> None:
    result = SearchResult(
        chunk=CodeChunk("auth.py", 1, 1, "def login(): pass"),
        score=1.0,
    )
    retriever = Mock()
    retriever.search.return_value = [result]
    chat = Mock()
    chat.stream.return_value = iter(["A função ", "está em auth.py:1."])
    assistant = RagAssistant(chat, retriever, top_k=3)

    answer = "".join(assistant.stream("Onde está o login?"))

    assert answer == "A função está em auth.py:1."
    retriever.search.assert_called_once_with("Onde está o login?", 3)
    chat.stream.assert_called_once_with(
        "Onde está o login?",
        context="Fonte: auth.py:1-1\ndef login(): pass",
    )
