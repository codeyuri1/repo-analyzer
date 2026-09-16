import json
from pathlib import Path
from typing import Any

from repo_agent_chat.repository import (
    RepositoryFileError,
    SourceFile,
    discover_source_files,
    read_source_file,
)
from repo_agent_chat.retrieval.retriever import RepositoryRetriever
from repo_agent_chat.tools.diagrams import generate_dependency_diagram
from repo_agent_chat.tools.security import analyze_source_files

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Busca trechos de código relacionados semanticamente a uma pergunta. "
                "Use para descobrir como ou onde um comportamento foi implementado."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "O que buscar no código.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Quantidade de resultados.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query", "top_k"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lê um intervalo de linhas de um arquivo do repositório.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Caminho relativo do arquivo.",
                    },
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Lista apenas nomes de arquivos para navegação. Não fornece evidência sobre o "
                "comportamento ou a responsabilidade do código."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Prefixo do caminho, ou texto vazio para listar tudo.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_vulnerabilities",
            "description": (
                "Executa análise estática de possíveis vulnerabilidades sem executar código "
                "nem revelar valores secretos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string"},
                    "max_findings": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_mermaid_diagram",
            "description": "Gera um flowchart Mermaid das dependências entre arquivos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {"type": "string"},
                    "max_nodes": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


class RepositoryTools:
    """Executa operações controladas sobre um repositório indexado."""

    def __init__(self, root: Path, retriever: RepositoryRetriever) -> None:
        self._root = root.resolve()
        self._retriever = retriever

    def execute(self, name: str, arguments: str) -> str:
        """Executa uma tool e sempre devolve uma resposta JSON para a LLM."""

        try:
            parsed_arguments = json.loads(arguments)
            if not isinstance(parsed_arguments, dict):
                raise TypeError("Os argumentos devem formar um objeto JSON.")

            handlers = {
                "semantic_search": self._semantic_search,
                "read_file": self._read_file,
                "list_files": self._list_files,
                "analyze_vulnerabilities": self._analyze_vulnerabilities,
                "generate_mermaid_diagram": self._generate_mermaid_diagram,
            }
            handler = handlers.get(name)
            if handler is None:
                raise ValueError(f"Tool desconhecida: {name}")

            result = handler(**parsed_arguments)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except (
            RepositoryFileError,
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)

    def _semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k deve estar entre 1 e 10.")

        results = self._retriever.search(query, top_k)
        return [
            {
                "path": result.chunk.path,
                "start_line": result.chunk.start_line,
                "end_line": result.chunk.end_line,
                "score": round(result.score, 6),
                "content": result.chunk.content,
            }
            for result in results
        ]

    def _read_file(self, path: str, start_line: int, end_line: int) -> dict[str, Any]:
        if start_line < 1 or end_line < start_line:
            raise ValueError("O intervalo de linhas é inválido.")
        if end_line - start_line + 1 > 300:
            raise ValueError("A leitura está limitada a 300 linhas.")

        source_file = read_source_file(self._root, self._root / path)
        lines = source_file.content.splitlines()
        selected_lines = lines[start_line - 1 : end_line]
        actual_end_line = start_line + len(selected_lines) - 1
        has_more = actual_end_line < len(lines)
        numbered_content = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected_lines, start=start_line)
        )
        return {
            "path": source_file.path,
            "start_line": start_line,
            "end_line": actual_end_line,
            "total_lines": len(lines),
            "has_more": has_more,
            "next_start_line": actual_end_line + 1 if has_more else None,
            "content": numbered_content,
        }

    def _list_files(self, prefix: str = "") -> list[str]:
        paths = (
            path.relative_to(self._root).as_posix()
            for path in discover_source_files(self._root)
        )
        return [path for path in paths if path.startswith(prefix)][:300]

    def _source_files(self, prefix: str) -> list[SourceFile]:
        return [
            read_source_file(self._root, path)
            for path in discover_source_files(self._root)
            if path.relative_to(self._root).as_posix().startswith(prefix)
        ]

    def _analyze_vulnerabilities(
        self,
        prefix: str = "",
        max_findings: int = 50,
    ) -> dict[str, object]:
        if not 1 <= max_findings <= 100:
            raise ValueError("max_findings deve estar entre 1 e 100.")
        findings = analyze_source_files(self._source_files(prefix), max_findings)
        return {"findings": findings, "count": len(findings), "prefix": prefix}

    def _generate_mermaid_diagram(
        self,
        prefix: str = "",
        max_nodes: int = 40,
    ) -> dict[str, object]:
        if not 1 <= max_nodes <= 100:
            raise ValueError("max_nodes deve estar entre 1 e 100.")
        return generate_dependency_diagram(self._source_files(prefix), max_nodes)
