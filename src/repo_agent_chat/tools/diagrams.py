import re
from pathlib import PurePosixPath

from repo_agent_chat.repository import SourceFile

IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+([\w.]+)\s+import\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+([\w.]+)\s*$", re.MULTILINE),
    re.compile(r"^\s*import\s+(?:[\w{},* ]+\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE),
    re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE),
)


def generate_dependency_diagram(
    source_files: list[SourceFile],
    max_nodes: int = 40,
) -> dict[str, object]:
    """Gera um flowchart Mermaid determinístico das dependências entre arquivos."""

    code_files = [source for source in source_files if _is_code_file(source.path)]
    selected = code_files[:max_nodes]
    node_ids = {source.path: f"n{index}" for index, source in enumerate(selected)}
    aliases = _build_aliases(selected)
    edge_evidence: dict[tuple[str, str], int] = {}

    for source in selected:
        for imported, line in _extract_imports(source.content):
            target = _resolve_import(imported, aliases)
            if target and target != source.path:
                edge_evidence.setdefault((source.path, target), line)

    lines = ["flowchart LR"]
    lines.extend(
        f'    {node_ids[path]}["{_escape_label(path)}"]' for path in node_ids
    )
    lines.extend(
        f"    {node_ids[source]} --> {node_ids[target]}"
        for source, target in sorted(edge_evidence)
    )
    return {
        "diagram": "\n".join(lines),
        "nodes": len(selected),
        "edges": len(edge_evidence),
        "truncated": len(code_files) > len(selected),
        "paths": list(node_ids),
        "edge_evidence": [
            {"source": source, "target": target, "line": line}
            for (source, target), line in sorted(edge_evidence.items())
        ],
    }


def _is_code_file(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in {
        ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs"
    }


def _build_aliases(source_files: list[SourceFile]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source in source_files:
        path = PurePosixPath(source.path)
        without_suffix = path.with_suffix("").as_posix()
        for alias in {
            path.stem,
            without_suffix,
            without_suffix.replace("/", "."),
            without_suffix.removeprefix("src."),
            without_suffix.removeprefix("src/"),
        }:
            aliases.setdefault(alias, source.path)
    return aliases


def _extract_imports(content: str) -> set[tuple[str, int]]:
    """Extrai módulo e linha do import para sustentar cada aresta do grafo."""

    imports: set[tuple[str, int]] = set()
    for pattern in IMPORT_PATTERNS:
        for match in pattern.finditer(content):
            imports.add((match.group(1), content.count("\n", 0, match.start()) + 1))
    return imports


def _resolve_import(imported: str, aliases: dict[str, str]) -> str | None:
    normalized = imported.removeprefix("./").replace("../", "")
    candidates = (
        normalized,
        normalized.replace("/", "."),
        normalized.split(".")[-1],
        normalized.split("/")[-1],
    )
    return next((aliases[candidate] for candidate in candidates if candidate in aliases), None)


def _escape_label(path: str) -> str:
    return path.replace('"', "'")
