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
    edges: set[tuple[str, str]] = set()

    for source in selected:
        for imported in _extract_imports(source.content):
            target = _resolve_import(imported, aliases)
            if target and target != source.path:
                edges.add((source.path, target))

    lines = ["flowchart LR"]
    lines.extend(
        f'    {node_ids[path]}["{_escape_label(path)}"]' for path in node_ids
    )
    lines.extend(
        f"    {node_ids[source]} --> {node_ids[target]}"
        for source, target in sorted(edges)
    )
    return {
        "diagram": "\n".join(lines),
        "nodes": len(selected),
        "edges": len(edges),
        "truncated": len(code_files) > len(selected),
        "paths": list(node_ids),
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


def _extract_imports(content: str) -> set[str]:
    return {
        imported
        for pattern in IMPORT_PATTERNS
        for imported in pattern.findall(content)
    }


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
