import ast
import re
from dataclasses import dataclass

EVIDENCE_LINE_PATTERN = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<code>.*)$")
SUPPORTED_CODE_EXTENSIONS = frozenset(
    {".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs"}
)
CONTROL_FLOW_NAMES = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "new", "sizeof"}
)


@dataclass(frozen=True, slots=True)
class CodeStructure:
    """Símbolos estruturais encontrados em um arquivo de código recuperado."""

    path: str
    symbols: tuple[str, ...]


class _StructureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[str] = []

    def _add(self, symbol: str) -> None:
        if symbol and symbol not in self.symbols:
            self.symbols.append(symbol)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node.name)
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if argument.arg not in {"self", "cls"}:
                self._add(argument.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        self._add(_call_name(node.func))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add(_call_name(node))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.isupper() or "_" in node.id:
            self._add(node.id)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self._add(node.arg)
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self._add(f"{node.arg}={node.value.value}")
        self.generic_visit(node)


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def extract_code_structures(evidence_lines: list[str]) -> list[CodeStructure]:
    """Reconstrói trechos e extrai estrutura sem conhecer o domínio do repositório."""

    files: dict[str, dict[int, str]] = {}
    for entry in evidence_lines:
        match = EVIDENCE_LINE_PATTERN.match(entry)
        if not match or not any(
            match.group("path").endswith(extension)
            for extension in SUPPORTED_CODE_EXTENSIONS
        ):
            continue
        code = match.group("code")
        files.setdefault(match.group("path"), {})[int(match.group("line"))] = (
            code.removeprefix(" ")
        )

    structures = []
    for path, numbered_lines in files.items():
        symbols = (
            _extract_python_symbols(numbered_lines)
            if path.endswith(".py")
            else _fallback_symbols(_join_source(numbered_lines))
        )
        if symbols:
            structures.append(CodeStructure(path, tuple(symbols[:50])))
    return structures


def _join_source(numbered_lines: dict[int, str]) -> str:
    first_line = min(numbered_lines)
    last_line = max(numbered_lines)
    return "\n".join(
        numbered_lines.get(line, "") for line in range(first_line, last_line + 1)
    )


def _extract_python_symbols(numbered_lines: dict[int, str]) -> list[str]:
    source = _join_source(numbered_lines)
    visitor = _StructureVisitor()
    try:
        tree = ast.parse(source)
    except (IndentationError, SyntaxError):
        return _fallback_symbols(source)
    visitor.visit(tree)
    return visitor.symbols


def _fallback_symbols(source: str) -> list[str]:
    """Extrai definições e chamadas quando um chunk isolado não forma uma AST completa."""

    candidates = re.findall(
        r"\b(?:class|interface|enum|record|struct|def|fn|func)\s+([A-Za-z_]\w*)|"
        r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(|"
        r"\b([A-Z][A-Z0-9_]+)\b|"
        r"\b(start_line|end_line|chunk_size|overlap|top_k|batch_size)\b",
        source,
    )
    symbols: list[str] = []
    for groups in candidates:
        symbol = next((group for group in groups if group), "")
        if symbol and symbol not in CONTROL_FLOW_NAMES and symbol not in symbols:
            symbols.append(symbol)
    return symbols


# Compatibilidade para consumidores da primeira versão, restrita a Python.
PythonStructure = CodeStructure


def extract_python_structures(evidence_lines: list[str]) -> list[CodeStructure]:
    return extract_code_structures(
        [line for line in evidence_lines if line.split(":", 1)[0].endswith(".py")]
    )
