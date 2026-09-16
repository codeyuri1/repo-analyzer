import re
from collections import Counter

NUMBERED_LINE = re.compile(r"^(?P<number>\d+): (?P<content>.*)$")
LANGUAGES = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".go": "Go",
    ".rs": "Rust",
}


def build_project_overview(paths: set[str], documents: dict[str, str]) -> str:
    """Produz uma visão geral rastreável sem depender da geração da LLM."""

    readme_path = next(
        (path for path in documents if path.rsplit("/", 1)[-1].lower().startswith("readme")),
        None,
    )
    readme_lines = _numbered_lines(documents.get(readme_path, ""))
    objective = _first_prose_line(readme_lines)
    objective_reference = _reference(readme_path, readme_lines, objective)

    language_counts = Counter(
        language
        for path in paths
        for suffix, language in LANGUAGES.items()
        if path.lower().endswith(suffix)
    )
    languages = ", ".join(language for language, _ in language_counts.most_common())

    manifest_path = next(
        (path for path in documents if path.rsplit("/", 1)[-1] in {"pyproject.toml", "package.json"}),
        None,
    )
    manifest_lines = _numbered_lines(documents.get(manifest_path, ""))
    dependencies = _dependency_names(manifest_lines)
    technology_details = languages or "Não identificadas pelos arquivos indexados"
    if dependencies:
        technology_details += ". Dependências declaradas: " + ", ".join(dependencies[:8])

    modules = sorted(
        path
        for path in paths
        if path.startswith("src/")
        and path.endswith(tuple(LANGUAGES))
        and not path.endswith(("/__init__.py", "/__init__.js", "/__init__.ts"))
    )
    module_lines = "\n".join(f"- `{path}`" for path in modules[:20])
    if not module_lines:
        module_lines = "- Nenhum módulo em `src/` foi identificado."

    entrypoint = _find_entrypoint(manifest_lines)
    flow_lines = []
    if entrypoint:
        line_number, command, target = entrypoint
        flow_lines.append(
            f"1. O comando `{command}` aponta para `{target}` "
            f"(`{manifest_path}:{line_number}`)."
        )
    main_path = next(
        (path for path in documents if path.rsplit("/", 1)[-1] in {"main.py", "app.py", "server.py"}),
        None,
    )
    main_lines = _numbered_lines(documents.get(main_path, ""))
    flow_steps = _main_flow_steps(main_path, main_lines)
    start = len(flow_lines) + 1
    flow_lines.extend(
        f"{index}. {description}" for index, description in enumerate(flow_steps, start=start)
    )
    if not flow_lines:
        flow_lines.append("1. O ponto de entrada não foi confirmado nos arquivos selecionados.")

    commands = _execution_commands(readme_lines)
    command_lines = "\n".join(f"- `{command}`" for command in commands)
    if not command_lines:
        command_lines = "- O README não informa um comando de execução."

    objective_text = objective or "O objetivo não está documentado no README selecionado."
    if objective_reference:
        objective_text += f" (`{objective_reference}`)"

    return (
        "## Objetivo\n\n"
        f"{objective_text}\n\n"
        "## Tecnologias\n\n"
        f"{technology_details}.\n\n"
        "## Módulos principais\n\n"
        f"{module_lines}\n\n"
        "Os nomes acima são o inventário dos módulos; responsabilidades não são inferidas "
        "apenas pelo nome.\n\n"
        "## Fluxo de execução\n\n"
        + "\n".join(flow_lines)
        + "\n\n## Como executar\n\n"
        + command_lines
    )


def _numbered_lines(content: str) -> list[tuple[int, str]]:
    lines = []
    for raw_line in content.splitlines():
        match = NUMBERED_LINE.match(raw_line)
        if match:
            lines.append((int(match.group("number")), match.group("content")))
    return lines


def _first_prose_line(lines: list[tuple[int, str]]) -> str | None:
    in_code_block = False
    for _, line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped or stripped.startswith(("#", "-", "[")):
            continue
        return stripped
    return None


def _reference(
    path: str | None,
    lines: list[tuple[int, str]],
    content: str | None,
) -> str | None:
    if not path or not content:
        return None
    return next((f"{path}:{number}" for number, line in lines if line == content), None)


def _dependency_names(lines: list[tuple[int, str]]) -> list[str]:
    names = []
    in_dependencies = False
    for _, line in lines:
        stripped = line.strip()
        if stripped.startswith("dependencies") and "[" in stripped:
            in_dependencies = True
            continue
        if in_dependencies and stripped.startswith("]"):
            break
        if in_dependencies:
            match = re.search(r'["\']([A-Za-z0-9_.-]+)', stripped)
            if match:
                names.append(match.group(1))
    return names


def _find_entrypoint(
    lines: list[tuple[int, str]],
) -> tuple[int, str, str] | None:
    in_scripts = False
    for number, line in lines:
        stripped = line.strip()
        if stripped in {"[project.scripts]", '"scripts": {'}:
            in_scripts = True
            continue
        if in_scripts:
            match = re.match(r'["\']?([\w-]+)["\']?\s*[:=]\s*["\']([^"\']+)', stripped)
            if match:
                return number, match.group(1), match.group(2)
            if stripped.startswith("[") or stripped == "}":
                break
    return None


def _main_flow_steps(path: str | None, lines: list[tuple[int, str]]) -> list[str]:
    if not path:
        return []
    operations = (
        ("get_settings(", "carrega e valida a configuração"),
        ("build_tool_agent(", "constrói o agente e indexa o repositório"),
        ("args.eval", "seleciona a suíte de evals quando `--eval` é usado"),
        ("args.web", "inicia a interface Gradio quando `--web` é usado"),
        ("run_chat(", "inicia o chat no terminal no modo padrão"),
    )
    steps = []
    for marker, description in operations:
        match = next(((number, line) for number, line in lines if marker in line), None)
        if match:
            steps.append(f"{description} (`{path}:{match[0]}`).")
    return steps


def _execution_commands(lines: list[tuple[int, str]]) -> list[str]:
    commands = []
    for _, line in lines:
        stripped = line.strip()
        if stripped.startswith(
            ("uv run ", "python ", "npm ", "pnpm ", "yarn ", "cargo run", "go run")
        ) and stripped not in commands:
            commands.append(stripped)
    return commands[:5]
