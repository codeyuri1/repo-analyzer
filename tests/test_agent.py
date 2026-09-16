import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from repo_agent_chat.agent import ToolAgent
from repo_agent_chat.app.config import Settings


def final_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_response(
    name: str,
    arguments: str,
    call_id: str = "call-1",
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    call = SimpleNamespace(id=call_id, function=function)
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def stream_chunks(*parts: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=part))])
        for part in parts
    ]


def test_agent_executa_tool_e_devolve_resultado_ao_modelo() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        tool_response("list_files", '{"prefix":"src/"}'),
        final_response("O código está em `src/main.py:1`."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":true,"result":["src/main.py"]}'
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Onde está o código principal?")

    assert answer == "O código está em `src/main.py:1`."
    assert agent.last_tool_calls == ["list_files"]
    tools.execute.assert_called_once_with("list_files", '{"prefix":"src/"}')
    second_messages = client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert second_messages[-3]["role"] == "assistant"
    assert second_messages[-2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"ok":true,"result":["src/main.py"]}',
    }
    assert second_messages[-1]["role"] == "system"
    assert "Não exponha o JSON bruto" in second_messages[-1]["content"]


def test_visao_geral_inicia_workflow_deterministico() -> None:
    client = Mock()
    client.chat.completions.create.return_value = final_response(
        "## Objetivo\nAnalisar repositórios (`src/main.py:10`).\n\n"
        "## Tecnologias\nPython.\n\n"
        "## Módulos principais\n`main.py`.\n\n"
        "## Fluxo de execução\nA função `main` inicia.\n\n"
        "## Como executar\nUse o comando documentado."
    )
    tools = Mock()
    tools.execute.side_effect = [
        json.dumps({"ok": True, "result": ["src/main.py"]}),
        json.dumps(
            {
                "ok": True,
                "result": {
                    "path": "src/main.py",
                    "start_line": 1,
                    "end_line": 20,
                    "total_lines": 20,
                    "has_more": False,
                    "content": "10: def main():",
                },
            }
        ),
    ]
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Explique o projeto")

    assert "Objetivo" in answer
    assert agent.last_tool_calls == ["list_files", "read_file"]
    assert tools.execute.call_args_list[0].args == ("list_files", '{"prefix": ""}')
    read_arguments = json.loads(tools.execute.call_args_list[1].args[1])
    assert read_arguments == {
        "path": "src/main.py",
        "start_line": 1,
        "end_line": 160,
    }
    client.chat.completions.create.assert_not_called()


def test_pedido_natural_de_seguranca_executa_tool_deterministicamente() -> None:
    client = Mock()
    tools = Mock()
    tools.execute.return_value = json.dumps(
        {"ok": True, "result": {"findings": [], "count": 0, "prefix": ""}}
    )
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Verifique a segurança deste repositório")

    assert "Nenhum possível achado" in answer
    tools.execute.assert_called_once_with(
        "analyze_vulnerabilities",
        '{"prefix": "", "max_findings": 50}',
    )
    client.chat.completions.create.assert_not_called()


def test_revisor_transmite_tokens_reais_quando_observador_esta_ativo() -> None:
    client = Mock()
    client.chat.completions.create.return_value = stream_chunks("Resposta ", "final.")
    tokens: list[str] = []
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=client)
    agent._turn_token_event = tokens.append

    answer = agent._revise_answer("Pergunta", "Rascunho", ["revise"])

    assert answer == "Resposta final."
    assert tokens == ["Resposta ", "final."]
    assert client.chat.completions.create.call_args.kwargs["stream"] is True


def test_agent_emite_eventos_de_tracing() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        tool_response("list_files", "{}"),
        final_response("Arquivos listados."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":true,"result":[]}'
    observer = Mock()
    agent = ToolAgent(
        Settings(_env_file=None),
        tools,
        client=client,
        on_tool_event=observer,
    )

    agent.ask("Liste os arquivos")

    assert observer.call_count == 2
    started = observer.call_args_list[0].args[0]
    completed = observer.call_args_list[1].args[0]
    assert started.phase == "started"
    assert completed.phase == "completed"
    assert completed.success is True
    assert completed.duration_ms is not None


def test_agent_recusa_resposta_enquanto_leitura_estiver_parcial() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        tool_response(
            "read_file",
            '{"path":"repository.py","start_line":1,"end_line":50}',
        ),
        final_response("Não há evidências."),
        final_response("A validação ocorre em `repository.py:63`."),
    ]
    tools = Mock()
    tools.execute.side_effect = [
        json.dumps(
            {
                "ok": True,
                "result": {
                    "path": "repository.py",
                    "start_line": 1,
                    "end_line": 50,
                    "total_lines": 113,
                    "has_more": True,
                    "next_start_line": 51,
                    "content": (
                        "10: if not resolved.is_relative_to(root):\n"
                        "11:     raise ValueError()"
                    ),
                },
            }
        ),
        json.dumps(
            {
                "ok": True,
                "result": {
                    "path": "repository.py",
                    "start_line": 51,
                    "end_line": 113,
                    "total_lines": 113,
                    "has_more": False,
                    "next_start_line": None,
                    "content": "10: if not safe:\n11:     raise ValueError()",
                },
            }
        ),
    ]
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Onde ocorre a leitura segura?")

    assert answer == "A validação ocorre em `repository.py:63`."
    assert client.chat.completions.create.call_count == 3
    assert tools.execute.call_count == 2
    continuation_arguments = json.loads(tools.execute.call_args_list[1].args[1])
    assert continuation_arguments == {
        "path": "repository.py",
        "start_line": 51,
        "end_line": 113,
    }
    rejected_messages = client.chat.completions.create.call_args_list[2].kwargs[
        "messages"
    ]
    guardrail_message = next(
        message["content"]
        for message in rejected_messages
        if message.get("content")
        and "orquestrador adicionou a continuação" in message["content"]
    )
    assert "leitura parcial" in guardrail_message


def test_agent_pede_uma_revisao_para_resposta_incompleta() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        tool_response("list_files", "{}"),
        tool_response(
            "read_file",
            '{"path":"src/repository.py","start_line":1,"end_line":20}',
            call_id="call-2",
        ),
        final_response("A função é segura.\n```python\ndef read(): ...\n```"),
        final_response(
            "Módulos: `agent.py`, `repository.py` e `tools.py`. "
            "A validação ocorre em `src/repository.py:10-20`."
        ),
    ]
    tools = Mock()
    tools.execute.side_effect = [
        json.dumps(
            {
                "ok": True,
                "result": [
                    "src/agent.py",
                    "src/repository.py",
                    "src/tools.py",
                ],
            }
        ),
        json.dumps(
            {
                "ok": True,
                "result": [
                    {
                        "path": "src/repository.py",
                        "start_line": 10,
                        "end_line": 11,
                        "content": (
                            "if not resolved.is_relative_to(root):\n"
                            "    raise ValueError()"
                        ),
                    }
                ],
            }
        ),
        json.dumps(
            {
                "ok": True,
                "result": {
                    "path": "src/repository.py",
                    "start_line": 1,
                    "end_line": 20,
                    "total_lines": 20,
                    "has_more": False,
                    "next_start_line": None,
                    "content": (
                        "10: if not resolved.is_relative_to(root):\n"
                        "11:     raise ValueError()"
                    ),
                },
            }
        ),
    ]
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Liste os módulos e explique a leitura segura")

    assert "src/repository.py:10" in answer
    assert "src/repository.py:11" in answer
    assert client.chat.completions.create.call_count == 4
    revision_messages = client.chat.completions.create.call_args_list[3].kwargs[
        "messages"
    ]
    revision_prompt = revision_messages[-1]["content"]
    assert "inclua ao menos duas citações" in revision_prompt
    assert "agent.py" in revision_prompt
    assert "remova o bloco de código" in revision_prompt
    assert "src/repository.py:1-20" in revision_prompt
    assert "src/repository.py:10: if not resolved.is_relative_to" in revision_prompt
    assert "Preserve todos os nomes técnicos" in revision_messages[0]["content"]
    assert "tools" not in client.chat.completions.create.call_args_list[3].kwargs


def test_agent_pede_citacao_mais_precisa() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_paths = {"src/repository.py"}

    issues = agent._answer_issues(
        "Explique a implementação",
        "A validação está em `src/repository.py:1-50`.",
    )

    assert any("no máximo 20 linhas" in issue for issue in issues)


def test_agent_remove_bloco_de_codigo_da_resposta_revisada() -> None:
    answer = (
        "Explicação antes.\n\n```python\ndef unsafe(): pass\n```\n\nExplicação depois."
    )

    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    sanitized = agent._sanitize_answer(answer)

    assert sanitized == "Explicação antes.\n\nExplicação depois."
    assert "unsafe" not in sanitized
    assert "```" not in sanitized


def test_agent_remove_alegacao_de_seguranca_sem_evidencia() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_excerpts = [
        "repository.py:63: if not resolved_file.is_relative_to(resolved_root):",
    ]
    answer = (
        "O caminho precisa ficar dentro da raiz. Isso evita ataques de injeção de código. "
        "A validação bloqueia caminhos externos."
    )

    sanitized = agent._sanitize_answer(answer, "Como funciona a leitura segura?")

    assert "injeção de código" not in sanitized
    assert "caminhos externos" in sanitized


def test_agent_normaliza_citacao_em_linguagem_natural() -> None:
    answer = "O método aparece na linha 50 do arquivo `src/app.py`."

    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    sanitized = agent._sanitize_answer(answer)

    assert sanitized == "O método aparece na `src/app.py:50`."


def test_agent_remove_secao_interna_e_adiciona_fontes_selecionadas() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_excerpts = [
        "src/app.py:10: def run():",
        "src/app.py:12: return result",
    ]
    answer = (
        "A função executa o fluxo.\n\n"
        "Faixas de evidência disponíveis:\nsrc/app.py:1-20\n"
        "Correções obrigatórias:\n- cite fontes"
    )

    sanitized = agent._sanitize_answer(answer)

    assert "Faixas de evidência" not in sanitized
    assert "Correções obrigatórias" not in sanitized
    assert "Fontes:" in sanitized
    assert "`src/app.py:10`" in sanitized


def test_agent_remove_secao_interna_com_titulo_markdown() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())

    sanitized = agent._sanitize_answer(
        "Resposta útil.\n\n### Faixas de Evidência Disponíveis\n- conteúdo interno"
    )

    assert sanitized == "Resposta útil."


def test_agent_remove_paragrafo_com_simbolo_inventado() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_excerpts = [
        "src/chunking.py:21: def chunk_source_file(source_file, chunk_size):",
    ]
    answer = (
        "A função `divide_chunks` faz a divisão.\n\n"
        "A chamada `RagAssistant().generate_response(query)` finaliza.\n\n"
        "A função `chunk_source_file` recebe o tamanho."
    )

    sanitized = agent._sanitize_answer(answer)

    assert "divide_chunks" not in sanitized
    assert "generate_response" not in sanitized
    assert "chunk_source_file" in sanitized


def test_agent_restringe_tools_a_implementacao_da_classe_perguntada() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_excerpts = [
        "src/rag.py:20: class RagAssistant:",
        "src/rag.py:38: results = self._retriever.search(question)",
        "src/agent.py:10: semantic_search(query)",
    ]
    answer = (
        "O `RagAssistant` chama semantic_search.\n\n"
        "O `RagAssistant` usa o retriever."
    )

    sanitized = agent._sanitize_answer(answer, "Como o RagAssistant funciona?")

    assert "semantic_search" not in sanitized
    assert "usa o retriever" in sanitized


def test_agent_prioriza_verificacao_em_vez_da_definicao_da_constante() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())
    agent.last_evidence_excerpts = [
        "repository.py:34: MAX_FILE_SIZE_BYTES = 500_000",
        "repository.py:69: if size > MAX_FILE_SIZE_BYTES:",
    ]

    selected = agent._select_evidence_excerpts(
        "Como o tamanho é validado?",
        "Usa MAX_FILE_SIZE_BYTES.",
    )

    assert selected[0] == "repository.py:69: if size > MAX_FILE_SIZE_BYTES:"


def test_agent_busca_evidencia_antes_de_aceitar_resposta_sobre_repo() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        final_response("Acho que funciona assim."),
        final_response("A busca encontrou a implementação."),
        final_response("Funciona em `main.py:1`."),
    ]
    tools = Mock()
    tools.execute.return_value = json.dumps(
        {
            "ok": True,
            "result": [
                {
                    "path": "main.py",
                    "start_line": 1,
                    "end_line": 1,
                    "content": "def main(): pass",
                }
            ],
        }
    )
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Como funciona o código de autenticação?")

    assert answer == "Funciona em `main.py:1`."
    assert [call.args[0] for call in tools.execute.call_args_list] == [
        "semantic_search",
        "read_file",
    ]


def test_agent_nao_aceita_listagem_como_evidencia_de_comportamento() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        tool_response("list_files", "{}"),
        final_response("O nome do arquivo indica o comportamento."),
        final_response("A busca encontrou a implementação."),
        final_response("A implementação está em `repository.py:10`."),
    ]
    tools = Mock()
    tools.execute.side_effect = [
        json.dumps({"ok": True, "result": ["repository.py"]}),
        json.dumps(
            {
                "ok": True,
                "result": [
                    {
                        "path": "repository.py",
                        "start_line": 1,
                        "end_line": 20,
                        "content": "10: def read_source_file(): pass",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "ok": True,
                "result": {
                    "path": "repository.py",
                    "start_line": 1,
                    "end_line": 20,
                    "content": "10: def read_source_file(): pass",
                },
            }
        ),
    ]
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Como o projeto lê arquivos?")

    assert "repository.py:10" in answer
    assert agent.last_tool_calls == ["list_files", "semantic_search", "read_file"]


def test_agent_responde_sem_tool_quando_nao_for_necessaria() -> None:
    client = Mock()
    client.chat.completions.create.return_value = final_response("Olá!")
    tools = Mock()
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    assert "".join(agent.stream("Olá")) == "Olá!"
    tools.execute.assert_not_called()


def test_agent_interpreta_tool_devolvida_como_json_textual() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        final_response(
            'Vou consultar os arquivos.\n{"name":"list_files","arguments":{}}'
        ),
        final_response("Os módulos foram listados."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":false,"error":"prefix ausente"}'
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Liste os módulos")

    assert answer == "Os módulos foram listados."
    assert agent.last_tool_calls == ["list_files"]
    tools.execute.assert_called_once_with("list_files", "{}")
    second_messages = client.chat.completions.create.call_args_list[1].kwargs[
        "messages"
    ]
    assert second_messages[-3]["content"] is None
    assert second_messages[-2]["tool_call_id"] == "text-tool-call-1"


def test_agent_interpreta_tool_em_bloco_markdown() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        final_response('```json\n{"name":"list_files","arguments":{}}\n```'),
        final_response("Arquivos listados."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":true,"result":["main.py"]}'
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Liste os arquivos")

    assert answer == "Arquivos listados."
    tools.execute.assert_called_once_with("list_files", "{}")


def test_agent_interpreta_tool_response_textual_com_narracao() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        final_response(
            "Vou ler o arquivo.\n"
            '<tool_response>{"name":"read_file","arguments":'
            '{"path":"src/main.py","start_line":1,"end_line":10}}'
            "</tool_response>"
        ),
        final_response("A leitura está em `src/main.py:1`."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":true,"result":{}}'
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Onde ocorre a leitura?")

    assert answer == "A leitura está em `src/main.py:1`."
    tools.execute.assert_called_once_with(
        "read_file",
        '{"path": "src/main.py", "start_line": 1, "end_line": 10}',
    )


def test_agent_repara_uma_unica_chave_final_ausente() -> None:
    client = Mock()
    client.chat.completions.create.side_effect = [
        final_response('{"name":"list_files","arguments":{}'),
        final_response("Arquivos listados."),
    ]
    tools = Mock()
    tools.execute.return_value = '{"ok":true,"result":["main.py"]}'
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Liste os arquivos")

    assert answer == "Arquivos listados."
    tools.execute.assert_called_once_with("list_files", "{}")


def test_agent_nao_executa_json_que_nao_seja_chamada_exata_de_tool() -> None:
    client = Mock()
    client.chat.completions.create.return_value = final_response(
        '{"name":"list_files","arguments":{},"exemplo":true}'
    )
    tools = Mock()
    agent = ToolAgent(Settings(_env_file=None), tools, client=client)

    answer = agent.ask("Mostre um exemplo de JSON")

    assert '"exemplo":true' in answer
    tools.execute.assert_not_called()


def test_agent_rejeita_pergunta_vazia() -> None:
    agent = ToolAgent(Settings(_env_file=None), Mock(), client=Mock())

    with pytest.raises(ValueError, match="não pode estar vazia"):
        agent.ask("   ")


def test_agent_limita_rodadas_de_tools() -> None:
    client = Mock()
    client.chat.completions.create.return_value = tool_response(
        "list_files", '{"prefix":""}'
    )
    settings = Settings(_env_file=None, max_tool_rounds=2)
    agent = ToolAgent(settings, Mock(), client=client)

    with pytest.raises(RuntimeError, match="excedeu o limite"):
        agent.ask("Continue chamando tools")

    assert client.chat.completions.create.call_count == 2
