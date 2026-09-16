# Repo Agent Chat

Assistente local para investigar repositórios com Ollama, RAG e tools. O projeto
indexa o código, recupera evidências e exige citações antes de responder.

## Pré-requisitos

- Python 3.12 ou superior.
- `uv` para ambiente e dependências.
- Ollama em execução local.
- Git instalado para carregar repositórios públicos do GitHub.

## Modelos locais

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen3-embedding:0.6b
```

Configure o `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
```

A chave da OpenAI não é necessária. A biblioteca usa a API compatível do
Ollama em `localhost`; internamente é enviado apenas um valor fictício exigido
pelo cliente.

## Executar

Chat no terminal:

```bash
uv run repo-agent-chat .
```

Repositório público do GitHub:

```bash
uv run repo-agent-chat https://github.com/dono/repositorio
```

O clone é raso, fica no workspace temporário da sessão e é removido no
encerramento. URLs com credenciais, hosts diferentes, caminhos internos,
query strings ou protocolo HTTP são rejeitadas.

Interface Gradio:

```bash
uv run repo-agent-chat --web .
```

Abra `http://127.0.0.1:7860` no navegador.

Na tela inicial, informe ou confirme o caminho/URL e clique em **Carregar
repositório**. Somente nesse momento o clone temporário e a indexação em memória
são iniciados. Carregar outra fonte limpa a conversa e descarta a sessão
anterior depois que a nova indexação termina com sucesso.

## Tools disponíveis

- `list_files`: navega pelos arquivos indexáveis.
- `read_file`: lê um intervalo validado de linhas.
- `semantic_search`: recupera chunks com busca híbrida.
- `analyze_vulnerabilities`: executa regras estáticas e retorna possíveis
  achados, sem afirmar que o código está seguro.
- `generate_mermaid_diagram`: extrai imports e produz um grafo Mermaid de
  dependências entre módulos.

Na interface web, o diagrama é renderizado no chat e o botão de download recebe
uma cópia `.mmd`. Durante a sessão, a cópia fica em um workspace temporário:

```text
/tmp/repo-agent-chat-<sessão>/artifacts/dependency-diagram-<data>-<id>.mmd
```

O workspace, os diagramas e o índice em memória são descartados quando o
processo encerra. O repositório local analisado nunca é removido.

Exemplos de perguntas:

- `Analise possíveis vulnerabilidades no código-fonte.`
- `Gere um diagrama Mermaid das dependências entre os módulos.`

## Qualidade

```bash
uv run pytest
uv run ruff check .
uv run repo-agent-chat --eval .
```

Os testes verificam componentes determinísticos. Os evals executam conversas
reais com a LLM e avaliam uso de tools, groundedness, citações e formato.

## Checklist de demonstração

1. Inicie com `uv run repo-agent-chat --web`.
2. Carregue um caminho local ou URL pública do GitHub.
3. Peça `Explique o projeto.` e confira as citações.
4. Execute a análise de possíveis vulnerabilidades.
5. Gere o Mermaid e faça download do `.mmd`.
6. Troque de repositório e confirme que o histórico foi limpo.
7. Encerre com `Ctrl+C`; clones, índice e artefatos temporários serão removidos.
