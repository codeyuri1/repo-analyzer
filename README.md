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
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5-coder:7b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
OLLAMA_TIMEOUT_SECONDS=300
```

A chave da OpenAI não é necessária. A biblioteca usa a API compatível do
Ollama em `localhost`; internamente é enviado apenas um valor fictício exigido
pelo cliente.

Para usar a API da OpenAI, altere somente a configuração:

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_TIMEOUT_SECONDS=300
```

Os modelos podem ser trocados pelas variáveis de ambiente sem modificar o código.
Não versione o arquivo `.env` nem exponha a chave na interface ou em demonstrações.

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

## Demonstração

O fluxo real de carregamento de um fixture e investigação de arquitetura está
gravado em [`assets/demo/repo-agent-chat-demo.webm`](assets/demo/repo-agent-chat-demo.webm).

Para regravar localmente, inicie a interface na porta `7861` apontando para
`tests/fixtures/simple_api` e execute:

```bash
uv run --with playwright python scripts/record_demo.py
```

Na tela inicial, informe ou confirme o caminho/URL e clique em **Carregar
repositório**. Somente nesse momento o clone temporário e a indexação em memória
são iniciados. Carregar outra fonte limpa a conversa e descarta a sessão
anterior depois que a nova indexação termina com sucesso.

## Docker

O container executa somente a aplicação. O Ollama continua no host e é acessado
por `host.docker.internal`; modelos e drivers de GPU não são incluídos na imagem.

Para analisar o próprio projeto:

```bash
docker compose up --build
```

Para montar outro repositório local, sempre como somente leitura:

```bash
REPOSITORY_PATH=/caminho/do/repositorio docker compose up --build
```

A interface fica disponível em `http://127.0.0.1:7860`. Para trocar a porta:

```bash
APP_PORT=8080 docker compose up --build
```

Com OpenAI, configure `AI_PROVIDER=openai` e `OPENAI_API_KEY` no `.env`. O
container roda sem root, com filesystem somente leitura, capabilities removidas
e `/tmp` efêmero para clones, índice da sessão e diagramas.

Ao usar a OpenAI, trechos recuperados do repositório e as perguntas da conversa
são enviados à API configurada. Não use esse modo com código confidencial sem
antes avaliar as políticas e os controles de dados aplicáveis. Com Ollama, o
processamento permanece na infraestrutura local configurada pelo usuário.

### Ubuntu no WSL 2

Com Docker Desktop no Windows, habilite **Settings > Resources > WSL
Integration**, ative a distribuição Ubuntu utilizada e selecione **Apply &
restart**. Em um novo terminal do Ubuntu, confirme a integração:

```bash
docker version
docker compose version
```

Mantenha o projeto no filesystem Linux (por exemplo,
`~/projetos/repo-agent-chat`) em vez de `/mnt/c/...`; isso evita problemas de
permissão e melhora o desempenho dos bind mounts. Então execute normalmente:

```bash
cp .env.example .env
docker compose up --build
```

O Compose publica a interface somente em `127.0.0.1` por padrão e adiciona o
gateway `host.docker.internal`, que permite ao container acessar o Ollama no
host. Se o Ollama estiver no Windows, confirme que ele aceita conexões vindas
do Docker/WSL; se necessário, configure no Windows `OLLAMA_HOST=0.0.0.0:11434`
e reinicie o Ollama. Essa configuração escuta em todas as interfaces: mantenha
a porta 11434 bloqueada para redes não confiáveis pelo Firewall do Windows. A
URL usada pelo container pode ser sobrescrita no `.env`:

```dotenv
CONTAINER_OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
```

Para disponibilizar a interface na rede local, faça isso explicitamente:

```dotenv
APP_HOST=0.0.0.0
```

O `bubblewrap` é um requisito do ambiente de sandbox no Ubuntu/WSL e não deve
ser instalado no container desta aplicação. Instale-o na distribuição:

```bash
sudo apt update
sudo apt install bubblewrap
```

## Segurança e publicação

- Nunca versione o `.env`; use apenas o `.env.example` com valores fictícios.
- Revise `git diff --cached` antes de cada push para evitar credenciais e dados
  locais adicionados por engano.
- Repositórios fornecidos para análise podem conter código não confiável. A
  aplicação somente lê os arquivos e não executa o código analisado.
- O conteúdo recuperado — incluindo código, comentários, documentação e nomes
  de arquivos — é tratado como dado não confiável. Linhas com indicadores de
  prompt injection são removidas antes de entrarem no contexto do modelo; esta
  é uma defesa em profundidade, não uma garantia absoluta contra conteúdo
  malicioso.
- A análise de vulnerabilidades cobre o repositório carregado inteiro e usa
  regras estáticas heurísticas. Ela pode ter falsos positivos e falsos
  negativos, portanto achados precisam de revisão manual e testes adicionais.
- Ao publicar a interface na rede com `APP_HOST=0.0.0.0`, use firewall ou proxy
  autenticado; a interface não implementa autenticação própria.

Para relatar uma vulnerabilidade, consulte [`SECURITY.md`](SECURITY.md).

Pull requests executam automaticamente lint, testes, build do pacote e um teste
de inicialização do Docker Compose com health check.

## Tools disponíveis

- `list_files`: navega pelos arquivos indexáveis.
- `read_file`: lê um intervalo validado de linhas.
- `semantic_search`: recupera chunks com busca híbrida.
- `analyze_vulnerabilities`: executa regras estáticas e retorna possíveis
  achados (como segredos no código, execução dinâmica, shell, desserialização,
  SQL dinâmico, TLS sem validação e indicadores de prompt injection), sem
  afirmar que o código está seguro.
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

O chat não usa uma lista fechada de perguntas. Os exemplos da interface funcionam
como atalhos para a demonstração, enquanto o agente aceita qualquer pergunta dentro
do domínio do repositório. Perguntas fora desse escopo são redirecionadas.

## Qualidade

```bash
uv run pytest
uv run ruff check .
# Requer provider configurado; não é executado pelo CI básico:
uv run repo-agent-chat --eval .
```

O CI básico executa `uv run pytest` e `uv run ruff check .`: são testes unitários,
de integração e evals determinísticos, incluindo roteamento de tools, fontes,
citações, símbolos e Mermaid. A suíte `--eval` cria casos a partir dos arquivos
do repositório carregado, em vez de pressupor uma linguagem, diretório ou
arquitetura específicos. A camada LLM é opcional e deve executar apenas quando
um provider estiver configurado; o judge recebe pergunta, resposta, evidências,
fontes e trace, nunca somente a resposta.

Para uma suíte de judge integrada ao pytest, reserve o marcador `llm`:

```bash
uv run pytest -m "not llm" -q  # CI básico
uv run pytest -m llm -q        # provider configurado
```

Uma descrição das responsabilidades, dependências e do fluxo interno está em
[`docs/architecture.md`](docs/architecture.md).

## Checklist de demonstração

1. Inicie com `uv run repo-agent-chat --web`.
2. Carregue um caminho local ou URL pública do GitHub.
3. Peça `Explique o projeto.` e confira as citações.
4. Execute a análise de possíveis vulnerabilidades.
5. Gere o Mermaid e faça download do `.mmd`.
6. Troque de repositório e confirme que o histórico foi limpo.
7. Encerre com `Ctrl+C`; clones, índice e artefatos temporários serão removidos.
