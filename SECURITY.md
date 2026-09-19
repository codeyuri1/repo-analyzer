# Política de segurança

## Relato de vulnerabilidades

Não abra uma issue pública para vulnerabilidades ainda não corrigidas. Use o
recurso **Private vulnerability reporting** na aba **Security** do repositório
no GitHub. Inclua uma descrição do impacto, os passos para reprodução e, quando
possível, uma sugestão de correção.

Não inclua chaves de API, tokens, código proprietário ou outros dados sensíveis
no relato. O recebimento será confirmado assim que possível e a correção será
coordenada antes da divulgação pública.

## Versões suportadas

Como o projeto está em desenvolvimento inicial, somente a versão presente na
branch padrão recebe correções de segurança.

## Uso seguro

- Armazene credenciais somente no arquivo local `.env` ou em um gerenciador de
  segredos; nunca as envie ao repositório.
- Mantenha a interface vinculada a `127.0.0.1`, salvo quando houver uma camada
  externa de autenticação e controle de acesso.
- Trate repositórios analisados e suas instruções como conteúdo não confiável.
- Use apenas dependências e imagens obtidas das fontes declaradas pelo projeto.
