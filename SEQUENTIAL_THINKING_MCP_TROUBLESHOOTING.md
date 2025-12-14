# Troubleshooting: Sequential Thinking MCP Server no Windows

## Problema Identificado

Ao tentar instalar o MCP server `sequential-thinking` no Claude Code, o servidor apresentou status "failed" na interface `/mcp`.

## Causa Raiz

**Nome incorreto do pacote npm**: O comando inicial utilizava `@anthropic-sdk/sequential-thinking`, mas o pacote correto é `@modelcontextprotocol/server-sequential-thinking`.

### Pacotes Testados

| Pacote | Status | Observação |
|--------|--------|------------|
| `@anthropic-sdk/sequential-thinking` | ❌ Falha | Pacote não existe |
| `@modelcontextprotocol/server-sequential-thinking` | ✅ Correto | Pacote oficial |

## Solução Implementada

### 1. Configuração Correta (.mcp.json)

```json
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "stdio",
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ],
      "env": {}
    }
  }
}
```

### 2. Detalhes Importantes para Windows

#### Wrapper cmd /c Obrigatório
No Windows nativo (não WSL), é **essencial** usar o wrapper `cmd /c` antes do `npx`:

```json
"command": "cmd",
"args": ["/c", "npx", "-y", "..."]
```

#### Configurações Alternativas

**Opção 1: NPX (Recomendado para desenvolvimento)**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

**Opção 2: Docker (Produção/Isolamento)**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sequentialthinking"]
    }
  }
}
```

## Recursos do Sequential Thinking MCP Server

### Ferramenta Principal: `sequential_thinking`

Facilita raciocínio estruturado passo a passo para resolução de problemas complexos.

#### Parâmetros de Entrada

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `thought` | string | Etapa atual de raciocínio |
| `nextThoughtNeeded` | boolean | Se continuação é necessária |
| `thoughtNumber` | integer | Identificador da etapa |
| `totalThoughts` | integer | Contagem projetada de etapas |
| `isRevision` | boolean | Indica reconsideração de trabalho anterior |
| `revisesThought` | integer | Qual etapa está sendo reconsiderada |
| `branchFromThought` | integer | Onde raciocínio alternativo começa |
| `branchId` | string | Identificador de ramificação |
| `needsMoreThoughts` | boolean | Sinaliza necessidade de expansão |

### Casos de Uso Ideais

- ✅ Decomposição sistemática de problemas difíceis
- ✅ Planejamento estratégico com refinamento iterativo
- ✅ Análise exigindo capacidade de correção de curso
- ✅ Situações com escopo inicialmente incerto
- ✅ Tarefas multi-etapas mantendo continuidade contextual
- ✅ Cenários requerendo filtragem de informações irrelevantes

## Troubleshooting Avançado

### Problema: Servidor falha ao inicializar

**Diagnóstico:**
```bash
# Verificar versões do Node.js
node --version  # Requer >= 18.0.0
npm --version   # Requer >= 8.0.0
npx --version

# Testar servidor manualmente
cmd /c npx -y @modelcontextprotocol/server-sequential-thinking
```

**Soluções:**

1. **Timeout durante inicialização**
```bash
# Aumentar timeout (padrão: 10 segundos)
set MCP_TIMEOUT=30000
claude
```

2. **Desabilitar logging de pensamentos**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-sequential-thinking"],
      "env": {
        "DISABLE_THOUGHT_LOGGING": "true"
      }
    }
  }
}
```

3. **Debug detalhado**
```bash
# Visualizar logs de inicialização
claude --debug

# Localização dos logs (Windows)
%APPDATA%\Local\claude-cli-nodejs\logs\
```

### Problema: Servidor trava/crash após primeira utilização

**Issue conhecida**: Claude Code Sonnet 4.5 pode crashar aleatoriamente ao usar sequential-thinking.

**Soluções temporárias**:
- Reiniciar a sessão do Claude Code
- Limpar contexto da conversa
- Usar em sessões menores/focadas

**Referências**:
- [Issue #713 - Claude Code gets stuck](https://github.com/modelcontextprotocol/servers/issues/713)
- [Issue #2792 - Random crash with sequentialthinking](https://github.com/modelcontextprotocol/servers/issues/2792)

### Problema: ENOENT errors no Windows

**Causa**: npx não encontrado no PATH ou problemas com variáveis de ambiente.

**Solução**:
```bash
# Verificar PATH do Node.js
where node
where npx

# Reinstalar Node.js se necessário
# Download: https://nodejs.org/
```

## Instalação via Claude Code CLI

### Método 1: Comando automático (Project scope)
```bash
claude mcp add --transport stdio sequential-thinking --scope project -- cmd /c npx -y @modelcontextprotocol/server-sequential-thinking
```

### Método 2: Comando automático (User scope)
```bash
claude mcp add --transport stdio sequential-thinking --scope user -- cmd /c npx -y @modelcontextprotocol/server-sequential-thinking
```

### Método 3: Configuração manual
1. Criar/editar `.mcp.json` na raiz do projeto
2. Adicionar configuração conforme mostrado acima
3. Reiniciar Claude Code

## Verificação de Instalação

```bash
# Listar servidores MCP configurados
claude mcp list

# Obter detalhes do servidor
claude mcp get sequential-thinking

# Verificar status dentro do Claude Code
/mcp
```

## Build a partir do Source (Opcional)

### Docker Build
```bash
docker build -t mcp/sequentialthinking -f src/sequentialthinking/Dockerfile .
```

## Requisitos do Sistema

### Software
- **Node.js**: >= 18.0.0 (versões LTS recomendadas)
- **npm**: >= 8.0.0
- **Claude Code**: Versão atual instalada
- **Windows**: 10/11 (nativo ou WSL2)

### Alternativos
- **Docker**: Para método de instalação via container

## Referências e Documentação

### Oficial
- [Sequential Thinking MCP Server - NPM](https://www.npmjs.com/package/@modelcontextprotocol/server-sequential-thinking)
- [GitHub - MCP Servers Repository](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)
- [Claude Code MCP Documentation](https://code.claude.com/docs/en/mcp.md)

### Troubleshooting
- [Ultimate Guide to Claude MCP Servers & Setup](https://generect.com/blog/claude-mcp/)
- [How to Debug MCP Server with Anthropic Inspector](https://snyk.io/articles/how-to-debug-mcp-server-with-anthropic-inspector/)
- [Claude Code MCP Setup Guide for Windows](https://lobehub.com/mcp/bunprinceton-claude-mcp-windows-guide)

### Issues Conhecidas
- [MCP Server Fails to Connect on Windows](https://github.com/ruvnet/claude-flow/issues/601)
- [Claude Code gets stuck when using sequentialthinking](https://github.com/modelcontextprotocol/servers/issues/713)
- [Claude Code Sonnet 4.5 randomly crash](https://github.com/modelcontextprotocol/servers/issues/2792)

## Licença

O Sequential Thinking MCP Server é distribuído sob licença MIT, permitindo uso, modificação e distribuição livres sob condições especificadas.

---

**Status do Documento**: Atualizado em 14/12/2025
**Versão**: 1.0
**Autor**: Claude Code Assistant
**Ambiente de Teste**: Windows 10/11, Node.js v22.18.0, npm 10.9.3
