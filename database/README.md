# SQLite MCP Server - Controle de Roadmap

Este diretório contém o banco de dados SQLite para controle de roadmap do projeto Synth Bot Buddy, integrado com o Claude Code via MCP (Model Context Protocol).

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Banco de Dados](#estrutura-do-banco-de-dados)
- [Instalação e Configuração](#instalação-e-configuração)
- [Uso com Claude Code](#uso-com-claude-code)
- [Queries Úteis](#queries-úteis)
- [Manutenção](#manutenção)
- [Referências](#referências)

## 🎯 Visão Geral

O banco de dados SQLite (`roadmap.db`) armazena e gerencia todas as tasks, milestones e progresso do projeto. Ele é acessível via:

1. **Claude Code MCP Server**: Permite que o Claude interaja diretamente com o banco
2. **Scripts Node.js**: Para automação e relatórios
3. **Clientes SQLite**: DB Browser, sqlite3 CLI, etc.

### Benefícios

- ✅ **Controle centralizado** de todas as tasks do roadmap
- ✅ **Histórico completo** de mudanças de status
- ✅ **Integração com IA** via Claude Code
- ✅ **Queries complexas** para análise de progresso
- ✅ **Triggers automáticos** para tracking de timestamps
- ✅ **Views prontas** para dashboards e relatórios

## 🗄️ Estrutura do Banco de Dados

### Tabelas Principais

#### 1. `roadmap_tasks`

Tabela principal contendo todas as tasks do roadmap.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | INTEGER | ID único auto-incrementado |
| `task_name` | TEXT | Nome da task |
| `description` | TEXT | Descrição detalhada |
| `status` | TEXT | Status: `todo`, `in_progress`, `completed`, `blocked`, `cancelled` |
| `priority` | TEXT | Prioridade: `low`, `medium`, `high`, `critical` |
| `category` | TEXT | Categoria: `frontend`, `backend`, `ml`, `infrastructure`, `documentation`, `testing` |
| `estimated_hours` | REAL | Horas estimadas |
| `actual_hours` | REAL | Horas reais gastas |
| `assigned_to` | TEXT | Responsável pela task |
| `dependencies` | TEXT | IDs de tasks dependentes (separados por vírgula) |
| `created_at` | DATETIME | Data de criação (automático) |
| `updated_at` | DATETIME | Última atualização (automático via trigger) |
| `started_at` | DATETIME | Quando iniciou (automático via trigger) |
| `completed_at` | DATETIME | Quando completou (automático via trigger) |
| `due_date` | DATE | Data limite |
| `notes` | TEXT | Notas adicionais |

#### 2. `task_history`

Histórico de mudanças de status das tasks.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | INTEGER | ID único |
| `task_id` | INTEGER | Referência para `roadmap_tasks.id` |
| `status_from` | TEXT | Status anterior |
| `status_to` | TEXT | Novo status |
| `changed_at` | DATETIME | Timestamp da mudança |
| `changed_by` | TEXT | Quem fez a mudança |
| `notes` | TEXT | Observações |

#### 3. `milestones`

Marcos importantes do projeto.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | INTEGER | ID único |
| `name` | TEXT | Nome do milestone |
| `description` | TEXT | Descrição |
| `target_date` | DATE | Data alvo |
| `status` | TEXT | Status: `pending`, `in_progress`, `completed`, `missed` |
| `created_at` | DATETIME | Data de criação |
| `completed_at` | DATETIME | Data de conclusão |

#### 4. `task_milestones`

Relação N:N entre tasks e milestones.

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `task_id` | INTEGER | ID da task |
| `milestone_id` | INTEGER | ID do milestone |

### Views (Consultas Prontas)

#### 1. `roadmap_progress`

Resumo de progresso por categoria.

```sql
SELECT * FROM roadmap_progress;
```

Retorna:
- `category`: Categoria da task
- `total_tasks`: Total de tasks
- `completed_tasks`: Tasks completadas
- `in_progress_tasks`: Tasks em progresso
- `todo_tasks`: Tasks a fazer
- `blocked_tasks`: Tasks bloqueadas
- `completion_percentage`: % de conclusão
- `total_estimated_hours`: Soma de horas estimadas
- `total_actual_hours`: Soma de horas reais

#### 2. `overdue_tasks`

Tasks atrasadas ordenadas por dias de atraso.

```sql
SELECT * FROM overdue_tasks;
```

Retorna tasks não concluídas com `due_date` já passado.

#### 3. `next_tasks`

Próximas tasks a fazer, ordenadas por prioridade.

```sql
SELECT * FROM next_tasks;
```

Útil para decidir o que trabalhar em seguida.

### Triggers (Automações)

1. **`update_roadmap_timestamp`**: Atualiza `updated_at` automaticamente
2. **`track_status_change`**: Registra mudanças de status no histórico
3. **`set_started_at`**: Define `started_at` quando status vira `in_progress`
4. **`set_completed_at`**: Define `completed_at` quando status vira `completed`

### Índices (Performance)

- `idx_roadmap_status`: Busca rápida por status
- `idx_roadmap_priority`: Busca rápida por prioridade
- `idx_roadmap_category`: Busca rápida por categoria
- `idx_roadmap_due_date`: Busca rápida por data limite
- `idx_task_history_task_id`: Busca rápida de histórico
- `idx_task_history_changed_at`: Busca rápida por data

## ⚙️ Instalação e Configuração

### Pré-requisitos

- Node.js >= 18.0.0
- npm >= 8.0.0
- Claude Code instalado

### Passo 1: Instalar Dependências

```bash
npm install sqlite3
```

### Passo 2: Inicializar Banco de Dados

```bash
node database/setup.js
```

Este script:
1. Cria o arquivo `roadmap.db`
2. Executa o script SQL `init_roadmap.sql`
3. Cria tabelas, views, triggers e índices
4. Insere dados iniciais de exemplo
5. Valida a criação

### Passo 3: Configurar MCP Server

O arquivo `.mcp.json` já está configurado:

```json
{
  "mcpServers": {
    "sqlite": {
      "type": "stdio",
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "-y",
        "mcp-server-sqlite-npx",
        "c:\\Users\\jeanz\\OneDrive\\Desktop\\Jizreel\\synth-bot-buddy-main\\database\\roadmap.db"
      ],
      "env": {}
    }
  }
}
```

### Passo 4: Reiniciar Claude Code

Para carregar o MCP server, reinicie o Claude Code.

### Passo 5: Verificar Instalação

No Claude Code, execute:

```
/mcp
```

Você deve ver o servidor `sqlite` com status **running** ou **connected**.

## 🤖 Uso com Claude Code

Após configurar o MCP server, você pode pedir ao Claude para interagir com o banco de dados diretamente.

### Exemplos de Comandos

#### Consultar progresso geral

```
Claude, mostre o progresso do roadmap por categoria
```

Claude irá executar:
```sql
SELECT * FROM roadmap_progress;
```

#### Adicionar nova task

```
Claude, adicione uma task "Implementar cache Redis" na categoria backend com prioridade high
```

Claude irá executar:
```sql
INSERT INTO roadmap_tasks (task_name, category, priority, status)
VALUES ('Implementar cache Redis', 'backend', 'high', 'todo');
```

#### Atualizar status

```
Claude, marque a task #5 como completed
```

Claude irá executar:
```sql
UPDATE roadmap_tasks SET status = 'completed' WHERE id = 5;
```

O trigger `set_completed_at` e `track_status_change` serão acionados automaticamente.

#### Ver tasks atrasadas

```
Claude, quais tasks estão atrasadas?
```

Claude irá executar:
```sql
SELECT * FROM overdue_tasks;
```

#### Ver histórico de uma task

```
Claude, mostre o histórico da task #3
```

Claude irá executar:
```sql
SELECT * FROM task_history WHERE task_id = 3 ORDER BY changed_at DESC;
```

#### Relatório de milestones

```
Claude, mostre todos os milestones e suas datas
```

Claude irá executar:
```sql
SELECT name, description, target_date, status FROM milestones ORDER BY target_date;
```

## 📊 Queries Úteis

### 1. Tasks por prioridade e status

```sql
SELECT
    priority,
    status,
    COUNT(*) as count
FROM roadmap_tasks
GROUP BY priority, status
ORDER BY
    CASE priority
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    status;
```

### 2. Tempo médio de conclusão por categoria

```sql
SELECT
    category,
    COUNT(*) as completed_tasks,
    ROUND(AVG(JULIANDAY(completed_at) - JULIANDAY(started_at)), 2) as avg_days_to_complete
FROM roadmap_tasks
WHERE status = 'completed'
  AND started_at IS NOT NULL
  AND completed_at IS NOT NULL
GROUP BY category;
```

### 3. Tasks bloqueadas por dependências

```sql
SELECT
    t1.id,
    t1.task_name,
    t1.dependencies,
    GROUP_CONCAT(t2.task_name || ' (' || t2.status || ')', ', ') as dependency_status
FROM roadmap_tasks t1
LEFT JOIN roadmap_tasks t2
    ON (',' || t1.dependencies || ',') LIKE ('%,' || t2.id || ',%')
WHERE t1.dependencies IS NOT NULL
  AND t1.status = 'todo'
GROUP BY t1.id, t1.task_name, t1.dependencies;
```

### 4. Burndown de horas estimadas vs reais

```sql
SELECT
    category,
    SUM(estimated_hours) as total_estimated,
    SUM(CASE WHEN status = 'completed' THEN actual_hours ELSE 0 END) as total_actual,
    SUM(CASE WHEN status = 'completed' THEN estimated_hours ELSE 0 END) as completed_estimated,
    ROUND(
        100.0 * SUM(CASE WHEN status = 'completed' THEN estimated_hours ELSE 0 END) / SUM(estimated_hours),
        2
    ) as percentage_complete
FROM roadmap_tasks
WHERE estimated_hours IS NOT NULL
GROUP BY category;
```

### 5. Timeline de mudanças (últimas 30 mudanças)

```sql
SELECT
    h.changed_at,
    t.task_name,
    h.status_from,
    h.status_to,
    h.notes
FROM task_history h
JOIN roadmap_tasks t ON h.task_id = t.id
ORDER BY h.changed_at DESC
LIMIT 30;
```

### 6. Tasks críticas sem data limite

```sql
SELECT
    id,
    task_name,
    category,
    priority,
    status
FROM roadmap_tasks
WHERE priority = 'critical'
  AND due_date IS NULL
  AND status NOT IN ('completed', 'cancelled');
```

## 🔧 Manutenção

### Backup do Banco de Dados

```bash
# Windows
copy database\roadmap.db database\backups\roadmap_%date:~-4,4%%date:~-10,2%%date:~-7,2%.db

# Linux/Mac
cp database/roadmap.db "database/backups/roadmap_$(date +%Y%m%d).db"
```

### Compactar Banco de Dados

```sql
VACUUM;
```

### Verificar Integridade

```sql
PRAGMA integrity_check;
```

### Atualizar Estatísticas

```sql
ANALYZE;
```

### Limpar Histórico Antigo (> 90 dias)

```sql
DELETE FROM task_history
WHERE changed_at < DATE('now', '-90 days');
```

## 🐛 Troubleshooting

### Problema: MCP server não conecta

**Solução:**
1. Verifique se o caminho do banco no `.mcp.json` está correto
2. Certifique-se de que o arquivo `roadmap.db` existe
3. Reinicie o Claude Code
4. Execute `/mcp` para ver logs de erro

### Problema: Permissões negadas

**Solução (Windows):**
```bash
icacls database\roadmap.db /grant %USERNAME%:F
```

**Solução (Linux/Mac):**
```bash
chmod 666 database/roadmap.db
```

### Problema: Banco corrompido

**Solução:**
1. Restaure do backup
2. Se não houver backup, tente:
```sql
.mode insert
.output dump.sql
.dump
.exit
```

3. Recrie o banco:
```bash
del roadmap.db
node database/setup.js
sqlite3 roadmap.db < dump.sql
```

## 📚 Referências

### Documentação Oficial

- [SQLite Official Documentation](https://www.sqlite.org/docs.html)
- [MCP Server SQLite NPX - GitHub](https://github.com/johnnyoshika/mcp-server-sqlite-npx)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### Ferramentas Úteis

- [DB Browser for SQLite](https://sqlitebrowser.org/) - GUI para SQLite
- [SQLite Viewer VS Code Extension](https://marketplace.visualstudio.com/items?itemName=qwtel.sqlite-viewer)
- [MCP Inspector](https://www.npmjs.com/package/@modelcontextprotocol/inspector) - Debug MCP servers

### Tutoriais

- [SQLite MCP server for AI agents](https://playbooks.com/mcp/johnnyoshika-sqlite-npx)
- [Claude MCP Servers Documentation](https://www.claudemcp.com/servers/sqlite)

---

## 📝 Changelog

### v1.0.0 - 2024-12-14

- ✨ Criação inicial do banco de dados
- ✨ Implementação de 4 tabelas principais
- ✨ 3 views para consultas comuns
- ✨ 4 triggers para automação
- ✨ 6 índices para performance
- ✨ Script de setup automatizado
- ✨ Integração com Claude Code via MCP
- ✨ Dados iniciais baseados no roadmap do projeto

---

**Última atualização**: 14/12/2024
**Versão**: 1.0.0
**Autor**: Claude Code Assistant
