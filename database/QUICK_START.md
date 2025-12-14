# 🚀 Quick Start - SQLite Roadmap com Claude Code

Guia rápido para começar a usar o controle de roadmap via Claude Code.

## ✅ Setup Completo

Tudo já está configurado! Você só precisa:

1. **Reiniciar o Claude Code** para carregar os MCP servers
2. Verificar se está funcionando: `/mcp`
3. Começar a usar!

## 🎯 Comandos Rápidos para Claude

### Ver Progresso Geral

```
Claude, mostre o progresso do roadmap
```

ou

```
Claude, SELECT * FROM roadmap_progress;
```

**Retorna**: Estatísticas por categoria (frontend, backend, ml, etc.)

### Ver Todas as Tasks

```
Claude, liste todas as tasks do roadmap
```

ou

```
Claude, SELECT id, task_name, status, priority, category FROM roadmap_tasks ORDER BY priority;
```

### Adicionar Nova Task

```
Claude, adicione uma task:
- Nome: Implementar autenticação OAuth
- Categoria: backend
- Prioridade: high
- Status: todo
```

Claude executará:
```sql
INSERT INTO roadmap_tasks (task_name, category, priority, status)
VALUES ('Implementar autenticação OAuth', 'backend', 'high', 'todo');
```

### Atualizar Status

```
Claude, marque a task #5 como in_progress
```

ou

```
Claude, a task "Configurar variáveis de ambiente" foi completada
```

### Ver Tasks Atrasadas

```
Claude, quais tasks estão atrasadas?
```

ou

```
Claude, SELECT * FROM overdue_tasks;
```

### Ver Próximas Tasks por Prioridade

```
Claude, o que devo fazer em seguida?
```

ou

```
Claude, SELECT * FROM next_tasks LIMIT 5;
```

### Ver Histórico de uma Task

```
Claude, mostre o histórico da task #3
```

ou

```
Claude, SELECT * FROM task_history WHERE task_id = 3 ORDER BY changed_at DESC;
```

### Ver Milestones

```
Claude, mostre os milestones do projeto
```

ou

```
Claude, SELECT * FROM milestones ORDER BY target_date;
```

### Relatório de Horas

```
Claude, quanto tempo estamos gastando vs estimado por categoria?
```

Claude executará:
```sql
SELECT
    category,
    SUM(estimated_hours) as estimado,
    SUM(actual_hours) as real,
    COUNT(*) as tasks
FROM roadmap_tasks
GROUP BY category;
```

## 📊 Consultas Úteis Prontas

### 1. Dashboard Executivo

```
Claude, gere um dashboard executivo do roadmap
```

Sugestão para Claude:
```sql
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completadas,
    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as em_progresso,
    SUM(CASE WHEN status = 'todo' THEN 1 ELSE 0 END) as a_fazer,
    SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) as bloqueadas,
    ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 1) as percentual
FROM roadmap_tasks;
```

### 2. Tasks Críticas

```
Claude, mostre todas as tasks críticas pendentes
```

```sql
SELECT id, task_name, category, due_date
FROM roadmap_tasks
WHERE priority = 'critical'
  AND status NOT IN ('completed', 'cancelled')
ORDER BY due_date;
```

### 3. Burndown de Categorias

```
Claude, mostre o burndown por categoria
```

```sql
SELECT * FROM roadmap_progress ORDER BY completion_percentage DESC;
```

## 🔧 Comandos de Manutenção

### Adicionar Horas Reais a uma Task

```
Claude, a task #7 levou 15 horas para completar
```

```sql
UPDATE roadmap_tasks
SET actual_hours = 15
WHERE id = 7;
```

### Definir Data Limite

```
Claude, defina a data limite da task #4 para 31/01/2025
```

```sql
UPDATE roadmap_tasks
SET due_date = '2025-01-31'
WHERE id = 4;
```

### Adicionar Notas

```
Claude, adicione uma nota na task #6: "Aguardando aprovação do cliente"
```

```sql
UPDATE roadmap_tasks
SET notes = 'Aguardando aprovação do cliente'
WHERE id = 6;
```

### Marcar Task como Bloqueada

```
Claude, marque a task #8 como bloqueada
```

```sql
UPDATE roadmap_tasks
SET status = 'blocked'
WHERE id = 8;
```

## 💡 Dicas Profissionais

### 1. Falar Naturalmente com Claude

Você **não precisa** escrever SQL manualmente! Claude entende linguagem natural:

❌ Ruim:
```
UPDATE roadmap_tasks SET status = 'completed' WHERE id = 5;
```

✅ Bom:
```
Claude, completei a task de WebSocket
```

### 2. Pedir Análises Complexas

```
Claude, quais categorias estão mais atrasadas no roadmap?
```

```
Claude, calcule o tempo médio de conclusão das tasks de backend
```

```
Claude, identifique tasks bloqueadas por dependências
```

### 3. Gerar Relatórios

```
Claude, gere um relatório semanal do progresso do roadmap
```

```
Claude, compare as horas estimadas vs reais de dezembro
```

### 4. Planejar Sprints

```
Claude, sugira 5 tasks para trabalhar essa semana baseado em prioridade
```

```
Claude, quais tasks posso fazer que não têm dependências bloqueadas?
```

## 🎨 Exemplos de Workflows

### Workflow 1: Começar o Dia

```
1. Claude, mostre as tasks em progresso
2. Claude, qual é a próxima task de maior prioridade?
3. Claude, marque a task #X como in_progress
```

### Workflow 2: Finalizar uma Task

```
1. Claude, marque a task "Deploy em produção" como completed
2. Claude, adicione 8 horas reais à task #4
3. Claude, mostre o progresso atualizado da categoria infrastructure
```

### Workflow 3: Planejamento Semanal

```
1. Claude, mostre todas as tasks atrasadas
2. Claude, liste tasks críticas para essa semana
3. Claude, identifique dependências bloqueadas
4. Claude, sugira ordem de execução das próximas 10 tasks
```

### Workflow 4: Review de Sprint

```
1. Claude, quantas tasks foram completadas nos últimos 7 dias?
2. Claude, mostre o histórico de mudanças da última semana
3. Claude, compare horas estimadas vs reais do sprint
4. Claude, atualize o status dos milestones
```

## 🔍 Troubleshooting Rápido

### MCP server não responde

```bash
# 1. Verificar se está rodando
/mcp

# 2. Se não aparecer, reiniciar Claude Code
# Ctrl+R ou fechar e abrir

# 3. Verificar se o banco existe
ls database/roadmap.db

# 4. Recriar se necessário
node database/setup.js
```

### Erro de sintaxe SQL

Sempre deixe o Claude construir a query! Exemplo:

❌ Não faça:
```
Claude, executa: SELET * FORM tasks
```

✅ Faça:
```
Claude, mostre todas as tasks
```

### Dados iniciais desapareceram

```bash
# Reinicializar banco
node database/setup.js
```

Isso recria tudo automaticamente.

## 📚 Atalhos de Comandos

| O que você quer | Comando rápido |
|----------------|----------------|
| Ver progresso | `Claude, progresso do roadmap` |
| Próxima task | `Claude, o que fazer agora?` |
| Adicionar task | `Claude, adicione task: [nome]` |
| Completar task | `Claude, completei a task [nome/id]` |
| Tasks atrasadas | `Claude, tarefas atrasadas` |
| Tasks críticas | `Claude, tarefas críticas` |
| Histórico | `Claude, histórico da task #X` |
| Milestones | `Claude, mostre milestones` |
| Horas | `Claude, relatório de horas` |
| Bloqueadas | `Claude, tasks bloqueadas` |

## 🎁 Bônus: Comandos Avançados

### Criar Dependências entre Tasks

```
Claude, a task #10 depende da task #4 e #5
```

```sql
UPDATE roadmap_tasks
SET dependencies = '4,5'
WHERE id = 10;
```

### Ver Tasks com Dependências Não Resolvidas

```
Claude, quais tasks estão bloqueadas por dependências?
```

### Associar Task a Milestone

```
Claude, associe a task #7 ao milestone "MVP Bot Deriv"
```

```sql
INSERT INTO task_milestones (task_id, milestone_id)
SELECT 7, id FROM milestones WHERE name = 'MVP Bot Deriv';
```

### Calcular Velocity

```
Claude, calcule quantas tasks por semana estamos completando no último mês
```

---

## 🚀 Pronto para Usar!

Agora você tem controle total do roadmap via Claude Code! Basta falar naturalmente com o Claude e ele gerenciará o banco de dados para você.

**Próximos passos:**
1. Reinicie o Claude Code
2. Execute `/mcp` para verificar
3. Comece a gerenciar seu roadmap! 🎉

**Lembre-se**: Claude entende linguagem natural, então não precisa decorar SQL. Apenas descreva o que você quer!
