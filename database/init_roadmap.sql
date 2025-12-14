-- Script de inicialização do banco de dados de controle de roadmap
-- Criado em: 2024-12-14

-- Tabela principal de roadmap
CREATE TABLE IF NOT EXISTS roadmap_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL CHECK(status IN ('todo', 'in_progress', 'completed', 'blocked', 'cancelled')) DEFAULT 'todo',
    priority TEXT CHECK(priority IN ('low', 'medium', 'high', 'critical')) DEFAULT 'medium',
    category TEXT CHECK(category IN ('frontend', 'backend', 'ml', 'infrastructure', 'documentation', 'testing')) NOT NULL,
    estimated_hours REAL,
    actual_hours REAL,
    assigned_to TEXT,
    dependencies TEXT, -- IDs de outras tasks separadas por vírgula
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    due_date DATE,
    notes TEXT
);

-- Tabela de progresso histórico
CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    status_from TEXT NOT NULL,
    status_to TEXT NOT NULL,
    changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    changed_by TEXT,
    notes TEXT,
    FOREIGN KEY (task_id) REFERENCES roadmap_tasks(id) ON DELETE CASCADE
);

-- Tabela de milestones
CREATE TABLE IF NOT EXISTS milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    target_date DATE,
    status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'missed')) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);

-- Tabela de relação entre tasks e milestones
CREATE TABLE IF NOT EXISTS task_milestones (
    task_id INTEGER NOT NULL,
    milestone_id INTEGER NOT NULL,
    PRIMARY KEY (task_id, milestone_id),
    FOREIGN KEY (task_id) REFERENCES roadmap_tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (milestone_id) REFERENCES milestones(id) ON DELETE CASCADE
);

-- Índices para melhorar performance
CREATE INDEX IF NOT EXISTS idx_roadmap_status ON roadmap_tasks(status);
CREATE INDEX IF NOT EXISTS idx_roadmap_priority ON roadmap_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_roadmap_category ON roadmap_tasks(category);
CREATE INDEX IF NOT EXISTS idx_roadmap_due_date ON roadmap_tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_task_history_task_id ON task_history(task_id);
CREATE INDEX IF NOT EXISTS idx_task_history_changed_at ON task_history(changed_at);

-- Trigger para atualizar updated_at automaticamente
CREATE TRIGGER IF NOT EXISTS update_roadmap_timestamp
AFTER UPDATE ON roadmap_tasks
FOR EACH ROW
BEGIN
    UPDATE roadmap_tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- Trigger para registrar mudanças de status no histórico
CREATE TRIGGER IF NOT EXISTS track_status_change
AFTER UPDATE OF status ON roadmap_tasks
FOR EACH ROW
WHEN OLD.status != NEW.status
BEGIN
    INSERT INTO task_history (task_id, status_from, status_to, notes)
    VALUES (OLD.id, OLD.status, NEW.status, 'Status changed automatically');
END;

-- Trigger para atualizar started_at quando status muda para in_progress
CREATE TRIGGER IF NOT EXISTS set_started_at
AFTER UPDATE OF status ON roadmap_tasks
FOR EACH ROW
WHEN NEW.status = 'in_progress' AND OLD.status != 'in_progress' AND OLD.started_at IS NULL
BEGIN
    UPDATE roadmap_tasks SET started_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- Trigger para atualizar completed_at quando status muda para completed
CREATE TRIGGER IF NOT EXISTS set_completed_at
AFTER UPDATE OF status ON roadmap_tasks
FOR EACH ROW
WHEN NEW.status = 'completed' AND OLD.status != 'completed'
BEGIN
    UPDATE roadmap_tasks SET completed_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

-- Inserir dados iniciais baseados no roadmap do projeto
INSERT INTO milestones (name, description, target_date, status) VALUES
    ('MVP Bot Deriv', 'Versão inicial funcional do bot de trading', '2025-01-31', 'in_progress'),
    ('ML Model v1', 'Primeira versão do modelo de ML em produção', '2025-02-15', 'pending'),
    ('Dashboard v1', 'Interface web para monitoramento', '2025-02-28', 'pending');

-- Inserir tasks iniciais do roadmap
INSERT INTO roadmap_tasks (task_name, description, status, priority, category, estimated_hours, dependencies) VALUES
    ('Configurar MCP Servers', 'Instalar e configurar sequential-thinking e sqlite-mcp', 'completed', 'high', 'infrastructure', 2, NULL),
    ('Testar endpoints ML em produção', 'Validar endpoints /api/ml/info e /api/ml/predict em servidor real', 'completed', 'critical', 'ml', 4, NULL),
    ('Configurar variáveis de ambiente', 'Setup de token Deriv API e outras env vars', 'todo', 'critical', 'infrastructure', 2, NULL),
    ('Deploy em Easypanel/Railway', 'Deploy do backend em plataforma de produção', 'todo', 'critical', 'infrastructure', 8, '3'),
    ('Implementar WebSocket dashboard', 'Endpoint /ws/dashboard para atualizações em tempo real', 'completed', 'high', 'backend', 6, NULL),
    ('Modo observação (1 semana)', 'Monitorar bot sem executar trades reais', 'todo', 'high', 'testing', 40, '4'),
    ('Trading com capital pequeno', 'Iniciar trading real com $100', 'todo', 'critical', 'backend', 20, '6'),
    ('Monitoramento e alertas', 'Sistema de monitoramento e notificações', 'todo', 'high', 'infrastructure', 12, '4'),
    ('Documentação API completa', 'Documentar todos os endpoints e fluxos', 'in_progress', 'medium', 'documentation', 8, NULL),
    ('Testes automatizados ML', 'Suite de testes para modelo ML', 'todo', 'medium', 'testing', 16, '2');

-- View para dashboard de progresso
CREATE VIEW IF NOT EXISTS roadmap_progress AS
SELECT
    category,
    COUNT(*) as total_tasks,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_tasks,
    SUM(CASE WHEN status = 'todo' THEN 1 ELSE 0 END) as todo_tasks,
    SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) as blocked_tasks,
    ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 2) as completion_percentage,
    SUM(estimated_hours) as total_estimated_hours,
    SUM(actual_hours) as total_actual_hours
FROM roadmap_tasks
GROUP BY category;

-- View para tasks atrasadas
CREATE VIEW IF NOT EXISTS overdue_tasks AS
SELECT
    id,
    task_name,
    description,
    status,
    priority,
    category,
    due_date,
    JULIANDAY('now') - JULIANDAY(due_date) as days_overdue
FROM roadmap_tasks
WHERE status NOT IN ('completed', 'cancelled')
  AND due_date IS NOT NULL
  AND due_date < DATE('now')
ORDER BY days_overdue DESC;

-- View para próximas tasks por prioridade
CREATE VIEW IF NOT EXISTS next_tasks AS
SELECT
    id,
    task_name,
    description,
    priority,
    category,
    due_date,
    estimated_hours
FROM roadmap_tasks
WHERE status = 'todo'
ORDER BY
    CASE priority
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    due_date ASC NULLS LAST;
