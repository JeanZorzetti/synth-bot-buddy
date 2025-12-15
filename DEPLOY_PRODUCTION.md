# 🚀 Guia de Deploy em Produção - Trading Bot

## 📋 Índice
1. [Pré-requisitos](#pré-requisitos)
2. [Configuração Inicial](#configuração-inicial)
3. [Deploy com Docker Compose](#deploy-com-docker-compose)
4. [Configuração de Monitoramento](#configuração-de-monitoramento)
5. [Sistema de Alertas](#sistema-de-alertas)
6. [Backup e Recuperação](#backup-e-recuperação)
7. [Troubleshooting](#troubleshooting)
8. [Checklist de Deploy](#checklist-de-deploy)

---

## ✅ Pré-requisitos

### Hardware Mínimo Recomendado
- **CPU**: 4 cores (2.0 GHz+)
- **RAM**: 8 GB
- **Disco**: 50 GB SSD
- **Rede**: 100 Mbps (baixa latência < 50ms)

### Software Necessário
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.30+
- **Certbot** (opcional, para SSL)

### Verificar Instalação
```bash
docker --version
docker-compose --version
git --version
```

---

## 🔧 Configuração Inicial

### 1. Clone o Repositório
```bash
git clone https://github.com/JeanZorzetti/synth-bot-buddy.git
cd synth-bot-buddy
```

### 2. Configure as Variáveis de Ambiente
```bash
cp .env.production.example .env.production
nano .env.production
```

**Variáveis OBRIGATÓRIAS para preencher:**
```bash
# Deriv API
DERIV_API_TOKEN=your_token_here
DERIV_APP_ID=your_app_id

# Database
DB_PASSWORD=strong_password_here

# Redis
REDIS_PASSWORD=strong_password_here

# Security
JWT_SECRET_KEY=min_32_character_secret_key

# Grafana
GRAFANA_ADMIN_PASSWORD=strong_password_here

# Telegram Alerts
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=-1001234567

# Email Alerts
ALERT_EMAIL_USERNAME=your_email@gmail.com
ALERT_EMAIL_PASSWORD=app_specific_password
ALERT_EMAIL_TO=alerts@yourcompany.com
```

### 3. Criar Bot do Telegram (para alertas)
1. Acesse [@BotFather](https://t.me/BotFather)
2. Digite `/newbot`
3. Escolha um nome e username
4. Copie o **token** para `TELEGRAM_BOT_TOKEN`
5. Acesse [@userinfobot](https://t.me/userinfobot)
6. Digite `/start` e copie seu **chat_id** para `TELEGRAM_CHAT_ID`

### 4. Configurar App Password do Gmail (para email alerts)
1. Acesse [Google Account Security](https://myaccount.google.com/security)
2. Ative **Verificação em 2 etapas**
3. Gere um **App Password** em "App passwords"
4. Use esse password em `ALERT_EMAIL_PASSWORD`

---

## 🐳 Deploy com Docker Compose

### 1. Build das Imagens
```bash
docker-compose -f docker-compose.prod.yml build
```

### 2. Iniciar Todos os Serviços
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Verificar Status dos Containers
```bash
docker-compose -f docker-compose.prod.yml ps
```

**Saída esperada:**
```
NAME                      STATUS    PORTS
trading-bot-app-prod      Up        0.0.0.0:8000->8000/tcp
trading-bot-db-prod       Up        0.0.0.0:5432->5432/tcp
trading-bot-redis-prod    Up        0.0.0.0:6379->6379/tcp
trading-bot-prometheus    Up        0.0.0.0:9090->9090/tcp
trading-bot-grafana       Up        0.0.0.0:3000->3000/tcp
trading-bot-alertmanager  Up        0.0.0.0:9093->9093/tcp
```

### 4. Verificar Logs
```bash
# Logs do Trading Bot
docker logs -f trading-bot-app-prod

# Logs de todos os serviços
docker-compose -f docker-compose.prod.yml logs -f
```

---

## 📊 Configuração de Monitoramento

### 1. Acessar Grafana
```
URL: http://seu-servidor:3000
Usuário: admin
Senha: <GRAFANA_ADMIN_PASSWORD do .env.production>
```

### 2. Verificar Dashboards
Após login, vá em:
- **Dashboards → Browse → Trading Bot → Trading Bot - Main Dashboard**

Você verá:
- 📊 Win Rate em tempo real
- 🎯 P&L acumulado
- 🧠 ML Model Accuracy
- ⚠️ Max Drawdown
- 📈 Gráficos de performance
- 🚀 Posições ativas
- 🔥 Últimas trades

### 3. Acessar Prometheus
```
URL: http://seu-servidor:9090
```

Métricas disponíveis:
```promql
trading_bot_total_pnl
trading_bot_win_rate_pct
trading_bot_sharpe_ratio
ml_model_accuracy
trading_bot_active_positions
```

### 4. Acessar Alertmanager
```
URL: http://seu-servidor:9093
```

---

## 🚨 Sistema de Alertas

### Alertas Configurados

#### 🔴 CRÍTICOS (Email + Telegram + Webhook)
- API desconectada por 5+ minutos
- Loss diário > 5%
- Drawdown > 15%
- Erro de execução de ordem

#### 🟡 WARNING (Apenas Telegram)
- Win rate < 50% (últimas 20 trades)
- Latência > 500ms
- Model accuracy < 65%

### Testar Sistema de Alertas

```bash
# Forçar um alerta de teste
docker exec -it trading-bot-prometheus promtool check rules /etc/prometheus/rules/trading-alerts.yml

# Enviar alerta de teste para Telegram
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d "chat_id=${TELEGRAM_CHAT_ID}" \
  -d "text=🧪 Teste de alerta do Trading Bot"
```

---

## 💾 Backup e Recuperação

### Backup Automático Configurado
O sistema faz backup automático **diariamente às 2 AM** dos seguintes dados:
- PostgreSQL database
- Models ML (XGBoost)
- Logs
- Configurações

### Backup Manual
```bash
# Backup do banco de dados
docker exec trading-bot-db-prod pg_dump -U trading_user trading_bot_prod > backup_$(date +%Y%m%d).sql

# Backup dos modelos ML
docker exec trading-bot-app-prod tar -czf /backups/models_$(date +%Y%m%d).tar.gz /app/models

# Backup completo (database + models + logs)
./scripts/backup.sh
```

### Restaurar Backup
```bash
# Restaurar database
docker exec -i trading-bot-db-prod psql -U trading_user trading_bot_prod < backup_20241215.sql

# Restaurar models
docker exec -i trading-bot-app-prod tar -xzf /backups/models_20241215.tar.gz -C /app
```

---

## 🐛 Troubleshooting

### Container não inicia

**Sintoma:** `docker-compose up -d` falha ou container morre imediatamente

**Diagnóstico:**
```bash
# Ver logs de erro
docker-compose -f docker-compose.prod.yml logs trading-bot

# Verificar variáveis de ambiente
docker-compose -f docker-compose.prod.yml config
```

**Soluções comuns:**
1. Verificar se `.env.production` existe e está preenchido
2. Verificar se as portas 8000, 3000, 9090, 5432, 6379 estão livres
3. Verificar se há espaço em disco (`df -h`)

### Erro: ModuleNotFoundError

**Sintoma:** `ModuleNotFoundError: No module named 'XXX'`

**Solução:**
```bash
# Rebuild a imagem com dependências atualizadas
docker-compose -f docker-compose.prod.yml build --no-cache trading-bot
docker-compose -f docker-compose.prod.yml up -d
```

### Erro de conexão com Deriv API

**Sintoma:** `WebSocket connection failed` nos logs

**Diagnóstico:**
```bash
# Testar conectividade
docker exec -it trading-bot-app-prod ping ws.derivws.com

# Verificar token
docker exec -it trading-bot-app-prod env | grep DERIV
```

**Soluções:**
1. Verificar se `DERIV_API_TOKEN` está correto
2. Verificar se token não expirou (regenerar em app.deriv.com)
3. Verificar firewall/proxy bloqueando WSS

### Grafana não mostra dados

**Sintoma:** Dashboards vazios ou "No data"

**Diagnóstico:**
```bash
# Verificar se Prometheus está coletando métricas
curl http://localhost:9090/api/v1/query?query=up

# Verificar se Trading Bot está expondo métricas
curl http://localhost:8000/metrics
```

**Soluções:**
1. Verificar se Prometheus está rodando: `docker ps | grep prometheus`
2. Verificar configuração de datasource no Grafana (Settings → Data Sources)
3. Reiniciar Prometheus: `docker restart trading-bot-prometheus`

### Alertas não chegam no Telegram

**Sintoma:** Prometheus dispara alertas mas não recebe no Telegram

**Diagnóstico:**
```bash
# Testar envio manual
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d "chat_id=${TELEGRAM_CHAT_ID}" \
  -d "text=Teste"

# Verificar logs do Alertmanager
docker logs trading-bot-alertmanager
```

**Soluções:**
1. Verificar `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` corretos
2. Adicionar bot ao grupo (se chat_id for de grupo)
3. Verificar se Alertmanager está rodando: `docker ps | grep alertmanager`

### Alto uso de CPU/Memória

**Sintoma:** Sistema lento, container usando > 80% CPU/RAM

**Diagnóstico:**
```bash
# Ver uso de recursos
docker stats

# Ver processos dentro do container
docker exec -it trading-bot-app-prod top
```

**Soluções:**
1. Aumentar limites de recursos no `docker-compose.prod.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
```
2. Otimizar workers: reduzir `UVICORN_WORKERS` no `.env.production`
3. Ativar cache Redis para reduzir queries no DB

---

## ✅ Checklist de Deploy

### Pré-Deploy
- [ ] Clone do repositório atualizado
- [ ] `.env.production` criado e preenchido
- [ ] Variáveis obrigatórias configuradas (DERIV_API_TOKEN, DB_PASSWORD, etc.)
- [ ] Bot do Telegram criado e testado
- [ ] Email App Password configurado
- [ ] Firewall liberado para portas: 8000, 3000, 9090, 5432, 6379
- [ ] SSL/TLS configurado (se produção pública)

### Deploy
- [ ] Build das imagens sem erros: `docker-compose build`
- [ ] Todos containers iniciaram: `docker-compose ps` mostra "Up"
- [ ] Logs sem erros críticos: `docker logs trading-bot-app-prod`
- [ ] API responde: `curl http://localhost:8000/health` retorna 200
- [ ] Prometheus coletando métricas: `curl http://localhost:9090/api/v1/targets`
- [ ] Grafana acessível: `http://localhost:3000`
- [ ] Alertmanager acessível: `http://localhost:9093`

### Pós-Deploy
- [ ] Dashboards do Grafana carregando dados
- [ ] Alerta de teste enviado para Telegram
- [ ] Alerta de teste enviado para Email
- [ ] Forward Testing iniciado via UI
- [ ] Backup automático agendado (cron)
- [ ] Monitoramento 24/7 ativo
- [ ] Documentação de runbook criada

### Testes de Funcionalidade
- [ ] Login no frontend funciona
- [ ] Dashboard mostra dados em tempo real
- [ ] ML Predictor gerando previsões
- [ ] Paper Trading executando trades simulados
- [ ] Forward Testing coletando métricas
- [ ] Alertas disparando corretamente
- [ ] Logs sendo escritos em `/app/logs`

---

## 📞 Suporte

### Logs Importantes
```bash
# Trading Bot
docker logs -f --tail 100 trading-bot-app-prod

# Database
docker logs -f trading-bot-db-prod

# Prometheus
docker logs -f trading-bot-prometheus

# Grafana
docker logs -f trading-bot-grafana
```

### Reiniciar Serviços
```bash
# Reiniciar apenas Trading Bot
docker restart trading-bot-app-prod

# Reiniciar todos os serviços
docker-compose -f docker-compose.prod.yml restart

# Parar tudo
docker-compose -f docker-compose.prod.yml down

# Parar e remover volumes (⚠️ APAGA DADOS!)
docker-compose -f docker-compose.prod.yml down -v
```

### Atualizar para Nova Versão
```bash
# 1. Pull do código atualizado
git pull origin main

# 2. Rebuild
docker-compose -f docker-compose.prod.yml build

# 3. Restart (sem downtime)
docker-compose -f docker-compose.prod.yml up -d --force-recreate
```

---

## 🎯 Próximos Passos Após Deploy

1. **Monitorar por 24h** - Verificar se tudo está estável
2. **Validar Alertas** - Confirmar que alertas críticos funcionam
3. **Rodar Forward Testing** - Coletar 4 semanas de dados
4. **Ajustar Parâmetros** - Otimizar baseado em métricas reais
5. **Ativar Trading Real** - Apenas após validação completa

---

## 🔒 Segurança em Produção

### Recomendações CRÍTICAS
- ✅ **NUNCA** commite `.env.production` no Git
- ✅ Use senhas fortes (min 16 caracteres)
- ✅ Ative autenticação em 2 fatores para Deriv
- ✅ Configure firewall (permitir apenas IPs confiáveis)
- ✅ Use SSL/TLS para comunicação externa
- ✅ Rotacione secrets a cada 90 dias
- ✅ Mantenha backups em local seguro (fora do servidor)
- ✅ Monitore logs de acesso suspeito

---

**Deploy realizado com sucesso? Parabéns! 🎉**

Agora é só deixar o bot rodando e monitorar as métricas no Grafana.
