# 🚀 PRODUCTION DEPLOYMENT GUIDE
## AI Trading Bot - Complete Production Setup

---

## 📋 **PRÉ-REQUISITOS**

### **🔧 Ambiente de Produção**
- **Docker & Docker Compose** (v20.10+)
- **Linux Server** (Ubuntu 20.04+ ou CentOS 8+)
- **Mínimo 8GB RAM**, 4 CPU cores
- **50GB SSD** para dados e logs
- **Conexão estável** com internet

### **🔑 Credenciais Necessárias**
- **Deriv API Key** (conta real ou demo)
- **JWT Secret Key** (gerado automaticamente se não fornecido)
- **SSL Certificates** (Let's Encrypt ou custom)
- **Database Password** (PostgreSQL)
- **Redis Password** (cache)

---

## 🚀 **DEPLOY RÁPIDO**

### **1. Clone e Configure**
```bash
# Clone do repositório
git clone https://github.com/your-repo/ai-trading-bot.git
cd ai-trading-bot

# Copiar arquivo de ambiente
cp .env.example .env.production
```

### **2. Configurar Variáveis de Ambiente**
```bash
# Editar arquivo .env.production
nano .env.production
```

```env
# === CORE CONFIGURATION ===
ENVIRONMENT=production
DERIV_API_KEY=your_deriv_api_key_here
JWT_SECRET_KEY=your_super_secret_jwt_key_here

# === DATABASE ===
DB_PASSWORD=secure_database_password_here
POSTGRES_DB=trading_bot_prod
POSTGRES_USER=trading_user

# === CACHE ===
REDIS_PASSWORD=secure_redis_password_here

# === MONITORING ===
GRAFANA_ADMIN_PASSWORD=secure_grafana_password
HEALTHCHECK_WEBHOOK_URL=https://your-webhook-url.com/alerts

# === SSL (Opcional) ===
SSL_CERT_FILE=/certs/server.crt
SSL_KEY_FILE=/certs/server.key

# === ALERTS ===
SLACK_WEBHOOK_URL=https://hooks.slack.com/your-webhook
ERROR_WEBHOOK_URL=https://your-error-alerts-webhook.com
```

### **3. Deploy Completo**
```bash
# Fazer deploy de produção
docker-compose -f docker-compose.prod.yml up -d

# Verificar status
docker-compose -f docker-compose.prod.yml ps
```

### **4. Verificar Health Checks**
```bash
# Health check principal
curl http://localhost:8000/health

# Health check detalhado
curl http://localhost:8000/health/detailed

# Logs em tempo real
docker-compose -f docker-compose.prod.yml logs -f trading-bot
```

---

## 🔧 **CONFIGURAÇÃO AVANÇADA**

### **🔐 SSL/HTTPS Setup**

#### **Opção 1: Let's Encrypt (Recomendado)**
```bash
# Instalar Certbot
sudo apt update
sudo apt install certbot

# Gerar certificados
sudo certbot certonly --standalone -d your-domain.com

# Copiar certificados
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./certs/server.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./certs/server.key
sudo chown $USER:$USER ./certs/*
```

#### **Opção 2: Self-Signed (Desenvolvimento)**
```bash
# Criar certificados auto-assinados
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes
```

### **🗄️ Database Initialization**
```bash
# Executar migrações
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/migrations/run_migrations.py

# Seed dados iniciais (opcional)
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/scripts/seed_data.py
```

### **📊 Monitoring Setup**

#### **Acessar Dashboards**
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

#### **Configurar Alertas**
```bash
# Editar regras de alerta
nano monitoring/prometheus/alert_rules.yml

# Reiniciar Prometheus
docker-compose -f docker-compose.prod.yml restart prometheus
```

---

## 📈 **MONITORAMENTO E ALERTAS**

### **🚨 Alertas Críticos**
O sistema monitora automaticamente:

- **🔴 CPU > 80%** - Alta utilização de CPU
- **🔴 Memory > 90%** - Uso excessivo de memória
- **🔴 API Errors > 5%** - Taxa de erro da API Deriv
- **🔴 Database Down** - Conexão perdida com BD
- **🔴 Trading Stopped** - Bot parou de operar
- **🔴 Model Accuracy < 60%** - Performance da IA degradou

### **📊 Métricas Principais**
```bash
# Ver métricas via API
curl http://localhost:8000/metrics

# Métricas de trading
curl http://localhost:8000/api/v1/trading/metrics

# Status da IA
curl http://localhost:8000/api/v1/ai/status
```

### **🔍 Log Analysis**
```bash
# Ver logs estruturados
docker-compose -f docker-compose.prod.yml logs trading-bot | grep ERROR

# Logs de trading
docker-compose -f docker-compose.prod.yml logs trading-bot | grep "TRADE_"

# Logs de IA
docker-compose -f docker-compose.prod.yml logs trading-bot | grep "AI_"
```

---

## 💾 **BACKUP E RECOVERY**

### **🔄 Backup Automático**
O sistema executa backups automáticos:

- **Database**: Diário às 2:00 AM
- **Models**: Backup após cada retreino
- **Config**: Backup a cada mudança
- **Logs**: Rotação semanal

### **📦 Backup Manual**
```bash
# Backup completo
docker-compose -f docker-compose.prod.yml exec backup /scripts/full_backup.sh

# Backup apenas database
docker-compose -f docker-compose.prod.yml exec backup /scripts/db_backup.sh

# Listar backups
ls -la backups/
```

### **⚡ Recovery**
```bash
# Restaurar database
docker-compose -f docker-compose.prod.yml exec backup /scripts/restore_db.sh backup_file.sql

# Restaurar modelo
cp backups/models/model_backup.h5 models/lstm_trading_model.h5

# Reiniciar sistema
docker-compose -f docker-compose.prod.yml restart trading-bot
```

---

## 🔧 **TROUBLESHOOTING**

### **❌ Problemas Comuns**

#### **1. Trading Bot não conecta à Deriv API**
```bash
# Verificar API key
docker-compose -f docker-compose.prod.yml exec trading-bot python -c "
import os
print('API Key:', os.getenv('DERIV_API_KEY', 'NOT_SET')[:10] + '...')
"

# Testar conexão manual
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/scripts/test_deriv_connection.py
```

#### **2. Database Connection Failed**
```bash
# Verificar status do PostgreSQL
docker-compose -f docker-compose.prod.yml exec database pg_isready -U trading_user

# Ver logs do database
docker-compose -f docker-compose.prod.yml logs database

# Resetar database (CUIDADO!)
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d database
```

#### **3. High Memory Usage**
```bash
# Verificar uso de memória
docker stats

# Otimizar configuração
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/performance_optimizer.py

# Reiniciar com mais memória
docker-compose -f docker-compose.prod.yml down
# Editar docker-compose.prod.yml: adicionar memory limits
docker-compose -f docker-compose.prod.yml up -d
```

#### **4. AI Model Not Loading**
```bash
# Verificar modelos
docker-compose -f docker-compose.prod.yml exec trading-bot ls -la models/

# Recarregar modelo
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/scripts/reload_model.py

# Usar modelo de backup
docker-compose -f docker-compose.prod.yml exec trading-bot cp models/backup/lstm_trading_model_backup.h5 models/lstm_trading_model.h5
```

### **🔍 Debug Mode**
```bash
# Ativar debug detalhado
export DEBUG=1
docker-compose -f docker-compose.prod.yml restart trading-bot

# Executar testes integrados
docker-compose -f docker-compose.prod.yml exec trading-bot python -m pytest tests/ -v

# Profiling de performance
docker-compose -f docker-compose.prod.yml exec trading-bot python backend/scripts/performance_profile.py
```

---

## 🔄 **UPDATE E MAINTENANCE**

### **📥 Deploy de Nova Versão**
```bash
# Pull da nova versão
git pull origin main

# Rebuild apenas se houver mudanças no código
docker-compose -f docker-compose.prod.yml build trading-bot

# Deploy com zero downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps trading-bot

# Verificar health
curl http://localhost:8000/health
```

### **🧹 Maintenance Tasks**
```bash
# Limpeza de logs antigos
docker-compose -f docker-compose.prod.yml exec trading-bot find logs/ -name "*.log" -mtime +7 -delete

# Otimização de database
docker-compose -f docker-compose.prod.yml exec database psql -U trading_user -d trading_bot_prod -c "VACUUM ANALYZE;"

# Limpeza de cache
docker-compose -f docker-compose.prod.yml exec redis redis-cli FLUSHDB

# Update de dependências
docker-compose -f docker-compose.prod.yml exec trading-bot pip install --upgrade -r requirements-prod.txt
```

---

## 📊 **PERFORMANCE TUNING**

### **⚡ Otimizações Recomendadas**

#### **1. Sistema Operacional**
```bash
# Aumentar limites de arquivo
echo "trading-bot soft nofile 65536" >> /etc/security/limits.conf
echo "trading-bot hard nofile 65536" >> /etc/security/limits.conf

# Otimizar TCP
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 1024" >> /etc/sysctl.conf
sysctl -p
```

#### **2. Docker Optimization**
```yaml
# Adicionar ao docker-compose.prod.yml
services:
  trading-bot:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
```

#### **3. Database Tuning**
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

---

## 📞 **SUPORTE E CONTATO**

### **🆘 Em Caso de Emergência**
1. **Parar trading imediatamente**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/trading/emergency-stop
   ```

2. **Verificar logs críticos**:
   ```bash
   docker-compose -f docker-compose.prod.yml logs trading-bot | grep -E "(ERROR|CRITICAL|EMERGENCY)"
   ```

3. **Backup de emergência**:
   ```bash
   docker-compose -f docker-compose.prod.yml exec backup /scripts/emergency_backup.sh
   ```

### **📋 Checklist de Produção**

- [ ] ✅ Variáveis de ambiente configuradas
- [ ] ✅ SSL certificates instalados
- [ ] ✅ Database inicializado e migrações executadas
- [ ] ✅ Redis cache configurado
- [ ] ✅ Monitoring dashboards acessíveis
- [ ] ✅ Alertas configurados e testados
- [ ] ✅ Backups automáticos funcionando
- [ ] ✅ Health checks passando
- [ ] ✅ API Deriv conectada e testada
- [ ] ✅ Modelo de IA carregado e validado
- [ ] ✅ Trading execution testado (modo paper)
- [ ] ✅ Performance benchmarks executados
- [ ] ✅ Security hardening aplicado
- [ ] ✅ Log aggregation configurado
- [ ] ✅ Disaster recovery testado

---

**🎉 Parabéns! Seu AI Trading Bot está pronto para produção!**

*Sistema desenvolvido com foco em segurança, performance e confiabilidade para trading autônomo 24/7.*