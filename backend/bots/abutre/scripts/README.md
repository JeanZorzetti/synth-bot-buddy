# ABUTRE BOT - Scripts Utilitários

Scripts de manutenção e monitoramento do bot Abutre.

---

## 📊 get_daily_metrics.py

**Objetivo:** Extrair métricas diárias do banco de dados para atualizar o `FORWARD_TEST_LOG.md`.

### Uso

```bash
cd backend/bots/abutre
python scripts/get_daily_metrics.py
```

### Output

O script exibe no console:

1. **Métricas do dia:**
   - Saldo atual
   - ROI total
   - Win Rate
   - Max Drawdown
   - Total de trades (geral e do dia)

2. **Eventos de hoje:**
   - Trades executados
   - Gatilhos detectados
   - Horário de cada evento

3. **Últimos 5 trades:**
   - Timestamp
   - Direção (CALL/PUT)
   - Nível Martingale
   - Stake
   - P&L
   - Status

4. **Entrada formatada para o log:**
   - Pronta para copiar/colar no `FORWARD_TEST_LOG.md`

### Exemplo de Saída

```
============================================================
📊 MÉTRICAS DIÁRIAS - 22/12/2025
============================================================

💰 Saldo: $2,012.35
📈 ROI: +0.62%
📊 Win Rate: 100.0%
📉 Max Drawdown: 0.50%
🔢 Total Trades: 3 (hoje: 3)

============================================================
📅 EVENTOS DE HOJE
============================================================

[14:23:15] Trigger detected: 8 velas consecutivas → CALL
[14:23:20] Trade CALL L1: WIN (+$0.95)
[16:45:30] Trigger detected: 9 velas consecutivas → PUT
[16:45:35] Trade PUT L1: WIN (+$0.95)
[18:10:05] Trade CALL L1: WIN (+$0.95)

============================================================
📜 ÚLTIMOS 5 TRADES
============================================================

Time                 | Dir   | Lv  | Stake    | P&L        | Status
----------------------------------------------------------------------
22/12 18:10:05       | CALL  | L1  | $1.00    | +$0.95     | WIN
22/12 16:45:35       | PUT   | L1  | $1.00    | +$0.95     | WIN
22/12 14:23:20       | CALL  | L1  | $1.00    | +$0.95     | WIN

============================================================

📝 COPIE ESTA ENTRADA PARA O FORWARD_TEST_LOG.md:

------------------------------------------------------------
#### 📆 Dia 1 - 22/12/2025

**Métricas:**
- Saldo: $2,012.35
- Total Trades: 3 (hoje: 3)
- Win Rate: 100.0%
- Max Drawdown: 0.50%
- ROI: +0.62%

**Eventos:**
- [14:23:15] Trigger detected: 8 velas consecutivas → CALL
- [14:23:20] Trade CALL L1: WIN (+$0.95)
- [16:45:30] Trigger detected: 9 velas consecutivas → PUT
- [16:45:35] Trade PUT L1: WIN (+$0.95)
- [18:10:05] Trade CALL L1: WIN (+$0.95)

**Observações:**
- ✅ Win rate perfeito mantido
- ✅ Drawdown sob controle
- ✅ ROI positivo

------------------------------------------------------------
```

### Quando Usar

**Recomendado:** Executar **diariamente** ao final do dia para:

1. Verificar progresso do Forward Test
2. Atualizar o `FORWARD_TEST_LOG.md` com dados reais
3. Identificar anomalias (drawdown alto, win rate baixo, etc.)
4. Registrar eventos importantes

### Requisitos

- Banco de dados: `backend/bots/abutre/data/abutre.db` deve existir
- Bot deve ter executado pelo menos uma vez
- Python 3.8+

### Troubleshooting

**Erro: "Database not found"**
- Causa: Bot ainda não criou o banco de dados
- Solução: Rodar o bot pelo menos uma vez (`python main.py --demo`)

**Nenhum trade mostrado**
- Normal se bot não detectou gatilho ainda
- Aguardar 8+ velas consecutivas para primeiro sinal

**Métricas zeradas**
- Bot em paper trading sem trades executados
- Aguardar atividade do mercado

---

## 📁 Estrutura de Arquivos

```
backend/bots/abutre/
├── scripts/
│   ├── README.md                 ← Este arquivo
│   └── get_daily_metrics.py     ← Extrator de métricas
├── data/
│   └── abutre.db                ← Banco de dados SQLite
├── logs/
│   └── abutre.log               ← Logs do bot
├── FORWARD_TEST_LOG.md          ← Log de acompanhamento (30 dias)
├── SETUP_DERIV_API.md           ← Guia de configuração
├── .env                         ← Variáveis de ambiente
└── main.py                      ← Bot principal
```

---

## 🔄 Workflow Recomendado

### Diário (5 minutos)

1. Acessar dashboard: https://botderiv.rollabs.com.br/abutre
2. Verificar status: Connected ✅, Paper Trading ON
3. Executar script de métricas:
   ```bash
   python scripts/get_daily_metrics.py
   ```
4. Copiar saída e atualizar `FORWARD_TEST_LOG.md`
5. Commitar mudanças:
   ```bash
   git add backend/bots/abutre/FORWARD_TEST_LOG.md
   git commit -m "docs: Forward Test Day X - atualização métricas"
   ```

### Semanal (15 minutos)

1. Revisar todos os dias da semana no log
2. Calcular médias semanais:
   - ROI médio
   - Win rate médio
   - Trades por dia
   - Max drawdown da semana
3. Preencher "Resumo Semanal" no `FORWARD_TEST_LOG.md`
4. Comparar com backtest:
   - ROI semanal vs esperado (+1.5% por semana)
   - Win rate vs backtest (100%)

### Ao Final dos 30 Dias

1. Executar análise final:
   ```bash
   python scripts/get_daily_metrics.py
   ```
2. Preencher seção "Análise Final" do `FORWARD_TEST_LOG.md`
3. Comparar métricas com critérios de aprovação:
   - ✅ ROI > 5% → APROVADO
   - ✅ Win Rate > 90% → APROVADO
   - ✅ Max DD < 30% → APROVADO
4. Decisão:
   - Se APROVADO → Avançar para FASE 3.2 (Paper Trading Real)
   - Se REPROVADO → Aumentar `DELAY_THRESHOLD=10` e repetir

---

## 🆘 Suporte

**Logs:**
```bash
# Ver logs em tempo real
tail -f logs/abutre.log

# Buscar por erros
grep ERROR logs/abutre.log

# Últimas 100 linhas
tail -n 100 logs/abutre.log
```

**Banco de dados:**
```bash
# Abrir SQLite interativo
sqlite3 data/abutre.db

# Ver tabelas
.tables

# Ver trades
SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;

# Sair
.quit
```

**Dashboard:**
- URL: https://botderiv.rollabs.com.br/abutre
- Backend: wss://botderivapi.roilabs.com.br
- Se desconectado: Verificar backend Easypanel

---

**Última atualização:** 22/12/2025
