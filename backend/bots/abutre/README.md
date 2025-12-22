# 🦅 ABUTRE BOT - Delayed Martingale Trading System

**Sistema de Trading Automatizado** para mercados de volatilidade (Deriv.com)

**Status:** 🟢 FORWARD TEST EM ANDAMENTO (Dia 1/30)

---

## 📊 Performance Backtest (180 dias)

| Métrica | Resultado |
|---------|-----------|
| **ROI** | +40.25% ($2,000 → $2,805) |
| **Win Rate** | 100% (1,018 trades) |
| **Max Drawdown** | 24.81% |
| **Busts** | 0 |
| **Profit/Trade** | $0.79 |

---

## 🚀 Quick Start

### Dashboard em Produção

Acesse: **https://botderiv.rollabs.com.br/abutre**

**O que você verá:**
- ✅ Métricas em tempo real (Balance, ROI, Win Rate)
- ✅ Gráfico de Equity Curve
- ✅ Tabela de trades
- ✅ Monitor de mercado (streak detector)
- ✅ Painel de risco

### Monitoramento Diário

```bash
cd backend/bots/abutre
python scripts/get_daily_metrics.py
```

**Output:**
- Saldo atual, ROI, Win Rate, Max Drawdown
- Trades do dia
- Últimos 5 trades
- **Entrada formatada para copiar no log**

---

## 📖 Documentação Completa

| Documento | Descrição |
|-----------|-----------|
| **[STATUS.md](STATUS.md)** | 📊 **Status atual do projeto** - Leia PRIMEIRO |
| **[SETUP_DERIV_API.md](SETUP_DERIV_API.md)** | 🔑 Como configurar token da Deriv API |
| **[FORWARD_TEST_LOG.md](FORWARD_TEST_LOG.md)** | 📅 Log de monitoramento (30 dias) |
| **[scripts/README.md](scripts/README.md)** | 🛠️ Guia de scripts utilitários |

---

## 🎯 Estratégia: Delayed Martingale

**Como funciona:**
1. Aguarda 8+ velas consecutivas da mesma cor
2. Abre trade na direção OPOSTA (reversão)
3. Se perder, dobra stake (Martingale até nível 10)
4. Win → Reset para nível 1

**Por que funciona:** Streaks longas eventualmente revertem (validado em 180 dias, 100% win rate)

---

## 📝 Workflow Diário (5 min)

1. Acessar: https://botderiv.rollabs.com.br/abutre
2. Executar: `python scripts/get_daily_metrics.py`
3. Copiar saída e atualizar `FORWARD_TEST_LOG.md`
4. Commit: `git add FORWARD_TEST_LOG.md && git commit -m "docs: Day X metrics"`

---

## 🔐 Segurança

**Configuração atual:** ✅ DEMO (sem risco financeiro)
- Conta: VRTC (virtual)
- Paper Trading: ATIVO (AUTO_TRADING=false)
- Execução: SIMULADA (não abre trades reais)

---

## ✅ Status Atual

**FASE 3.1:** Forward Test iniciado (22/12/2025)
**DURAÇÃO:** 30 dias
**OBJETIVO:** ROI > 5%, Win Rate > 90%
**PRÓXIMO:** Monitoramento diário

---

**Desenvolvido com Claude Code** 🤖
