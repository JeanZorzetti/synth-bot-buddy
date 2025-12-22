# ABUTRE BOT - FORWARD TEST LOG (30 DIAS)

**Objetivo:** Validar estratégia Delayed Martingale com dados reais do mercado antes de usar capital real.

---

## 📊 INFORMAÇÕES DO TESTE

| Parâmetro | Valor |
|-----------|-------|
| **Início** | 2025-12-22 |
| **Fim previsto** | 2026-01-21 (30 dias) |
| **Modo** | Paper Trading (AUTO_TRADING=false) |
| **Conta** | DEMO (VRTC) - Token: paE5sSemx3oANLE |
| **Símbolo** | V100 (1HZ100V) |
| **Banca inicial** | $2,000.00 |
| **Stake inicial** | $1.00 |
| **Delay** | 8 velas |
| **Max Level** | 10 |
| **Multiplier** | 2.0 |

---

## 🎯 MÉTRICAS ALVO (30 DIAS)

| Métrica | Backtest (180d) | Esperado (30d) | Status |
|---------|-----------------|----------------|--------|
| **ROI** | +40.25% | +6-7% | ⏳ Aguardando |
| **Win Rate** | 100% | > 95% | ⏳ Aguardando |
| **Max Drawdown** | 18.2% | < 30% | ⏳ Aguardando |
| **Total Trades** | 1,018 | ~170 | ⏳ Aguardando |
| **Busts** | 0 | 0 | ⏳ Aguardando |

**Critério de Aprovação:**
- ✅ ROI > 5% E Win Rate > 90% → Avançar para Fase 3.2 (Paper Trading Real)
- ❌ ROI < 0% OU Bust → Aumentar Delay para 10 e repetir

---

## 📅 LOG DIÁRIO

### Semana 1 (22/12/2025 - 28/12/2025)

#### 📆 Dia 1 - 22/12/2025

**Status:** ✅ Bot iniciado e conectado

**Configuração:**
- Dashboard: https://botderiv.rollabs.com.br/abutre
- Backend: wss://botderivapi.roilabs.com.br/ws/dashboard
- WebSocket: Conectado ✅
- Paper Trading: Ativo ✅

**Métricas:**
- Saldo: $2,000.00
- Total Trades: 0
- Win Rate: N/A
- Max Drawdown: 0%
- ROI: 0%

**Eventos:**
- [00:00] Bot iniciado em produção
- [00:00] Dashboard conectado ao backend Easypanel
- [00:00] Aguardando primeiro gatilho (8+ velas consecutivas)

**Observações:**
- ✅ WebSocket funcionando corretamente
- ✅ Token DEMO configurado
- ✅ Paper trading confirmado (sem execução real)
- ⏳ Aguardando sinais do mercado

---

#### 📆 Dia 2 - 23/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

---

#### 📆 Dia 3 - 24/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

---

#### 📆 Dia 4 - 25/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

---

#### 📆 Dia 5 - 26/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

---

#### 📆 Dia 6 - 27/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

---

#### 📆 Dia 7 - 28/12/2025

**Métricas:**
- Saldo: $
- Total Trades:
- Win Rate: %
- Max Drawdown: %
- ROI: %

**Eventos:**
-

**Observações:**
-

**📊 Resumo Semanal:**
- ROI Semanal: %
- Trades/dia médio:
- Win Rate: %
- Max DD: %

---

### Semana 2 (29/12/2025 - 04/01/2026)

#### 📆 Dia 8 - 29/12/2025

_A preencher..._

---

### Semana 3 (05/01/2026 - 11/01/2026)

#### 📆 Dia 15 - 05/01/2026

_A preencher..._

---

### Semana 4 (12/01/2026 - 18/01/2026)

#### 📆 Dia 22 - 12/01/2026

_A preencher..._

---

### Semana 5 (19/01/2026 - 21/01/2026)

#### 📆 Dia 29 - 19/01/2026

_A preencher..._

---

## 📊 ANÁLISE FINAL (21/01/2026)

_A ser preenchido ao final dos 30 dias._

**Métricas Finais:**
- Saldo Final: $
- ROI: %
- Win Rate: %
- Max Drawdown: %
- Total Trades:
- Busts:

**Comparação com Backtest:**
| Métrica | Backtest (180d) | Forward (30d) | Diferença |
|---------|-----------------|---------------|-----------|
| ROI | +40.25% | % | % |
| Win Rate | 100% | % | % |
| Max DD | 18.2% | % | % |
| Trades | 1,018 | | |

**Decisão:**
- [ ] ✅ APROVADO - Avançar para FASE 3.2 (Paper Trading Real)
  - Motivo: ROI > 5% E Win Rate > 90%
- [ ] ❌ REPROVADO - Ajustar parâmetros e repetir
  - Motivo: ROI < 0% OU Bust ocorreu
  - Ação: Aumentar DELAY_THRESHOLD para 10 e repetir teste

---

## 🔧 TROUBLESHOOTING

### Problemas Comuns

**Bot não conecta:**
- Verificar token DEMO em `.env`
- Verificar URL WebSocket
- Verificar logs: `backend/bots/abutre/logs/abutre.log`

**Dashboard mostra "Disconnected":**
- Verificar backend Easypanel está rodando
- Verificar firewall/proxy WebSocket
- Verificar VITE_WS_URL em produção

**Trades não aparecem:**
- Normal se não houver gatilho (8+ velas)
- Mercado pode estar sem streaks
- Verificar logs para "trigger_detected"

---

## 📁 RECURSOS

**Arquivos importantes:**
- Configuração: `backend/bots/abutre/.env`
- Logs: `backend/bots/abutre/logs/abutre.log`
- Database: `backend/bots/abutre/data/abutre.db`
- Dashboard: https://botderiv.rollabs.com.br/abutre

**Comandos úteis:**
```bash
# Ver logs em tempo real
tail -f backend/bots/abutre/logs/abutre.log

# Exportar dados do banco
sqlite3 backend/bots/abutre/data/abutre.db ".dump" > backup.sql

# Verificar status do bot
curl https://botderivapi.roilabs.com.br/health
```

---

## 📝 NOTAS

- **22/12/2025:** Forward Test iniciado. Bot em paper trading (sem risco financeiro).
- Dashboard integrado ao frontend principal em `/abutre`
- WebSocket conectado ao backend Easypanel (wss://botderivapi.roilabs.com.br)
- Próxima revisão: Diária (até 21/01/2026)

---

**Status:** 🟢 EM ANDAMENTO

**Última atualização:** 22/12/2025
