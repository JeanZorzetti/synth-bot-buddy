# Relatório de Validação - Forward Testing

## Informações Gerais

- **Símbolo**: 1HZ100V
- **Início**: 2025-12-17T23:41:25.605118
- **Duração**: 0.1 dias (2.4 horas)
- **Status**: 🔴 Parado

## Parâmetros de Trading

- **Capital Inicial**: $10,000.00
- **Confidence Threshold**: 40.0%
- **Tamanho Máximo de Posição**: 2.0% do capital
- **Stop Loss**: 0.5%
- **Take Profit**: 0.75%
- **Risk:Reward Ratio**: 1:1.5

## Performance de Trading

### Métricas Gerais
- **Capital Atual**: $4,100.81
- **P&L Total**: $-5,899.19 (-58.99%)
- **Capital Máximo**: $10,000.00
- **Max Drawdown**: 58.99%

### Trades
- **Total de Trades**: 42
- **Trades Vencedores**: 7
- **Trades Perdedores**: 35
- **Win Rate**: 16.67%
- **Profit Factor**: 0.08
- **Sharpe Ratio**: -0.89
- **Lucro Médio por Trade**: $-140.46

## Previsões ML

- **Total de Previsões**: 0
- **Confidence Média**: 0.00%
- **Previsões com Alta Confidence (>40%)**: 0.0%
- **Execução Rate**: 0.0% (trades executados / previsões)

## Bugs e Problemas

- **Total de Bugs Registrados**: 1

### Bugs por Tipo

- **market_data_fetch_error**: 1

### Bugs Críticos

- [2025-12-18T00:55:28.784192] no close frame received or sent

## Validação de Objetivos

### Critérios de Aprovação (FASE 8)

| Métrica | Objetivo | Atual | Status |
|---------|----------|-------|--------|
| Win Rate | > 60% | 16.7% | ❌ FAIL |
| Sharpe Ratio | > 1.5 | -0.89 | ❌ FAIL |
| Max Drawdown | < 15% | 59.0% | ❌ FAIL |
| Profit Factor | > 1.5 | 0.08 | ❌ FAIL |

### Status Geral

**❌ REPROVADO**

Sistema atendeu apenas 0/4 critérios. Necessário ajustes significativos.

## Próximos Passos

1. Analisar trades perdedores para identificar padrões
2. Ajustar thresholds de confidence se necessário
3. Considerar otimização de stop loss / take profit
4. Avaliar adicionar filtros de contexto de mercado
5. Testar em outros símbolos para validar robustez

---

*Relatório gerado automaticamente pelo Forward Testing Engine*
*Data: 2025-12-18T02:03:24.856434*
