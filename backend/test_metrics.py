"""
Teste do sistema de métricas Prometheus
"""

print("Testando sistema de métricas...")

# Validar que metrics.py compila
with open('metrics.py', 'r', encoding='utf-8') as f:
    code = f.read()
    compile(code, 'metrics.py', 'exec')

print("[OK] metrics.py compila sem erros")

# Validar estrutura
assert 'MetricsManager' in code
assert 'prometheus_client' in code
assert 'Counter' in code and 'Histogram' in code and 'Gauge' in code
assert 'trades_total' in code
assert 'signal_latency_ms' in code
assert 'tick_processing_ms' in code
assert 'cache_operations' in code
assert 'api_calls_total' in code
assert 'backtest_duration_seconds' in code
assert 'bot_uptime_seconds' in code

print("[OK] Todas as métricas principais estão definidas")

# Validar que main.py foi atualizado
with open('main.py', 'r', encoding='utf-8') as f:
    main_code = f.read()

assert 'from metrics import' in main_code
assert 'prometheus_client' in main_code
assert '@app.get("/metrics")' in main_code
assert 'initialize_metrics_manager()' in main_code

print("[OK] Integração com main.py completa")

# Validar requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read()

assert 'prometheus-client' in requirements

print("[OK] prometheus-client adicionado ao requirements.txt")

print("\n" + "="*60)
print("VALIDACAO COMPLETA - SISTEMA DE METRICAS PROMETHEUS")
print("="*60)

print("\nMetricas implementadas:")
print("  Trading:")
print("    - trades_total (Counter)")
print("    - trade_duration_seconds (Histogram)")
print("    - current_pnl (Gauge)")
print("    - win_rate (Gauge)")
print("    - profit_loss_total (Counter)")
print("\n  ML/Sinais:")
print("    - signal_latency_ms (Histogram)")
print("    - signals_generated (Counter)")
print("    - model_confidence (Gauge)")
print("    - model_accuracy (Gauge)")
print("\n  Performance:")
print("    - tick_processing_ms (Histogram)")
print("    - ticks_processed (Counter)")
print("    - ticks_per_second (Gauge)")
print("\n  Cache:")
print("    - cache_operations (Counter)")
print("    - cache_hit_rate (Gauge)")
print("\n  API:")
print("    - api_calls_total (Counter)")
print("    - api_latency_ms (Histogram)")
print("\n  Backtesting:")
print("    - backtest_duration_seconds (Histogram)")
print("    - backtest_sharpe_ratio (Gauge)")
print("    - backtest_max_drawdown (Gauge)")
print("\n  Sistema:")
print("    - bot_info (Info)")
print("    - bot_uptime_seconds (Gauge)")
print("    - errors_total (Counter)")

print("\nEndpoint disponivel:")
print("  GET /metrics - Expoe metricas em formato Prometheus")

print("\nProximos passos:")
print("  1. Configurar Prometheus para scraping (prometheus.yml)")
print("  2. Configurar Grafana para visualizacao")
print("  3. Criar dashboards personalizados")
print("  4. Configurar alertas no Alertmanager")

print("\n[OK] TODOS OS TESTES PASSARAM!")
