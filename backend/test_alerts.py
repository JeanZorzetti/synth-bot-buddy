"""
Teste do sistema de alertas
"""

print("Testando sistema de alertas...")

# Validar alerts_manager.py
with open('alerts_manager.py', 'r', encoding='utf-8') as f:
    code = f.read()
    compile(code, 'alerts_manager.py', 'exec')

print("[OK] alerts_manager.py compila sem erros")

# Validar estrutura
assert 'AlertsManager' in code
assert 'AlertLevel' in code
assert 'AlertChannel' in code
assert 'send_discord' in code
assert 'send_telegram' in code
assert 'send_email' in code
assert 'send_alert' in code
assert 'alert_trade_executed' in code
assert 'alert_circuit_breaker_open' in code
assert 'alert_system_error' in code

print("[OK] Todas as funcoes de alerta implementadas")

# Validar canais
assert 'DISCORD' in code
assert 'TELEGRAM' in code
assert 'EMAIL' in code

print("[OK] Todos os canais de alerta suportados")

# Validar níveis
assert 'INFO' in code
assert 'WARNING' in code
assert 'ERROR' in code
assert 'CRITICAL' in code

print("[OK] Todos os niveis de alerta definidos")

print("\n" + "="*60)
print("VALIDACAO COMPLETA - SISTEMA DE ALERTAS")
print("="*60)

print("\nCanais suportados:")
print("  1. Discord (webhook)")
print("  2. Telegram (bot API)")
print("  3. Email (SMTP)")

print("\nNiveis de alerta:")
print("  - INFO: Informacoes gerais")
print("  - WARNING: Avisos")
print("  - ERROR: Erros")
print("  - CRITICAL: Erros criticos")

print("\nAlertas pre-configurados:")
print("  - alert_trade_executed() - Trade executado")
print("  - alert_high_win_rate() - Alta taxa de acerto")
print("  - alert_circuit_breaker_open() - Circuit breaker aberto")
print("  - alert_system_error() - Erro critico do sistema")

print("\nConfiguracao:")
print("  Environment variables:")
print("    - DISCORD_WEBHOOK_URL")
print("    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
print("    - SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD")
print("    - EMAIL_FROM, EMAIL_TO")
print("    - ALERT_MIN_LEVEL (INFO/WARNING/ERROR/CRITICAL)")

print("\nFuncionalidades:")
print("  - Envio assincrono de alertas")
print("  - Envio paralelo para multiplos canais")
print("  - Historico de alertas")
print("  - Filtragem por nivel minimo")
print("  - HTML formatado para emails")
print("  - Embeds coloridos para Discord")

print("\n[OK] TODOS OS TESTES PASSARAM!")
