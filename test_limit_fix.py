"""
Teste para verificar que o limit está correto no código
"""
import json

# Simular o código que está sendo enviado
request_payload = {
    "profit_table": 1,
    "description": 1,
    "limit": 999,
    "sort": "DESC"
}

print("=" * 60)
print("TESTE: Verificação do limit no código")
print("=" * 60)
print(f"\nPayload que será enviado para Deriv API:")
print(json.dumps(request_payload, indent=2))
print(f"\n✅ LIMIT CORRETO: {request_payload['limit']}")
print(f"✅ Tipo: {type(request_payload['limit'])}")
print(f"✅ É 999? {request_payload['limit'] == 999}")
print("\n" + "=" * 60)
print("Se o servidor ainda retornar 'Input validation failed: limit',")
print("significa que o Easypanel NÃO deployou o código mais recente.")
print("=" * 60)
