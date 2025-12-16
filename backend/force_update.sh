#!/bin/bash
# Script para forçar update do código no Easypanel
# Execute no Easypanel Console: bash force_update.sh

echo "🔧 FORÇANDO UPDATE DO CÓDIGO"
echo "===================================================="

# 1. Ir para diretório raiz
cd /app 2>/dev/null || cd /workspace 2>/dev/null || cd $(git rev-parse --show-toplevel)
echo "📂 Diretório: $(pwd)"

# 2. Mostrar versão atual
echo ""
echo "📦 Versão ANTES do update:"
git log -1 --format='%h - %s'

# 3. Fetch da origin
echo ""
echo "📥 Buscando atualizações..."
git fetch origin main

# 4. Reset hard para origin/main
echo ""
echo "🔄 Aplicando código novo..."
git reset --hard origin/main

# 5. Mostrar nova versão
echo ""
echo "📦 Versão DEPOIS do update:"
git log -1 --format='%h - %s'

# 6. Verificar se fix está presente
echo ""
echo "✅ Verificando fixes aplicados:"

if grep -q "get_latest_tick" backend/deriv_api_legacy.py; then
    echo "   ✅ Fix ticks_history presente"
else
    echo "   ❌ Fix ticks_history FALTANDO"
fi

if grep -q "Aguardando histórico" backend/forward_testing.py; then
    echo "   ✅ Fix warm-up filter presente"
else
    echo "   ❌ Fix warm-up filter FALTANDO"
fi

if grep -q "code_version" backend/main.py; then
    echo "   ✅ Verificação de versão presente"
else
    echo "   ❌ Verificação de versão FALTANDO"
fi

# 7. Instruções finais
echo ""
echo "===================================================="
echo "✅ UPDATE COMPLETO!"
echo ""
echo "⚠️  IMPORTANTE: Agora você precisa reiniciar o backend"
echo ""
echo "Via Easypanel UI:"
echo "  1. Ir em Services → Backend"
echo "  2. Clicar em 'Restart'"
echo ""
echo "Ou via Console (se disponível):"
echo "  supervisorctl restart backend"
echo ""
echo "Após reiniciar, verificar:"
echo "  curl http://localhost:8000/health | jq '.git_commit'"
echo "  # Deve retornar: \"3bd2f36\" ou superior"
echo "===================================================="
