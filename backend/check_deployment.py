#!/usr/bin/env python3
"""
Script de diagnóstico para verificar deployment em produção
Execute no Easypanel Console: python check_deployment.py
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    """Executa comando e retorna output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return f"ERROR: {e}", 1

def check_git_version():
    """Verifica versão do código via git"""
    print("\n" + "="*60)
    print("1. VERIFICANDO VERSÃO DO CÓDIGO")
    print("="*60)

    # Verificar se está em um repo git
    output, code = run_command("git rev-parse --is-inside-work-tree")
    if code != 0:
        print("❌ Não está em um repositório git")
        return None

    # Pegar commit atual
    commit, code = run_command("git rev-parse --short HEAD")
    if code == 0:
        print(f"📦 Commit atual: {commit}")
    else:
        print(f"❌ Erro ao obter commit: {commit}")
        return None

    # Pegar branch
    branch, code = run_command("git branch --show-current")
    if code == 0:
        print(f"🌿 Branch: {branch}")
    else:
        print(f"⚠️  Não foi possível determinar branch")

    # Verificar status
    status, code = run_command("git status --short")
    if status:
        print(f"⚠️  Arquivos modificados:\n{status}")
    else:
        print("✅ Working directory limpo")

    # Verificar se está atualizado com origin
    output, code = run_command("git fetch origin && git rev-list HEAD..origin/main --count")
    if code == 0 and output:
        commits_behind = int(output)
        if commits_behind > 0:
            print(f"⚠️  {commits_behind} commits atrás de origin/main")
            print("   Execute: git pull origin main")
        else:
            print("✅ Atualizado com origin/main")

    return commit

def check_critical_files():
    """Verifica se arquivos críticos existem e têm os fixes"""
    print("\n" + "="*60)
    print("2. VERIFICANDO ARQUIVOS CRÍTICOS")
    print("="*60)

    files_to_check = {
        "backend/forward_testing.py": [
            "get_latest_tick",  # Fix de ticks_history
            "Aguardando histórico",  # Fix de warm-up
        ],
        "backend/deriv_api_legacy.py": [
            "async def get_latest_tick",  # Método novo
        ],
        "backend/main.py": [
            "code_version",  # Verificação de versão
            "git_commit",  # Health check
        ]
    }

    all_ok = True

    for filepath, patterns in files_to_check.items():
        if not Path(filepath).exists():
            print(f"❌ {filepath} NÃO ENCONTRADO")
            all_ok = False
            continue

        print(f"\n📄 {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        for pattern in patterns:
            if pattern in content:
                print(f"   ✅ {pattern}")
            else:
                print(f"   ❌ {pattern} FALTANDO")
                all_ok = False

    return all_ok

def check_deriv_token():
    """Verifica se token Deriv está configurado"""
    print("\n" + "="*60)
    print("3. VERIFICANDO CONFIGURAÇÃO")
    print("="*60)

    token = os.getenv("DERIV_API_TOKEN")
    if token:
        print(f"✅ DERIV_API_TOKEN configurado ({len(token)} caracteres)")
        print(f"   Primeiros 10 chars: {token[:10]}...")
    else:
        print("❌ DERIV_API_TOKEN não configurado")
        print("   Configure em Easypanel: Settings → Environment Variables")

    return bool(token)

def check_running_processes():
    """Verifica processos rodando"""
    print("\n" + "="*60)
    print("4. VERIFICANDO PROCESSOS")
    print("="*60)

    # Verificar se uvicorn está rodando
    output, code = run_command("ps aux | grep uvicorn | grep -v grep")
    if code == 0 and output:
        print("✅ Uvicorn está rodando:")
        for line in output.split('\n')[:3]:  # Primeiras 3 linhas
            print(f"   {line}")
    else:
        print("⚠️  Uvicorn não encontrado")

    # Verificar portas abertas
    output, code = run_command("netstat -tlnp 2>/dev/null | grep :8000 || ss -tlnp 2>/dev/null | grep :8000")
    if code == 0 and output:
        print("✅ Porta 8000 está em uso")
    else:
        print("⚠️  Porta 8000 não está aberta")

def suggest_actions(commit, files_ok, token_ok):
    """Sugere ações baseadas no diagnóstico"""
    print("\n" + "="*60)
    print("5. AÇÕES RECOMENDADAS")
    print("="*60)

    # Versão esperada
    expected_commit = "3bd2f36"

    if commit != expected_commit:
        print(f"\n⚠️  CÓDIGO DESATUALIZADO")
        print(f"   Atual: {commit}")
        print(f"   Esperado: {expected_commit}")
        print("\n   AÇÕES:")
        print("   1. git fetch origin")
        print("   2. git reset --hard origin/main")
        print("   3. Reiniciar backend (via Easypanel UI)")

    if not files_ok:
        print(f"\n❌ ARQUIVOS CRÍTICOS COM PROBLEMAS")
        print("   Provavelmente código não foi atualizado corretamente")
        print("   Execute: git reset --hard origin/main")

    if not token_ok:
        print(f"\n❌ TOKEN DERIV NÃO CONFIGURADO")
        print("   Configure em Easypanel:")
        print("   Settings → Environment Variables → Add Variable")
        print("   Nome: DERIV_API_TOKEN")
        print("   Valor: <seu_token>")

    if commit == expected_commit and files_ok and token_ok:
        print("\n✅ TUDO OK!")
        print("   Sistema deve estar funcionando corretamente")
        print("\n   Verificar Forward Testing:")
        print("   1. Acessar: https://botderiv.roilabs.com.br/api/forward-testing/status")
        print("   2. Verificar: code_version.commit deve ser '9ec01f0' ou superior")
        print("   3. Iniciar: POST https://botderiv.roilabs.com.br/api/forward-testing/start")

def main():
    print("🔍 DIAGNÓSTICO DE DEPLOYMENT - FORWARD TESTING")
    print("="*60)

    commit = check_git_version()
    files_ok = check_critical_files()
    token_ok = check_deriv_token()
    check_running_processes()
    suggest_actions(commit, files_ok, token_ok)

    print("\n" + "="*60)
    print("✅ DIAGNÓSTICO COMPLETO")
    print("="*60)
    print("\nPróximo passo: Seguir as ações recomendadas acima")

if __name__ == "__main__":
    main()
