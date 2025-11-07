# 🔧 INSTALAR DEPENDÊNCIAS - GUIA DE SOLUÇÃO

**Problema:** `ModuleNotFoundError: No module named 'websockets'`

---

## ✅ SOLUÇÃO PARA SEU SISTEMA

Você tem Python 3.13 instalado em `C:\Python313\` mas precisa instalar as dependências.

### Opção 1: Script Automático (RECOMENDADO)

```powershell
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main\backend
.\install_fix.bat
```

### Opção 2: Comandos Manuais

```powershell
# Instalar websockets e ujson
C:\Python313\python.exe -m pip install websockets ujson

# Verificar instalação
C:\Python313\python.exe -c "import websockets; import ujson; print('OK')"
```

### Opção 3: Instalar TODAS as dependências

```powershell
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main\backend
C:\Python313\python.exe -m pip install -r requirements.txt
```

---

## 🚀 DEPOIS DE INSTALAR

Execute o teste com o caminho completo do Python:

```powershell
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main\backend
C:\Python313\python.exe test_simple_order.py
```

---

## 💡 ADICIONAR PYTHON AO PATH (OPCIONAL)

Para poder usar apenas `python` sem o caminho completo:

1. Pressione `Win + R`
2. Digite `sysdm.cpl` e pressione Enter
3. Vá para a aba "Avançado"
4. Clique em "Variáveis de Ambiente"
5. Em "Variáveis do sistema", selecione "Path" e clique em "Editar"
6. Clique em "Novo" e adicione: `C:\Python313`
7. Clique em "Novo" novamente e adicione: `C:\Python313\Scripts`
8. Clique em "OK" em todas as janelas
9. **REINICIE o PowerShell/Terminal**

Depois disso você poderá usar:
```powershell
python test_simple_order.py
pip install websockets
```

---

## 🐛 TROUBLESHOOTING

### Erro: "Fatal error in launcher: Unable to create process"
**Solução:** Use o caminho completo do Python:
```powershell
C:\Python313\python.exe -m pip install websockets ujson
```

### Erro: "Could not find platform independent libraries"
**Causa:** Instalação do Python incompleta ou corrompida
**Solução:**
1. Desinstale o Python
2. Reinstale de https://www.python.org/downloads/
3. Durante a instalação, marque "Add Python to PATH"

### Python não encontrado
**Verifique a instalação:**
```powershell
# Verificar se existe
dir C:\Python313\python.exe

# Se não existir, procure em:
dir "C:\Program Files\Python313\python.exe"
dir "C:\Users\jeanz\AppData\Local\Programs\Python\Python313\python.exe"
```

---

## ✅ VALIDAR INSTALAÇÃO

Depois de instalar, execute este comando para verificar:

```powershell
C:\Python313\python.exe -c "import sys; import websockets; import ujson; print(f'Python {sys.version}'); print('✓ websockets instalado'); print('✓ ujson instalado')"
```

**Resultado esperado:**
```
Python 3.13.x ...
✓ websockets instalado
✓ ujson instalado
```

---

## 📞 AINDA COM PROBLEMAS?

### Última tentativa - Criar ambiente virtual:

```powershell
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main

# Criar venv
C:\Python313\python.exe -m venv venv

# Ativar venv
.\venv\Scripts\Activate.ps1

# Instalar dependências
pip install websockets ujson

# Testar
cd backend
python test_simple_order.py
```

Se o PowerShell bloquear a execução de scripts, execute antes:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

**Criado:** 2025-11-06
**Para:** Sistema com Python 3.13 em C:\Python313\
