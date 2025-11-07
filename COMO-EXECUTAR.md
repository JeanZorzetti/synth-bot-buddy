# 🚀 COMO EXECUTAR O PROJETO

**Solução definitiva para problemas de Python/Dependências**

---

## ⚠️ SEU PROBLEMA

Você está tendo erro: `Could not find platform independent libraries`

**Causa:** Instalação do Python 3.13 está incompleta ou corrompida.

---

## ✅ SOLUÇÃO DEFINITIVA (Ambiente Virtual)

### Passo 1: Criar Ambiente Virtual

Execute este comando na **raiz do projeto**:

```powershell
.\setup_venv.bat
```

Isso vai:
1. Criar um ambiente virtual limpo em `.venv`
2. Instalar todas as dependências necessárias
3. Isolar o projeto de problemas do Python do sistema

---

### Passo 2: Ativar o Ambiente Virtual

**Sempre que abrir um novo terminal**, execute:

```powershell
.venv\Scripts\activate
```

Você verá `(.venv)` no início da linha do terminal.

---

### Passo 3: Executar o Teste

```powershell
cd backend
python test_simple_order.py
```

---

## 📋 COMANDOS COMPLETOS (COPIE E COLE)

### Primeira vez (Setup):

```powershell
# Na raiz do projeto
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main

# Criar ambiente virtual e instalar dependências
.\setup_venv.bat

# Ativar ambiente
.venv\Scripts\activate

# Testar
cd backend
python test_simple_order.py
```

### Próximas vezes (já tem ambiente configurado):

```powershell
# Na raiz do projeto
cd C:\Users\jeanz\OneDrive\Desktop\Jizreel\synth-bot-buddy-main

# Ativar ambiente
.venv\Scripts\activate

# Testar
cd backend
python test_simple_order.py
```

---

## 🎯 RESULTADO ESPERADO

Após executar `setup_venv.bat`:

```
============================================================
  ✓ Setup Concluido com Sucesso!
============================================================

COMO USAR:

1. Ative o ambiente virtual:
   .venv\Scripts\activate

2. Execute o teste:
   cd backend
   python test_simple_order.py
```

---

## 🔧 EXECUTAR O BACKEND

```powershell
# Ativar ambiente
.venv\Scripts\activate

# Ir para backend
cd backend

# Executar servidor
python start.py
```

O servidor estará em: http://localhost:8000

---

## 📊 ESTRUTURA DO AMBIENTE VIRTUAL

```
synth-bot-buddy-main/
├── .venv/                    ← Ambiente virtual (criado)
│   ├── Scripts/
│   │   ├── activate.bat      ← Ativar no Windows
│   │   └── python.exe        ← Python isolado
│   └── Lib/                  ← Dependências isoladas
├── backend/
│   ├── test_simple_order.py  ← Teste de ordem
│   └── start.py              ← Servidor backend
└── setup_venv.bat            ← Script de setup
```

---

## ❓ PERGUNTAS FREQUENTES

### Por que usar ambiente virtual?

**Vantagens:**
- ✅ Isola o projeto de problemas do Python do sistema
- ✅ Cada projeto tem suas próprias dependências
- ✅ Evita conflitos de versão
- ✅ Fácil de limpar e recriar

### Como saber se o ambiente está ativo?

Você verá `(.venv)` no início da linha do terminal:
```powershell
(.venv) PS C:\Users\jeanz\...\synth-bot-buddy-main>
```

### Como desativar o ambiente?

```powershell
deactivate
```

### Posso deletar o .venv e criar de novo?

Sim! Se algo der errado:
```powershell
# Deletar
rmdir /s /q .venv

# Criar novamente
.\setup_venv.bat
```

---

## 🐛 TROUBLESHOOTING

### Erro: "execution of scripts is disabled"

Execute antes de ativar o ambiente:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Erro ao criar venv

Se `C:\Python313\python.exe -m venv .venv` falhar:

**Opção 1:** Reinstalar Python
1. Desinstale Python atual
2. Baixe de: https://www.python.org/downloads/
3. Durante instalação, marque "Add Python to PATH"

**Opção 2:** Usar Python portável
1. Baixe Python embeddable de python.org
2. Extraia em uma pasta
3. Use o caminho completo no script

---

## ✅ CHECKLIST

Após executar `setup_venv.bat`:

- [ ] Viu mensagem "Setup Concluido com Sucesso"
- [ ] Existe pasta `.venv` na raiz do projeto
- [ ] Consegue ativar ambiente: `.venv\Scripts\activate`
- [ ] Vê `(.venv)` no terminal
- [ ] Consegue executar: `python test_simple_order.py`

---

## 🎯 PRÓXIMOS PASSOS

Depois que o ambiente funcionar:

1. **Configurar token Deriv**
   - Edite `backend/test_simple_order.py` linha 16
   - Ou use: `set DERIV_TOKEN=seu_token`

2. **Executar teste**
   ```powershell
   cd backend
   python test_simple_order.py
   ```

3. **Ver resultado na Deriv**
   - Acesse link do contrato
   - Aguarde resultado (5 min)

---

**Criado:** 2025-11-06
**Versão:** 1.0
