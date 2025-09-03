# 🐍 Guia de Instalação do Python

## Problema Identificado
O Python está instalado, mas há problemas com o pip e as bibliotecas. Isso é comum em algumas instalações.

## ✅ Solução Recomendada

### 1. Reinstalar Python Completamente
1. **Baixe** a versão mais recente: https://python.org/downloads/
2. **Durante a instalação**:
   - ✅ Marque "Add Python to PATH"
   - ✅ Marque "Install pip"
   - ✅ Escolha "Install for all users"

### 2. Verificar Instalação
Após reinstalar, teste no terminal:
```bash
python --version
python -m pip --version
```

### 3. Instalar Dependências do Projeto
```bash
cd backend
python -m pip install -r requirements.txt
```

### 4. Testar Conexão
```bash
python test_connection.py
```

### 5. Executar Servidor
```bash
python start.py
```

## 🔧 Alternativa: Anaconda
Se continuar com problemas, use o Anaconda:
1. Baixe: https://www.anaconda.com/download
2. Instale normalmente
3. Use o "Anaconda Prompt"
4. Execute os comandos acima

## 🐳 Alternativa: Docker
Se preferir usar Docker:
```bash
cd backend
docker build -t synth-bot-buddy .
docker run -p 8000:8000 --env-file .env synth-bot-buddy
```

## 🆘 Se Nada Funcionar
Use o VS Code com extensão Python:
1. Instale VS Code
2. Instale extensão Python
3. Use o terminal integrado do VS Code
4. Execute os comandos normalmente

---
**💡 Dica**: Após resolver, execute `python test_connection.py` para validar que tudo está funcionando com seu token Deriv!