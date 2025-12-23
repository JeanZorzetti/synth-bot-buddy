# TESTE DA VERSÃO SIMPLIFICADA - LOCAL

## O que você vai ver:

Quando abrir http://localhost:8081/abutre você vai ver:

**APENAS ISSO:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Abutre Bot - Histórico de Trades          [🔄 Atualizar]     │
│  Todas as operações executadas                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ID | Data/Hora | Direção | Stake | Level | Resultado | Profit │
│ ────┼───────────┼─────────┼───────┼───────┼───────────┼────── │
│  #1 | 22/12 ... │  CALL   │ $1.00 │   1   │    WIN    │ +$0.95│
│  #2 | 22/12 ... │  PUT    │ $1.00 │   1   │    LOSS   │ -$1.00│
│  ...                                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Mostrando X trades
```

**NADA DE:**
- ❌ Cards no topo
- ❌ Gráficos
- ❌ Botões "Iniciar/Parar"
- ❌ Configurações
- ❌ Alertas
- ❌ Market Monitor

**SÓ:**
- ✅ Título
- ✅ Botão "Atualizar"
- ✅ Tabela de trades

---

## Como testar AGORA:

### 1. Rodar localmente (5 segundos)

```bash
cd frontend
npm run dev
```

Depois abra: **http://localhost:8081/abutre**

Você vai ver a página preta com a tabela simples!

---

## Se quiser fazer BUILD de produção localmente:

```bash
cd frontend
npm run build
```

Vai criar a pasta `frontend/dist/` com os arquivos otimizados.

---

## Próximo passo: DEPLOY EM PRODUÇÃO

Depois que confirmar que funciona localmente, siga as instruções em:
**DEPLOY_FRONTEND_NOW.md**

---

**CÓDIGO JÁ ESTÁ NO GITHUB!**
Commit: `ca40d82`

A única coisa que falta é fazer o deploy no servidor de produção.
