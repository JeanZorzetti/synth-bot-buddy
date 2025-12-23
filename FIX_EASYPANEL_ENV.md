# 🔧 FIX EASYPANEL - Variável de Ambiente Faltando

## ❌ Erro Atual

```
Environment Variable "NEXT_PUBLIC_WS_URL" references Secret "ws_url", which does not exist.
```

---

## ✅ SOLUÇÃO

No Easypanel, você precisa **ADICIONAR a variável de ambiente**:

### 1. Acessar Configurações do Frontend

**Easypanel → Frontend (abutre-dashboard) → Environment Variables**

### 2. Adicionar Variável

**Nome**: `NEXT_PUBLIC_WS_URL`

**Valor**: `https://botderivapi.roilabs.com.br`

**Tipo**: Environment Variable (NÃO Secret)

### 3. Salvar e Rebuild

1. Clicar em **"Save"**
2. Ir em **"Deployments"**
3. Clicar em **"Force Rebuild"**

---

## 📋 Valores Esperados

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `NEXT_PUBLIC_WS_URL` | `https://botderivapi.roilabs.com.br` | URL do backend (WebSocket e API REST) |
| `NEXT_PUBLIC_DEBUG` | `false` | (Opcional) Debug mode |

---

## 🔍 Explicação

O frontend Next.js precisa saber onde está o backend para:
- Conectar ao WebSocket (`/ws/abutre`)
- Fazer chamadas às APIs REST (`/api/abutre/*`)

O código usa essa variável aqui:
```typescript
// frontend/abutre-dashboard/src/lib/websocket-client.ts:262
const url = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000'
```

Se não configurar, vai tentar conectar em `localhost:8000` (que não existe em produção).

---

## ⚠️ IMPORTANTE

### NÃO use Secret

O erro diz:
```
references Secret "ws_url", which does not exist
```

Isso significa que você configurou como **Secret** no Easypanel.

**Correto**: Environment Variable (público)
**Errado**: Secret

Variáveis `NEXT_PUBLIC_*` do Next.js precisam ser **públicas** (não secretas) porque são embutidas no bundle do frontend.

---

## 🚀 Após Adicionar

**Log esperado no build**:
```
✓ Compiled successfully
✓ Linting and checking validity of types
✓ Generating static pages (6/6)

Route (app)
├ ○ /                    111 kB     209 kB
├ ○ /history             8.82 kB    90.8 kB  ← NOVA
└ ○ /settings            4.41 kB    102 kB
```

**Teste**:
1. Acessar: `https://botderiv.roilabs.com.br/abutre`
2. Abrir DevTools → Console
3. Não deve ter erro de conexão WebSocket
4. Dashboard deve carregar normalmente

---

## 📝 Checklist

- [ ] Ir em Easypanel → Frontend → Environment Variables
- [ ] Adicionar `NEXT_PUBLIC_WS_URL` = `https://botderivapi.roilabs.com.br`
- [ ] Tipo: Environment Variable (NÃO Secret)
- [ ] Salvar
- [ ] Force Rebuild
- [ ] Aguardar build (2-3 min)
- [ ] Acessar `https://botderiv.roilabs.com.br/abutre`
- [ ] Verificar que não tem erro 404 em `/history`

---

**🎯 AÇÃO IMEDIATA**: Adicionar variável de ambiente no Easypanel e rebuild!
