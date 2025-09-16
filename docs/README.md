# 📚 Documentação - Synth Bot Buddy

Esta pasta contém a documentação completa da API Deriv e guias para desenvolvedores do projeto Synth Bot Buddy.

## 📋 Índice de Documentação

### 🔗 API Deriv

| Arquivo | Descrição | Conteúdo |
|---------|-----------|----------|
| [`deriv-api-overview.md`](./deriv-api-overview.md) | **Visão geral completa da API Deriv** | Endpoints, autenticação, exemplos de código, best practices |
| [`deriv-api-buy-endpoint.md`](./deriv-api-buy-endpoint.md) | **Documentação detalhada do endpoint 'buy'** | Parâmetros, estruturas de request/response, códigos de erro |

## 🎯 Conteúdo por Categoria

### 🔐 Autenticação
- **OAuth 2.0** (Recomendado): Fluxo completo de autenticação segura
- **API Tokens** (Legacy): Método tradicional com tokens permanentes
- **Exemplos práticos** em JavaScript e Python

### 📊 Trading & Contratos
- **Tipos de contrato**: Rise/Fall, Touch/No Touch, Digits, etc.
- **Símbolos disponíveis**: Volatility Indices, Jump Indices, Boom/Crash
- **Gestão de risco**: Capital management, stop loss, take profit

### 🔧 Implementação Técnica
- **WebSocket connections**: Conexão em tempo real
- **Rate limiting**: Limites de requisições e melhores práticas
- **Error handling**: Tratamento de erros e códigos de resposta
- **Reconnection logic**: Reconexão automática e tolerância a falhas

### 📈 Exemplos de Código
- **Bot básico**: Implementação simples de trading automatizado
- **Capital management**: Gestão inteligente de capital
- **Request queue**: Fila de requisições respeitando rate limits
- **Real-time monitoring**: Monitoramento em tempo real

## 🚀 Quick Start

### 1. Autenticação OAuth (Recomendada)

```javascript
// Redirecionar para OAuth
const authUrl = 'https://oauth.deriv.com/oauth2/authorize?app_id=99188';
window.location.href = authUrl;

// Processar callback
const token = new URLSearchParams(window.location.search).get('token1');
```

### 2. Conexão WebSocket

```javascript
const ws = new WebSocket('wss://ws.binaryws.com/websockets/v3?app_id=99188');
ws.send(JSON.stringify({ authorize: token }));
```

### 3. Comprar Contrato

```javascript
ws.send(JSON.stringify({
  buy: 1,
  price: 10,
  parameters: {
    amount: 10,
    basis: "stake",
    contract_type: "CALL",
    currency: "USD",
    duration: 5,
    duration_unit: "m",
    symbol: "R_100"
  }
}));
```

## 📖 Como Usar Esta Documentação

### Para Desenvolvedores Iniciantes
1. Comece com [`deriv-api-overview.md`](./deriv-api-overview.md)
2. Estude os exemplos de código básicos
3. Implemente autenticação OAuth primeiro
4. Teste com contratos simples

### Para Desenvolvedores Experientes
1. Consulte [`deriv-api-buy-endpoint.md`](./deriv-api-buy-endpoint.md) para detalhes técnicos
2. Implemente gestão de capital avançada
3. Configure rate limiting e error handling
4. Otimize performance para high-frequency trading

### Para Debugging
1. Verifique códigos de erro na documentação
2. Use os exemplos de tratamento de erro
3. Implemente logging detalhado
4. Teste em ambiente demo primeiro

## 🔍 Informações Importantes

### ⚠️ Avisos de Segurança

- **Nunca exponha** tokens de API em código público
- **Use sempre HTTPS** para comunicação
- **Teste extensivamente** em contas demo antes de usar dinheiro real
- **Implemente stop loss** e gestão de risco adequada

### 📊 Limites da API

- **Rate limit**: 150 requests por minuto
- **Conexões simultâneas**: 5 por conta
- **Stake mínimo**: $0.35
- **Stake máximo**: $50,000

### 🎯 Symbols Mais Utilizados

| Símbolo | Nome | Volatilidade | Recomendado Para |
|---------|------|--------------|------------------|
| `R_100` | Volatility 100 | Alta | Trading ativo |
| `R_75` | Volatility 75 | Média-Alta | Estratégias médias |
| `R_50` | Volatility 50 | Média | Iniciantes |
| `R_25` | Volatility 25 | Baixa | Trading conservador |

## 🔗 Links Úteis

### 🌐 Recursos Oficiais Deriv
- [API Explorer](https://api.deriv.com/api-explorer/) - Playground interativo
- [WebSocket Playground](https://api.deriv.com/playground/) - Teste em tempo real
- [Developer Community](https://community.deriv.com/) - Comunidade de desenvolvedores
- [GitHub Examples](https://github.com/deriv-com/) - Exemplos oficiais

### 📚 Documentação do Projeto
- [README Principal](../README.md) - Visão geral do projeto
- [Frontend](../frontend/README.md) - Documentação do frontend
- [Backend](../backend/README.md) - Documentação do backend

## 🤝 Contribuição

Para contribuir com a documentação:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/nova-documentacao`)
3. **Commit** suas mudanças (`git commit -m 'docs: adicionar nova documentação'`)
4. **Push** para a branch (`git push origin feature/nova-documentacao`)
5. **Abra** um Pull Request

### 📝 Padrões de Documentação

- Use **markdown** para formatação
- Inclua **exemplos práticos** sempre que possível
- **Mantenha** a linguagem clara e acessível
- **Adicione** links para recursos relacionados
- **Teste** todos os exemplos de código

---

**Última Atualização:** Janeiro 2025
**Versão:** 1.0
**Mantido por:** Synth Bot Buddy Team

Para dúvidas ou sugestões, abra uma [issue](https://github.com/JeanZorzetti/synth-bot-buddy/issues) no GitHub.