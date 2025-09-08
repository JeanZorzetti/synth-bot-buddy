# Configuração OAuth para Deriv

## Configuração Necessária no Gerenciador de Aplicativos da Deriv

Para que o OAuth funcione corretamente, você precisa configurar a URL de redirecionamento no Gerenciador de Aplicativos da Deriv:

### Passos:

1. **Acesse o Gerenciador de Aplicativos**
   - Vá para https://app.deriv.com
   - Navegue para "Settings" → "API Token" → "Manage Applications"

2. **Encontre sua aplicação**
   - Procure pela aplicação com App ID: `99188`
   - Clique no ícone de lápis para editar

3. **Configure a URL de Redirecionamento OAuth**
   - **Para produção:** `https://botderiv.roilabs.com.br/auth`
   - **Para desenvolvimento:** `http://localhost:5173/auth`

### URLs Atuais Configuradas:
- **Produção:** https://botderiv.roilabs.com.br/auth
- **Desenvolvimento:** http://localhost:5173/auth

### Como o OAuth Funciona:

1. **Clique em "Fazer Login com Deriv"**
   - Redireciona para: `https://oauth.deriv.com/oauth2/authorize?app_id=99188`

2. **Login na Deriv**
   - Usuário faz login com suas credenciais

3. **Autorização**
   - Usuário autoriza acesso às suas contas

4. **Redirecionamento de Retorno**
   - Deriv redireciona para: `https://botderiv.roilabs.com.br/auth?acct1=CR123&token1=abc123...`

5. **Processamento dos Tokens**
   - Frontend processa os parâmetros da URL
   - Usuário seleciona conta desejada
   - Sistema conecta usando o token da conta selecionada

### Parâmetros de Retorno Esperados:

```
?acct1=CR799393&token1=a1-f7pnteezo4jzhpxclctizt27hyeot&cur1=usd&
 acct2=VRTC1859315&token2=a1-clwe3vfuuus5kraceykdsoqm4snfq&cur2=usd&
 acct3=CRW1157&token3=a1-Yxh5gJS8m406Jopon5JlvKNRsxLMC&cur3=usd&
 acct4=VRW1160&token4=a1-yUqdjiIN0t6ICRc4eIMHDr1i6uKSV&cur4=usd
```

### Benefícios do OAuth:

- ✅ **Maior Segurança**: Tokens de sessão temporários
- ✅ **Múltiplas Contas**: Suporte a contas demo e reais
- ✅ **Processo Oficial**: Recomendado pela Deriv
- ✅ **Não Exposição**: Usuário não vê tokens permanentes