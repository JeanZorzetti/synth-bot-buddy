import React, { useState, useEffect } from 'react';
import { ExternalLink, Shield, User, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { apiService, handleDerivOAuthRedirect } from '../services/api';

interface DerivOAuthProps {
  onAuthSuccess?: (tokenData: any) => void;
  onAuthError?: (error: string) => void;
}

export const DerivOAuth: React.FC<DerivOAuthProps> = ({ onAuthSuccess, onAuthError }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [authStatus, setAuthStatus] = useState<'idle' | 'connecting' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [sessionData, setSessionData] = useState<any>(null);

  useEffect(() => {
    // Verificar se há dados OAuth armazenados
    const storedData = apiService.getStoredDerivOAuthData();
    if (storedData.primaryToken && apiService.hasValidDerivOAuthSession()) {
      setSessionData(storedData);
      setAuthStatus('success');
      setStatusMessage('Sessão OAuth ativa encontrada');
    }

    // Verificar se estamos processando um callback OAuth
    checkForOAuthCallback();
  }, []);

  const checkForOAuthCallback = async () => {
    try {
      const result = await handleDerivOAuthRedirect();
      if (result) {
        setSessionData(result.session_data);
        setAuthStatus('success');
        setStatusMessage('Autenticação OAuth concluída com sucesso!');
        onAuthSuccess?.(result);
      }
    } catch (error: any) {
      setAuthStatus('error');
      setStatusMessage(`Erro no callback OAuth: ${error.message}`);
      onAuthError?.(error.message);
    }
  };

  const startOAuthFlow = async () => {
    setIsLoading(true);
    setAuthStatus('connecting');
    setStatusMessage('Iniciando fluxo OAuth...');

    try {
      const authUrl = apiService.getDerivOAuthUrl({
        appId: '99188',
        utmCampaign: 'synth_bot_buddy'
      });

      setStatusMessage('Redirecionando para Deriv...');

      // Redirecionar para a URL de autorização
      window.location.href = authUrl;
    } catch (error: any) {
      setAuthStatus('error');
      setStatusMessage(`Erro ao iniciar OAuth: ${error.message}`);
      onAuthError?.(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const connectWithStoredToken = async () => {
    const { primaryToken } = apiService.getStoredDerivOAuthData();
    if (!primaryToken) {
      setAuthStatus('error');
      setStatusMessage('Nenhum token OAuth encontrado');
      return;
    }

    setIsLoading(true);
    setAuthStatus('connecting');
    setStatusMessage('Conectando com Deriv API...');

    try {
      const result = await apiService.connectWithDerivOAuth(primaryToken, true);
      setAuthStatus('success');
      setStatusMessage('Conectado com sucesso à Deriv API!');
      onAuthSuccess?.(result);
    } catch (error: any) {
      setAuthStatus('error');
      setStatusMessage(`Erro ao conectar: ${error.message}`);
      onAuthError?.(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const clearSession = () => {
    apiService.clearDerivOAuthSession();
    setSessionData(null);
    setAuthStatus('idle');
    setStatusMessage('');
  };

  const getStatusIcon = () => {
    switch (authStatus) {
      case 'connecting':
        return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Shield className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (authStatus) {
      case 'connecting':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2 bg-orange-100 rounded-lg">
          <Shield className="w-6 h-6 text-orange-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">
            Autenticação OAuth Deriv
          </h3>
          <p className="text-sm text-gray-600">
            Conecte-se de forma segura usando OAuth 2.0
          </p>
        </div>
      </div>

      {/* Status */}
      {statusMessage && (
        <div className={`flex items-center space-x-3 p-3 rounded-lg border mb-4 ${getStatusColor()}`}>
          {getStatusIcon()}
          <span className="text-sm font-medium">{statusMessage}</span>
        </div>
      )}

      {/* Sessão ativa */}
      {sessionData && authStatus === 'success' && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <User className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-800">Sessão OAuth Ativa</span>
          </div>
          <div className="text-xs text-green-700 space-y-1">
            <div>Conta Principal: {sessionData.primaryAccount || 'N/A'}</div>
            <div>Contas: {sessionData.sessionData?.accounts?.length || 0}</div>
            <div>Tokens: {sessionData.sessionData?.tokens?.length || 0}</div>
          </div>
          <div className="flex space-x-2 mt-3">
            <button
              onClick={connectWithStoredToken}
              disabled={isLoading}
              className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 disabled:opacity-50"
            >
              Conectar API
            </button>
            <button
              onClick={clearSession}
              className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-700"
            >
              Limpar Sessão
            </button>
          </div>
        </div>
      )}

      {/* Botão para iniciar OAuth */}
      {(!sessionData || authStatus !== 'success') && (
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="text-sm font-medium text-blue-800 mb-2">
              Como funciona a autenticação OAuth:
            </h4>
            <ul className="text-xs text-blue-700 space-y-1">
              <li>• Você será redirecionado para o site oficial da Deriv</li>
              <li>• Faça login ou cadastre-se na sua conta Deriv</li>
              <li>• Autorize o acesso para o Synth Bot Buddy</li>
              <li>• Você será redirecionado de volta com tokens seguros</li>
            </ul>
          </div>

          <button
            onClick={startOAuthFlow}
            disabled={isLoading}
            className="w-full flex items-center justify-center space-x-3 bg-orange-600 text-white py-3 px-4 rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <ExternalLink className="w-5 h-5" />
            )}
            <span>
              {isLoading ? 'Processando...' : 'Iniciar Autenticação OAuth'}
            </span>
          </button>
        </div>
      )}

      {/* Informações de segurança */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-start space-x-2">
          <Shield className="w-4 h-4 text-gray-400 mt-0.5" />
          <div className="text-xs text-gray-600">
            <strong>Segurança:</strong> OAuth 2.0 é o padrão da indústria para autenticação segura.
            Seus dados de login nunca passam pelo nosso servidor - a autenticação é feita diretamente
            com a Deriv.
          </div>
        </div>
      </div>
    </div>
  );
};

export default DerivOAuth;