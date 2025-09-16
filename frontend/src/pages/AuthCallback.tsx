import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, CheckCircle, AlertCircle, ArrowRight } from 'lucide-react';
import { handleDerivOAuthRedirect, apiService } from '../services/api';

export const AuthCallback: React.FC = () => {
  const navigate = useNavigate();
  const [status, setStatus] = useState<'processing' | 'success' | 'error'>('processing');
  const [message, setMessage] = useState('Processando autenticação OAuth...');
  const [sessionData, setSessionData] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');

  useEffect(() => {
    processOAuthCallback();
  }, []);

  const processOAuthCallback = async () => {
    try {
      setMessage('Processando callback OAuth da Deriv...');

      const result = await handleDerivOAuthRedirect();

      if (!result) {
        setStatus('error');
        setMessage('Nenhum parâmetro OAuth válido encontrado na URL');
        return;
      }

      setStatus('success');
      setMessage('Autenticação OAuth concluída com sucesso!');
      setSessionData(result);

      // Tentar conectar automaticamente com o token
      await connectWithOAuthToken(result.primary_token);

    } catch (error: any) {
      console.error('Erro no callback OAuth:', error);
      setStatus('error');
      setMessage(`Erro durante autenticação: ${error.message}`);
    }
  };

  const connectWithOAuthToken = async (token: string) => {
    try {
      setConnectionStatus('connecting');
      setMessage('Conectando com a Deriv API usando OAuth...');

      const connectionResult = await apiService.connectWithDerivOAuth(token, true);

      setConnectionStatus('connected');
      setMessage('Conectado com sucesso! Redirecionando para a página de trading...');

      // Redirecionar para trading após sucesso
      setTimeout(() => {
        navigate('/trading', { replace: true });
      }, 2000);

    } catch (error: any) {
      console.error('Erro ao conectar com OAuth:', error);
      setConnectionStatus('error');
      setMessage(`Erro ao conectar: ${error.message}`);
    }
  };

  const redirectToTrading = () => {
    navigate('/trading', { replace: true });
  };

  const redirectToAuth = () => {
    navigate('/auth', { replace: true });
  };

  const getStatusIcon = () => {
    if (connectionStatus === 'connecting' || status === 'processing') {
      return <Loader2 className="w-8 h-8 animate-spin text-blue-500" />;
    } else if (status === 'success' && connectionStatus === 'connected') {
      return <CheckCircle className="w-8 h-8 text-green-500" />;
    } else if (status === 'error' || connectionStatus === 'error') {
      return <AlertCircle className="w-8 h-8 text-red-500" />;
    } else {
      return <CheckCircle className="w-8 h-8 text-green-500" />;
    }
  };

  const getStatusColor = () => {
    if (connectionStatus === 'connecting' || status === 'processing') {
      return 'border-blue-200 bg-blue-50';
    } else if (status === 'success' && connectionStatus === 'connected') {
      return 'border-green-200 bg-green-50';
    } else if (status === 'error' || connectionStatus === 'error') {
      return 'border-red-200 bg-red-50';
    } else {
      return 'border-yellow-200 bg-yellow-50';
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        <div className={`bg-white rounded-lg shadow-lg border-2 ${getStatusColor()} p-8`}>
          <div className="text-center">
            <div className="flex justify-center mb-6">
              {getStatusIcon()}
            </div>

            <h1 className="text-2xl font-bold text-gray-900 mb-4">
              Autenticação OAuth
            </h1>

            <p className="text-gray-600 mb-6">
              {message}
            </p>

            {/* Dados da sessão */}
            {sessionData && (
              <div className="bg-gray-50 rounded-lg p-4 mb-6 text-left">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">
                  Dados da Sessão:
                </h3>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>
                    <strong>Conta Principal:</strong> {sessionData.primary_account}
                  </div>
                  <div>
                    <strong>Total de Contas:</strong> {sessionData.session_data?.accounts?.length || 0}
                  </div>
                  <div>
                    <strong>Tokens Recebidos:</strong> {sessionData.session_data?.tokens?.length || 0}
                  </div>
                  <div>
                    <strong>Moedas:</strong> {sessionData.session_data?.currencies?.join(', ') || 'N/A'}
                  </div>
                </div>
              </div>
            )}

            {/* Status da conexão */}
            {connectionStatus !== 'idle' && (
              <div className="bg-blue-50 rounded-lg p-4 mb-6">
                <div className="text-sm text-blue-800">
                  <strong>Status da Conexão:</strong>
                  <div className="mt-1">
                    {connectionStatus === 'connecting' && 'Conectando com Deriv API...'}
                    {connectionStatus === 'connected' && '✅ Conectado com sucesso!'}
                    {connectionStatus === 'error' && '❌ Erro na conexão'}
                  </div>
                </div>
              </div>
            )}

            {/* Botões de ação */}
            <div className="space-y-3">
              {status === 'success' && connectionStatus === 'connected' && (
                <button
                  onClick={redirectToTrading}
                  className="w-full flex items-center justify-center space-x-2 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 transition-colors"
                >
                  <span>Ir para Trading</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}

              {status === 'success' && connectionStatus !== 'connected' && connectionStatus !== 'connecting' && (
                <button
                  onClick={() => connectWithOAuthToken(sessionData.primary_token)}
                  className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <span>Conectar com Deriv API</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}

              {(status === 'error' || connectionStatus === 'error') && (
                <button
                  onClick={redirectToAuth}
                  className="w-full flex items-center justify-center space-x-2 bg-orange-600 text-white py-3 px-4 rounded-lg hover:bg-orange-700 transition-colors"
                >
                  <span>Tentar Novamente</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              )}

              <button
                onClick={redirectToTrading}
                className="w-full text-gray-600 hover:text-gray-800 text-sm"
              >
                Pular para Trading
              </button>
            </div>
          </div>
        </div>

        {/* Informações de depuração */}
        {import.meta.env.DEV && (
          <div className="mt-6 bg-gray-800 text-gray-300 rounded-lg p-4 text-xs">
            <h4 className="font-semibold mb-2">Debug Info:</h4>
            <div>URL: {window.location.href}</div>
            <div>Status: {status}</div>
            <div>Connection: {connectionStatus}</div>
            {sessionData && (
              <pre className="mt-2 text-xs overflow-auto">
                {JSON.stringify(sessionData, null, 2)}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AuthCallback;