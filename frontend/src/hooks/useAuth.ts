import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '@/services/api';
import { useToast } from '@/components/ui/use-toast';

export interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  apiKey: string | null;
  isValidating: boolean;
}

export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    isLoading: true,
    apiKey: null,
    isValidating: false,
  });
  
  const navigate = useNavigate();
  const { toast } = useToast();

  // Check authentication status on mount
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const storedKey = localStorage.getItem('deriv_api_key');
        
        if (!storedKey) {
          setAuthState(prev => ({ ...prev, isLoading: false }));
          return;
        }

        // Validate stored key with backend
        const isBackendOnline = await apiService.isAvailable();
        
        if (!isBackendOnline) {
          // Backend is offline, assume valid for now
          setAuthState({
            isAuthenticated: true,
            isLoading: false,
            apiKey: storedKey,
            isValidating: false,
          });
          return;
        }

        // Try to connect with stored key to validate it
        try {
          await apiService.connectToApi(storedKey);
          setAuthState({
            isAuthenticated: true,
            isLoading: false,
            apiKey: storedKey,
            isValidating: false,
          });
        } catch (error) {
          console.error('Token validation failed:', error);
          // Keep token but note connection failed
          setAuthState({
            isAuthenticated: true,
            isLoading: false,
            apiKey: storedKey,
            isValidating: false,
          });
        }

      } catch (error) {
        console.error('Auth check failed:', error);
        // Clear invalid token
        localStorage.removeItem('deriv_api_key');
        setAuthState({
          isAuthenticated: false,
          isLoading: false,
          apiKey: null,
          isValidating: false,
        });
      }
    };

    checkAuthStatus();
  }, []);

  const login = async (apiKey: string): Promise<boolean> => {
    setAuthState(prev => ({ ...prev, isValidating: true }));

    try {
      // Basic validation - accept shorter tokens for development
      if (!apiKey.trim() || apiKey.length < 10) {
        throw new Error('Token da API inválido. Deve ter pelo menos 10 caracteres.');
      }

      // Check if backend is available
      const isBackendOnline = await apiService.isAvailable();
      
      if (!isBackendOnline) {
        toast({
          title: "⚠️ Backend Offline",
          description: "Salvando token localmente. Será validado quando o backend estiver online.",
          variant: "default",
        });
      }

      // Store the API key
      localStorage.setItem('deriv_api_key', apiKey);

      // Try to validate token with backend
      if (isBackendOnline) {
        try {
          const validation = await apiService.validateToken(apiKey);
          
          if (validation.valid) {
            toast({
              title: "✅ Login realizado!",
              description: "Token validado e conectado à Deriv API com sucesso.",
            });
          } else {
            toast({
              title: "⚠️ Token inválido",
              description: `Erro: ${validation.error || 'Token rejeitado pela API da Deriv'}. Verifique se o token está correto.`,
              variant: "destructive",
            });
            // Don't save invalid tokens
            setAuthState(prev => ({ ...prev, isValidating: false }));
            return false;
          }
        } catch (error: any) {
          console.error('Token validation failed:', error);
          toast({
            title: "⚠️ Não foi possível validar",
            description: "Token salvo localmente. Será validado quando possível.",
            variant: "default",
          });
        }
      } else {
        toast({
          title: "✅ Token salvo",
          description: "Token salvo localmente. Será validado quando backend estiver online.",
        });
      }

      // Update auth state
      setAuthState({
        isAuthenticated: true,
        isLoading: false,
        apiKey: apiKey,
        isValidating: false,
      });

      return true;

    } catch (error: any) {
      console.error('Login failed:', error);
      
      toast({
        title: "❌ Falha no login",
        description: error.message || "Não foi possível validar o token da API.",
        variant: "destructive",
      });

      setAuthState(prev => ({ ...prev, isValidating: false }));
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('deriv_api_key');
    setAuthState({
      isAuthenticated: false,
      isLoading: false,
      apiKey: null,
      isValidating: false,
    });
    
    toast({
      title: "Logout realizado",
      description: "Você foi desconectado com sucesso.",
    });
    
    navigate('/auth');
  };

  const validateWithBackend = async (): Promise<boolean> => {
    const apiKey = authState.apiKey;
    if (!apiKey) return false;

    try {
      // Try to get bot status - this requires a valid connection
      await apiService.getBotStatus();
      return true;
    } catch (error) {
      console.error('Backend validation failed:', error);
      return false;
    }
  };

  return {
    ...authState,
    login,
    logout,
    validateWithBackend,
  };
};