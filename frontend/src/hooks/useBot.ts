import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService, BotStatus } from '@/services/api';
import { toast } from '@/components/ui/use-toast';

export interface UseBotReturn {
  // Status
  botStatus: BotStatus | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  isApiAvailable: boolean;
  
  // Actions
  startBot: () => void;
  stopBot: () => void;
  connectToApi: () => void;
  disconnectFromApi: () => void;
  buyContract: (params: { contract_type: string; amount: number; duration?: number; symbol?: string }) => void;
  
  // State flags
  isStarting: boolean;
  isStopping: boolean;
  isConnecting: boolean;
  isDisconnecting: boolean;
  isBuying: boolean;
}

export const useBot = (autoRefresh: boolean = true): UseBotReturn => {
  const queryClient = useQueryClient();
  const [isApiAvailable, setIsApiAvailable] = useState(false);

  // Query for bot status
  const {
    data: botStatus,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['botStatus'],
    queryFn: () => apiService.getBotStatus(),
    refetchInterval: autoRefresh ? 2000 : false, // Refresh every 2 seconds
    retry: (failureCount, error) => {
      // Don't retry if it's a connection error
      if (error.message.includes('Failed to fetch')) {
        return false;
      }
      return failureCount < 3;
    },
    onError: (error) => {
      console.error('Error fetching bot status:', error);
      setIsApiAvailable(false);
    },
    onSuccess: () => {
      setIsApiAvailable(true);
    }
  });

  // Check API availability on mount
  useEffect(() => {
    const checkApiAvailability = async () => {
      const available = await apiService.isAvailable();
      setIsApiAvailable(available);
      
      if (!available) {
        toast({
          title: "⚠️ Backend Desconectado",
          description: "Não foi possível conectar ao servidor backend. Verifique se está rodando na porta 8000.",
          variant: "destructive",
        });
      }
    };
    
    checkApiAvailability();
  }, []);

  // Mutations
  const connectMutation = useMutation({
    mutationFn: (apiToken: string) => apiService.connectToApi(apiToken),
    onSuccess: (response) => {
      toast({
        title: "🔗 Conexão",
        description: response.message || "Conectando à API Deriv...",
      });
      queryClient.invalidateQueries({ queryKey: ['botStatus'] });
    },
    onError: (error) => {
      toast({
        title: "❌ Erro de Conexão",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: () => apiService.disconnectFromApi(),
    onSuccess: (response) => {
      toast({
        title: "🔌 Desconexão",
        description: response.message || "Desconectado da API Deriv",
      });
      queryClient.invalidateQueries({ queryKey: ['botStatus'] });
    },
    onError: (error) => {
      toast({
        title: "❌ Erro ao Desconectar",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const startMutation = useMutation({
    mutationFn: () => apiService.startBot(),
    onSuccess: (response) => {
      toast({
        title: "🚀 Bot Iniciado",
        description: response.message || "Bot de trading iniciado com sucesso!",
      });
      queryClient.invalidateQueries({ queryKey: ['botStatus'] });
    },
    onError: (error) => {
      toast({
        title: "❌ Erro ao Iniciar Bot",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => apiService.stopBot(),
    onSuccess: (response) => {
      toast({
        title: "⏹️ Bot Parado",
        description: response.message || "Bot de trading parado",
      });
      queryClient.invalidateQueries({ queryKey: ['botStatus'] });
    },
    onError: (error) => {
      toast({
        title: "❌ Erro ao Parar Bot",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const buyMutation = useMutation({
    mutationFn: (params: { contract_type: string; amount: number; duration?: number; symbol?: string }) => 
      apiService.buyContract(params),
    onSuccess: (response) => {
      toast({
        title: "📈 Ordem Executada",
        description: response.message || "Ordem de compra enviada!",
      });
      queryClient.invalidateQueries({ queryKey: ['botStatus'] });
    },
    onError: (error) => {
      toast({
        title: "❌ Erro na Ordem",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Action callbacks
  const startBot = useCallback(async () => {
    // Ensure we have API token and connection before starting
    const apiToken = localStorage.getItem('deriv_api_key');
    if (!apiToken) {
      toast({
        title: "❌ Token não encontrado",
        description: "Faça login primeiro para obter o token da API.",
        variant: "destructive",
      });
      return;
    }

    try {
      // First ensure connection is established
      await connectMutation.mutateAsync(apiToken);
      // Then start the bot
      startMutation.mutate();
    } catch (error) {
      // Connection failed, don't try to start bot
      console.error('Failed to connect before starting bot:', error);
    }
  }, [startMutation, connectMutation]);

  const stopBot = useCallback(() => {
    stopMutation.mutate();
  }, [stopMutation]);

  const connectToApi = useCallback(() => {
    const apiToken = localStorage.getItem('deriv_api_key');
    if (!apiToken) {
      toast({
        title: "❌ Token não encontrado",
        description: "Faça login primeiro para obter o token da API.",
        variant: "destructive",
      });
      return;
    }
    connectMutation.mutate(apiToken);
  }, [connectMutation]);

  const disconnectFromApi = useCallback(() => {
    disconnectMutation.mutate();
  }, [disconnectMutation]);

  const buyContract = useCallback((params: { contract_type: string; amount: number; duration?: number; symbol?: string }) => {
    buyMutation.mutate(params);
  }, [buyMutation]);

  return {
    // Status
    botStatus,
    isLoading,
    isError,
    error: error as Error | null,
    isApiAvailable,
    
    // Actions
    startBot,
    stopBot,
    connectToApi,
    disconnectFromApi,
    buyContract,
    
    // State flags
    isStarting: startMutation.isPending,
    isStopping: stopMutation.isPending,
    isConnecting: connectMutation.isPending,
    isDisconnecting: disconnectMutation.isPending,
    isBuying: buyMutation.isPending,
  };
};