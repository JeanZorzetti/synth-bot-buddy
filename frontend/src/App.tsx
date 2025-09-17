import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Dashboard from "./pages/Dashboard";
import Training from "./pages/Training";
import Trading from "./pages/Trading";
import Settings from "./pages/Settings";
import History from "./pages/History";
import Performance from "./pages/Performance";
import Auth from "./pages/Auth";
import AuthCallback from "./pages/AuthCallback";
import DerivTrading from "./pages/DerivTrading";
import RealTimeData from "./pages/RealTimeData";
import AIControlCenter from "./pages/AIControlCenter";
import MultiAssetManagement from "./pages/MultiAssetManagement";
import StrategyMarketplace from "./pages/StrategyMarketplace";
import EnterprisePlatform from "./pages/EnterprisePlatform";

const queryClient = new QueryClient();

// Enhanced auth check - check for both API key and OAuth session
const isAuthenticated = () => {
  const hasApiKey = localStorage.getItem('deriv_api_key') !== null;
  const hasOAuthToken = localStorage.getItem('deriv_primary_token') !== null;
  return hasApiKey || hasOAuthToken;
};

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  return isAuthenticated() ? <Layout>{children}</Layout> : <Navigate to="/auth" replace />;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/auth/callback" element={<AuthCallback />} />
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/training" element={
            <ProtectedRoute>
              <Training />
            </ProtectedRoute>
          } />
          <Route path="/trading" element={
            <ProtectedRoute>
              <Trading />
            </ProtectedRoute>
          } />
          <Route path="/settings" element={
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          } />
          <Route path="/history" element={
            <ProtectedRoute>
              <History />
            </ProtectedRoute>
          } />
          <Route path="/performance" element={
            <ProtectedRoute>
              <Performance />
            </ProtectedRoute>
          } />
          <Route path="/real-time-data" element={
            <ProtectedRoute>
              <RealTimeData />
            </ProtectedRoute>
          } />
          <Route path="/ai-control-center" element={
            <ProtectedRoute>
              <AIControlCenter />
            </ProtectedRoute>
          } />
          <Route path="/multi-asset-management" element={
            <ProtectedRoute>
              <MultiAssetManagement />
            </ProtectedRoute>
          } />
          <Route path="/strategy-marketplace" element={
            <ProtectedRoute>
              <StrategyMarketplace />
            </ProtectedRoute>
          } />
          <Route path="/enterprise-platform" element={
            <ProtectedRoute>
              <EnterprisePlatform />
            </ProtectedRoute>
          } />
          <Route path="/deriv-trading" element={<DerivTrading />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
