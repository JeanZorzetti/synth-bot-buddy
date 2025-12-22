import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Dashboard from "./pages/Dashboard";
import Trading from "./pages/Trading";
import Settings from "./pages/Settings";
import Auth from "./pages/Auth";
import AuthCallback from "./pages/AuthCallback";
import TechnicalAnalysis from "./pages/TechnicalAnalysis";
import RiskManagement from "./pages/RiskManagement";
import OrderFlow from "./pages/OrderFlow";
import TradeHistory from "./pages/TradeHistory";
import BacktestingVisual from "./pages/BacktestingVisual";
import AlertsConfig from "./pages/AlertsConfig";
import PaperTrading from "./pages/PaperTrading";
import ForwardTesting from "./pages/ForwardTesting";
import AbutreDashboard from "./pages/AbutreDashboard";

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
          <Route path="/technical-analysis" element={
            <ProtectedRoute>
              <TechnicalAnalysis />
            </ProtectedRoute>
          } />
          <Route path="/risk-management" element={
            <ProtectedRoute>
              <RiskManagement />
            </ProtectedRoute>
          } />
          <Route path="/order-flow" element={
            <ProtectedRoute>
              <OrderFlow />
            </ProtectedRoute>
          } />
          <Route path="/trade-history" element={
            <ProtectedRoute>
              <TradeHistory />
            </ProtectedRoute>
          } />
          <Route path="/backtesting" element={
            <ProtectedRoute>
              <BacktestingVisual />
            </ProtectedRoute>
          } />
          <Route path="/alerts" element={
            <ProtectedRoute>
              <AlertsConfig />
            </ProtectedRoute>
          } />
          <Route path="/paper-trading" element={
            <ProtectedRoute>
              <PaperTrading />
            </ProtectedRoute>
          } />
          <Route path="/forward-testing" element={
            <ProtectedRoute>
              <ForwardTesting />
            </ProtectedRoute>
          } />
          <Route path="/abutre" element={
            <ProtectedRoute>
              <AbutreDashboard />
            </ProtectedRoute>
          } />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
