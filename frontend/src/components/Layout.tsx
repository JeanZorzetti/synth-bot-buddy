import { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { 
  BarChart3, 
  Settings, 
  History, 
  Power, 
  TrendingUp,
  Shield,
  Menu,
  X,
  Wifi,
  WifiOff,
  AlertTriangle
} from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useBot } from '@/hooks/useBot';
import { cn } from '@/lib/utils';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { logout } = useAuth();
  const { isApiAvailable } = useBot();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'Configurações', href: '/settings', icon: Settings },
    { name: 'Histórico', href: '/history', icon: History },
  ];

  const isActive = (href: string) => location.pathname === href;

  const getConnectionStatus = () => {
    if (isApiAvailable) {
      return {
        icon: Wifi,
        text: 'Backend Online',
        color: 'text-success'
      };
    } else {
      return {
        icon: WifiOff,
        text: 'Backend Offline',
        color: 'text-danger'
      };
    }
  };

  const connectionStatus = getConnectionStatus();
  const StatusIcon = connectionStatus.icon;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="flex items-center justify-between px-4 py-3 lg:px-6">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-6 w-6 text-primary" />
              <span className="text-xl font-semibold">BotDeriv</span>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => (
              <Link key={item.name} to={item.href}>
                <Button
                  variant={isActive(item.href) ? 'default' : 'ghost'}
                  size="sm"
                  className="flex items-center space-x-2"
                >
                  <item.icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Button>
              </Link>
            ))}
          </nav>

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </Button>

          {/* Status & Actions */}
          <div className="hidden md:flex items-center space-x-3">
            <div className="flex items-center space-x-2 text-sm">
              <StatusIcon className={`h-4 w-4 ${connectionStatus.color}`} />
              <span className={connectionStatus.color}>{connectionStatus.text}</span>
            </div>
            <Button variant="outline" size="sm" onClick={logout}>
              <Power className="h-4 w-4 mr-2" />
              Sair
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-muted/30">
            <nav className="flex flex-col p-4 space-y-2">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <Button
                    variant={isActive(item.href) ? 'default' : 'ghost'}
                    size="sm"
                    className="w-full justify-start"
                  >
                    <item.icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Button>
                </Link>
              ))}
              <div className="pt-2 mt-2 border-t border-border">
                <div className="flex items-center space-x-2 text-sm mb-3">
                  <StatusIcon className={`h-4 w-4 ${connectionStatus.color}`} />
                  <span className={connectionStatus.color}>{connectionStatus.text}</span>
                </div>
                <Button variant="outline" size="sm" className="w-full" onClick={logout}>
                  <Power className="h-4 w-4 mr-2" />
                  Sair
                </Button>
              </div>
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 lg:px-6">
        {children}
      </main>
    </div>
  );
}