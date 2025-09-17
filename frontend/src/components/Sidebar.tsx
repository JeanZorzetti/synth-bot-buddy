import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  LayoutDashboard,
  GraduationCap,
  TrendingUp,
  Settings,
  History,
  BarChart3,
  Brain,
  Activity,
  Shield,
  Database,
  Cpu,
  Globe,
  Store,
  Building
} from 'lucide-react';

interface SidebarProps {
  isCollapsed?: boolean;
}

interface NavItem {
  title: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  badge?: string;
  badgeVariant?: "default" | "secondary" | "destructive" | "outline";
}

const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    href: '/dashboard',
    icon: LayoutDashboard,
    description: 'Métricas em tempo real'
  },
  {
    title: 'Dados Tempo Real',
    href: '/real-time-data',
    icon: Database,
    description: 'Market Data & Features',
    badge: 'Phase 6',
    badgeVariant: 'secondary'
  },
  {
    title: 'AI Control Center',
    href: '/ai-control-center',
    icon: Cpu,
    description: 'AI Ensemble & Learning',
    badge: 'Phase 8',
    badgeVariant: 'secondary'
  },
  {
    title: 'Multi-Asset',
    href: '/multi-asset-management',
    icon: Globe,
    description: 'Portfolio Multi-Ativos',
    badge: 'Phase 9',
    badgeVariant: 'secondary'
  },
  {
    title: 'Marketplace',
    href: '/strategy-marketplace',
    icon: Store,
    description: 'Strategy Marketplace',
    badge: 'Phase 10',
    badgeVariant: 'secondary'
  },
  {
    title: 'Enterprise',
    href: '/enterprise-platform',
    icon: Building,
    description: 'Plataforma Empresarial',
    badge: 'Phase 10',
    badgeVariant: 'secondary'
  },
  {
    title: 'Treinamento',
    href: '/training',
    icon: GraduationCap,
    description: 'IA/ML Configuration'
  },
  {
    title: 'Trading',
    href: '/trading',
    icon: TrendingUp,
    description: 'Execução Autônoma',
    badge: 'AI',
    badgeVariant: 'default'
  },
  {
    title: 'Histórico',
    href: '/history',
    icon: History,
    description: 'Trades executados'
  },
  {
    title: 'Performance',
    href: '/performance',
    icon: BarChart3,
    description: 'Análise de resultados'
  },
  {
    title: 'Configurações',
    href: '/settings',
    icon: Settings,
    description: 'Sistema e API'
  }
];

const Sidebar: React.FC<SidebarProps> = ({ isCollapsed = false }) => {
  const location = useLocation();

  return (
    <div className={cn(
      "border-r bg-gray-50/40 transition-all duration-300",
      isCollapsed ? "w-16" : "w-64"
    )}>
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex h-16 items-center border-b px-4">
          {!isCollapsed ? (
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-blue-600" />
              <span className="font-bold text-lg">AI Trading Bot</span>
            </div>
          ) : (
            <div className="flex justify-center w-full">
              <Brain className="h-6 w-6 text-blue-600" />
            </div>
          )}
        </div>

        {/* Status Indicators */}
        {!isCollapsed && (
          <div className="px-4 py-3 border-b">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">SISTEMA</span>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-xs text-green-600">Online</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">IA STATUS</span>
                <div className="flex items-center space-x-1">
                  <Activity className="w-3 h-3 text-blue-500" />
                  <span className="text-xs text-blue-600">Ativo</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">RISK LEVEL</span>
                <div className="flex items-center space-x-1">
                  <Shield className="w-3 h-3 text-orange-500" />
                  <span className="text-xs text-orange-600">Baixo</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 px-4 py-4">
          <div className="space-y-2">
            {navItems.map((item) => {
              const isActive = location.pathname === item.href;
              const Icon = item.icon;

              return (
                <NavLink
                  key={item.href}
                  to={item.href}
                  className={cn(
                    "flex items-center rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    "hover:bg-accent hover:text-accent-foreground",
                    isActive
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground"
                  )}
                >
                  <Icon className={cn("h-4 w-4", isCollapsed ? "" : "mr-3")} />

                  {!isCollapsed && (
                    <>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span>{item.title}</span>
                          {item.badge && (
                            <Badge variant={item.badgeVariant || "secondary"} className="text-xs">
                              {item.badge}
                            </Badge>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {item.description}
                        </div>
                      </div>
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>
        </nav>

        {/* Footer */}
        {!isCollapsed && (
          <div className="border-t px-4 py-3">
            <div className="text-xs text-muted-foreground">
              <div className="font-medium mb-1">🤖 AI Trading Bot v2.1</div>
              <div>Autonomous Trading System</div>
              <div className="mt-2 text-center">
                <Badge variant="outline" className="text-xs">
                  Phases 6-10 Integration
                </Badge>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sidebar;