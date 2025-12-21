/**
 * ABUTRE DASHBOARD - Metrics Card Component
 *
 * Reusable card component for displaying key performance metrics
 * with icons, values, and change indicators
 */

interface MetricsCardProps {
  title: string
  value: string
  change: string
  changeType: 'positive' | 'negative' | 'neutral' | 'warning'
  icon: React.ReactNode
  iconColor: string
  iconBg: string
}

export default function MetricsCard({
  title,
  value,
  change,
  changeType,
  icon,
  iconColor,
  iconBg
}: MetricsCardProps) {
  const changeColors = {
    positive: 'text-emerald-400',
    negative: 'text-red-400',
    neutral: 'text-slate-400',
    warning: 'text-amber-400',
  }

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-5 hover:border-slate-600/50 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className={`w-10 h-10 rounded-lg ${iconBg} flex items-center justify-center ${iconColor}`}>
          {icon}
        </div>
      </div>
      <div>
        <p className="text-sm text-slate-400 mb-1">{title}</p>
        <p className="text-2xl font-bold mb-1">{value}</p>
        <p className={`text-sm ${changeColors[changeType]}`}>{change}</p>
      </div>
    </div>
  )
}
