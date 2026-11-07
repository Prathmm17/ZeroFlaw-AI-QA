import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  variant?: 'blue' | 'emerald' | 'amber' | 'rose' | 'default';
  trend?: {
    value: string;
    isPositive: boolean;
  };
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  variant = 'default',
  trend,
}) => {
  const variantStyles = {
    default: {
      border: 'border-white/[0.1] hover:border-violet-500/40',
      iconBg: 'bg-violet-500/20 text-violet-200 border border-violet-400/30 shadow-md shadow-violet-500/20',
      glowOrb: 'from-violet-600/30 via-purple-600/15 to-transparent',
    },
    blue: {
      border: 'border-violet-500/30 hover:border-violet-400/50',
      iconBg: 'bg-gradient-to-br from-violet-600/30 to-purple-600/30 text-violet-100 border border-violet-400/40 shadow-violet-500/25 shadow-md',
      glowOrb: 'from-violet-500/35 via-purple-600/20 to-transparent',
    },
    emerald: {
      border: 'border-emerald-500/30 hover:border-emerald-400/50',
      iconBg: 'bg-emerald-500/20 text-emerald-200 border border-emerald-400/40 shadow-emerald-500/20 shadow-md',
      glowOrb: 'from-emerald-500/25 via-teal-600/15 to-transparent',
    },
    amber: {
      border: 'border-amber-500/30 hover:border-amber-400/50',
      iconBg: 'bg-amber-500/20 text-amber-200 border border-amber-400/40 shadow-amber-500/20 shadow-md',
      glowOrb: 'from-amber-500/25 via-orange-600/15 to-transparent',
    },
    rose: {
      border: 'border-rose-500/30 hover:border-rose-400/50',
      iconBg: 'bg-rose-500/20 text-rose-200 border border-rose-400/40 shadow-rose-500/20 shadow-md',
      glowOrb: 'from-rose-500/30 via-purple-600/15 to-transparent',
    },
  }[variant];

  return (
    <div
      className={`relative overflow-hidden rounded-3xl bg-[#0e0a1c]/65 backdrop-blur-3xl p-5 border shadow-[0_16px_40px_rgba(0,0,0,0.65),inset_0_1px_1px_rgba(255,255,255,0.12)] transition-all duration-300 hover:bg-[#140e28]/75 hover:shadow-[0_20px_50px_rgba(0,0,0,0.85),0_0_25px_rgba(139,92,246,0.25)] hover:-translate-y-0.5 ${variantStyles.border}`}
    >
      {/* Subtle Specular Ambient Violet Flare */}
      <div className={`absolute -right-8 -top-8 w-36 h-36 rounded-full bg-gradient-to-br ${variantStyles.glowOrb} blur-xl pointer-events-none`} />

      <div className="flex items-center justify-between relative z-10">
        <div className="space-y-1">
          <p className="text-[11px] font-semibold text-violet-200/70 uppercase tracking-wider">{title}</p>
          <p className="text-2xl font-bold font-mono tracking-tight text-white drop-shadow-sm">{value}</p>
        </div>
        <div className={`p-3 rounded-2xl backdrop-blur-xl ${variantStyles.iconBg}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>

      {(subtitle || trend) && (
        <div className="mt-3.5 pt-3 border-t border-white/[0.08] flex items-center justify-between text-xs relative z-10">
          {subtitle && <span className="text-slate-400 font-medium text-[11px]">{subtitle}</span>}
          {trend && (
            <span
              className={`font-mono font-semibold ${
                trend.isPositive ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {trend.isPositive ? '↑' : '↓'} {trend.value}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

