import React from 'react';
import { ShieldX, AlertTriangle, Eye, CheckCircle2, Ban } from 'lucide-react';

interface ActionBadgeProps {
  action: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

export const ActionBadge: React.FC<ActionBadgeProps> = ({
  action,
  size = 'md',
  showIcon = true,
}) => {
  const normAction = (action || '').toLowerCase();

  let colorClasses = 'bg-slate-800 text-slate-300 border-slate-700';
  let Icon = Eye;
  let label = action || 'Unknown';

  if (normAction.includes('remove') || normAction.includes('red')) {
    colorClasses = 'bg-rose-500/15 text-rose-400 border-rose-500/40 shadow-sm shadow-rose-950/40';
    Icon = ShieldX;
    label = 'Remove';
  } else if (normAction.includes('hide') || normAction.includes('orange')) {
    colorClasses = 'bg-amber-500/15 text-amber-400 border-amber-500/40 shadow-sm shadow-amber-950/40';
    Icon = AlertTriangle;
    label = 'Hide with warning';
  } else if (normAction.includes('flag') || normAction.includes('yellow')) {
    colorClasses = 'bg-yellow-500/15 text-yellow-400 border-yellow-500/40 shadow-sm shadow-yellow-950/40';
    Icon = Eye;
    label = 'Flag for review';
  } else if (normAction.includes('allow') || normAction.includes('green')) {
    colorClasses = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40 shadow-sm shadow-emerald-950/40';
    Icon = CheckCircle2;
    label = 'Allow';
  } else if (normAction.includes('unavail')) {
    colorClasses = 'bg-slate-800/60 text-slate-400 border-slate-700/60';
    Icon = Ban;
    label = 'Unavailable';
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-xs md:text-sm px-2.5 py-1 gap-1.5 font-medium',
    lg: 'text-sm md:text-base px-3.5 py-1.5 gap-2 font-semibold',
  }[size];

  const iconSizes = {
    sm: 13,
    md: 15,
    lg: 18,
  }[size];

  return (
    <span
      className={`inline-flex items-center rounded-full border transition-all ${colorClasses} ${sizeClasses}`}
    >
      {showIcon && <Icon size={iconSizes} className="shrink-0" />}
      <span>{label}</span>
    </span>
  );
};
