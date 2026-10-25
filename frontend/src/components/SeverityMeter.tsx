import React from 'react';
import type { SeverityInfo } from '../types/api';

interface SeverityMeterProps {
  severity: SeverityInfo;
  size?: 'sm' | 'md' | 'lg';
}

export const SeverityMeter: React.FC<SeverityMeterProps> = ({
  severity,
}) => {
  const score = severity?.score ?? 0.0;
  const band = severity?.band ?? 'Low';
  const percentage = Math.min(Math.max((score / 10.0) * 100, 0), 100);

  let barGradient = 'from-emerald-500 to-emerald-400';
  let badgeColor = 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';

  if (band === 'Medium') {
    barGradient = 'from-yellow-500 to-amber-400';
    badgeColor = 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30';
  } else if (band === 'High') {
    barGradient = 'from-orange-500 to-rose-400';
    badgeColor = 'bg-orange-500/15 text-orange-400 border-orange-500/30';
  } else if (band === 'Critical') {
    barGradient = 'from-rose-600 to-red-500';
    badgeColor = 'bg-rose-500/20 text-rose-300 border-rose-500/40 animate-pulse';
  }

  return (
    <div className="flex flex-col gap-1.5 w-full">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium">Severity Index</span>
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-slate-200">
            {score.toFixed(1)} <span className="text-slate-500 text-[10px]">/ 10</span>
          </span>
          <span className={`px-2 py-0.5 rounded-full border text-[11px] font-medium ${badgeColor}`}>
            {band}
          </span>
        </div>
      </div>

      {/* Progress Track */}
      <div className="relative w-full bg-slate-900/80 rounded-full h-2 overflow-hidden border border-slate-800">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barGradient} transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
