import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

interface ProbabilityBarProps {
  label: string;
  probability: number;
  threshold: number;
  normalizedScore: number;
  isFired: boolean;
}

const LABEL_METADATA: Record<string, { tier: 'critical' | 'moderate' | 'base'; desc: string }> = {
  threat: { tier: 'critical', desc: 'Direct threats of physical harm or violence' },
  identity_hate: { tier: 'critical', desc: 'Attacks targeting race, religion, gender, sexual orientation' },
  severe_toxic: { tier: 'critical', desc: 'Extremely aggressive, abusive harassment' },
  insult: { tier: 'moderate', desc: 'Disparaging, demeaning, or inflammatory remarks' },
  obscene: { tier: 'moderate', desc: 'Vulgar, sexually explicit, or profane terminology' },
  toxic: { tier: 'base', desc: 'General rude, aggressive, or uncivil rhetoric' },
};

export const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  label,
  probability,
  threshold,
  normalizedScore,
  isFired,
}) => {
  const meta = LABEL_METADATA[label] || { tier: 'base', desc: label };
  const pClamped = Math.min(Math.max(probability, 0.0), 1.0);
  const pPercent = pClamped * 100;
  const tPercent = Math.min(Math.max(threshold, 0.0), 1.0) * 100;

  // Determine bar fill color
  let barColor = 'bg-slate-600';
  if (isFired) {
    if (meta.tier === 'critical') barColor = 'bg-rose-500 shadow-sm shadow-rose-500/50';
    else if (meta.tier === 'moderate') barColor = 'bg-amber-500 shadow-sm shadow-amber-500/50';
    else barColor = 'bg-yellow-500 shadow-sm shadow-yellow-500/50';
  } else if (normalizedScore >= 0.30 && meta.tier === 'critical') {
    // Watch zone
    barColor = 'bg-yellow-500/80';
  }

  const tierColors = {
    critical: 'text-rose-400 bg-rose-500/10 border-rose-500/25',
    moderate: 'text-amber-400 bg-amber-500/10 border-amber-500/25',
    base: 'text-blue-400 bg-blue-500/10 border-blue-500/25',
  }[meta.tier];

  return (
    <div className="flex flex-col gap-1.5 p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className="font-mono font-medium text-slate-200 capitalize">
            {label.replace('_', ' ')}
          </span>
          <span className={`px-1.5 py-0.2 rounded border text-[10px] uppercase font-semibold tracking-wider ${tierColors}`}>
            {meta.tier}
          </span>
        </div>

        <div className="flex items-center gap-2.5">
          <span className="font-mono text-slate-400 text-[11px]">
            p = <span className="font-semibold text-slate-200">{probability.toFixed(3)}</span>
          </span>
          <span className="font-mono text-slate-500 text-[11px]">
            (n = {normalizedScore.toFixed(2)})
          </span>
          {isFired ? (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-rose-400 bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 rounded-full">
              <AlertCircle size={12} /> Fired
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500 bg-slate-800/80 border border-slate-700/60 px-2 py-0.5 rounded-full">
              <CheckCircle size={12} className="text-emerald-500/70" /> Clear
            </span>
          )}
        </div>
      </div>

      {/* Progress Track with Calibrated Threshold Line */}
      <div className="relative w-full h-3 bg-slate-950 rounded-full overflow-visible border border-slate-800/80 mt-1">
        {/* Filled Probability Bar */}
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out ${barColor}`}
          style={{ width: `${pPercent}%` }}
        />

        {/* Vertical Threshold Marker Line */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-slate-300 z-10 shadow-sm pointer-events-none"
          style={{ left: `${tPercent}%` }}
          title={`Calibrated decision threshold: ${threshold.toFixed(2)}`}
        >
          <div className="absolute -top-3.5 -translate-x-1/2 text-[9px] font-mono text-slate-400 whitespace-nowrap bg-slate-900/90 px-1 rounded border border-slate-700">
            t={threshold.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
};
