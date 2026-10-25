import React from 'react';
import type { ModelPrediction } from '../types/api';
import { ActionBadge } from './ActionBadge';
import { SeverityMeter } from './SeverityMeter';
import { Cpu, Zap, Ban } from 'lucide-react';

interface OverviewComparisonProps {
  models: Record<string, ModelPrediction>;
  headlineModelId: string;
}

const LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];

export const OverviewComparison: React.FC<OverviewComparisonProps> = ({
  models,
  headlineModelId,
}) => {
  const modelKeys = ['1_tfidf_lr', '2b_bigru_attn_v2', '3_distilroberta', 'detoxify'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100">
            Multi-Model Comparative Decision Matrix
          </h3>
          <p className="text-xs text-slate-400">
            Side-by-side evaluation across all 4 architectures under their individual calibrated thresholds.
          </p>
        </div>
      </div>

      {/* Grid of Model Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {modelKeys.map((mId) => {
          const m = models[mId];
          const isHeadline = mId === headlineModelId;
          const isAvailable = m?.available ?? false;

          return (
            <div
              key={mId}
              className={`p-4 rounded-xl border flex flex-col justify-between transition-all ${
                isHeadline
                  ? 'bg-slate-900/90 border-emerald-500/50 shadow-md shadow-emerald-950/30'
                  : 'bg-slate-900/50 border-slate-800'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                    {mId.split('_')[0]}
                  </span>
                  {isHeadline && (
                    <span className="text-[10px] font-semibold text-emerald-400 bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 rounded-full">
                      Headline Model
                    </span>
                  )}
                  {mId === 'detoxify' && (
                    <span className="text-[10px] font-medium text-purple-400 bg-purple-500/15 border border-purple-500/30 px-1.5 py-0.5 rounded">
                      External Baseline
                    </span>
                  )}
                </div>

                <h4 className="font-semibold text-sm text-slate-200 line-clamp-1">
                  {m?.model_name || mId}
                </h4>
                <p className="text-xs text-slate-400 mb-3">{m?.role}</p>

                {isAvailable ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">Decision:</span>
                      <ActionBadge action={m.action} size="sm" />
                    </div>

                    <SeverityMeter severity={m.severity} size="sm" />

                    <div className="flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                      <span className="flex items-center gap-1">
                        <Cpu size={12} className="text-slate-500" />
                        {m.device}
                      </span>
                      <span className="flex items-center gap-1 font-mono text-slate-300">
                        <Zap size={12} className="text-amber-400" />
                        {m.execution_time_ms.toFixed(1)} ms
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="py-6 text-center text-xs text-slate-500 flex flex-col items-center gap-2">
                    <Ban size={20} className="text-slate-600" />
                    <span>Model Unavailable</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparative Probability Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/40">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Toxicity Category</th>
              {modelKeys.map((mId) => (
                <th key={mId} className="py-3 px-4">
                  {models[mId]?.model_name || mId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {LABELS.map((lbl) => (
              <tr key={lbl} className="hover:bg-slate-800/30 transition-colors">
                <td className="py-2.5 px-4 font-sans font-medium text-slate-300 capitalize">
                  {lbl.replace('_', ' ')}
                </td>
                {modelKeys.map((mId) => {
                  const m = models[mId];
                  if (!m?.available) {
                    return (
                      <td key={mId} className="py-2.5 px-4 text-slate-600">
                        —
                      </td>
                    );
                  }
                  const p = m.probabilities[lbl] ?? 0.0;
                  const n = m.normalized_scores[lbl] ?? 0.0;
                  const isFired = m.per_label_flags[lbl] ?? false;

                  return (
                    <td key={mId} className="py-2.5 px-4">
                      <div className="flex items-center gap-2">
                        <span className={isFired ? 'text-rose-400 font-bold' : 'text-slate-300'}>
                          {p.toFixed(3)}
                        </span>
                        <span className="text-slate-500 text-[10px]">
                          (n={n.toFixed(2)})
                        </span>
                        {isFired && (
                          <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
