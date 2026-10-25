import React, { useEffect, useState } from 'react';
import { fetchResearchComparisons } from '../../services/api';
import { Loader2, Scale, AlertTriangle, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

export const ResearchComparisons: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeComparison, setActiveComparison] = useState<'vs_2b' | 'vs_detox'>('vs_2b');

  useEffect(() => {
    fetchResearchComparisons()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading paired bootstrap significance tests...</span>
      </div>
    );
  }

  const table2b = data?.paired_bootstrap_vs_model2b || [];
  const tableDetox = data?.paired_bootstrap_vs_detoxify || [];

  const currentTable = activeComparison === 'vs_2b' ? table2b : tableDetox;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Scale className="text-cyan-400" size={20} />
            Statistical Significance Testing (Paired Bootstrap)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Non-parametric paired bootstrap resampling (B = 1,000 resamples) measuring whether performance deltas are statistically significant.
          </p>
        </div>

        <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
          <button
            onClick={() => setActiveComparison('vs_2b')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all ${
              activeComparison === 'vs_2b'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Model 3 vs Model 2b (In-Domain)
          </button>
          <button
            onClick={() => setActiveComparison('vs_detox')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all ${
              activeComparison === 'vs_detox'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Model 3 vs Detoxify (Pretrained Baseline)
          </button>
        </div>
      </div>

      {/* Comparison Context Banner */}
      {activeComparison === 'vs_2b' ? (
        <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-500/30 text-xs text-cyan-200 space-y-1">
          <span className="font-semibold text-cyan-300">In-Domain Head-to-Head: DistilRoBERTa vs BiGRU+Attention v2</span>
          <p className="text-[11px] text-cyan-300/80 leading-relaxed">
            Evaluates whether DistilRoBERTa significantly improves upon the best recurrent architecture trained on identical zero-leakage splits. 
            Positive delta (<span className="text-emerald-400 font-mono">+Δ</span>) indicates DistilRoBERTa superiority.
          </p>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-500/30 text-xs text-purple-200 space-y-1">
          <span className="font-semibold text-purple-300 flex items-center gap-1.5">
            <AlertTriangle size={14} />
            External Baseline Comparison: DistilRoBERTa vs Detoxify (Unitary AI)
          </span>
          <p className="text-[11px] text-purple-300/80 leading-relaxed">
            Detoxify was trained on raw 2018 competition data without our MinHash LSH deduplication. Negative deltas reflect Detoxify's exposure to near-duplicates across its training cycle, serving as an external upper-bound benchmark rather than a direct in-domain peer.
          </p>
        </div>
      )}

      {/* Significance Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-slate-200">
            Paired Bootstrap Hypothesis Results (B = 1000)
          </h4>
          <span className="text-xs text-slate-400 font-mono">
            p &lt; 0.05 (*) | p &lt; 0.01 (**) | ns = not significant
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Evaluation Scope</th>
                <th className="py-2.5 px-3">Metric</th>
                <th className="py-2.5 px-3">Mean Delta (M3 − Ref)</th>
                <th className="py-2.5 px-3">95% Bootstrap CI</th>
                <th className="py-2.5 px-3">Empirical p-value</th>
                <th className="py-2.5 px-3">Significance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {currentTable.map((row: any, idx: number) => {
                const delta = typeof row.mean_delta === 'number' ? row.mean_delta : parseFloat(row.mean_delta);
                const isSig = row.significance === '*' || row.significance === '**';
                const isPositive = delta > 0;

                return (
                  <tr key={idx} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-200">
                      {row.category}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{row.metric}</td>
                    <td className="py-2.5 px-3 font-bold">
                      <span
                        className={`inline-flex items-center gap-1 ${
                          isPositive ? 'text-emerald-400' : delta < 0 ? 'text-rose-400' : 'text-slate-400'
                        }`}
                      >
                        {isPositive ? (
                          <ArrowUpRight size={13} />
                        ) : delta < 0 ? (
                          <ArrowDownRight size={13} />
                        ) : (
                          <Minus size={13} />
                        )}
                        {delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      [{typeof row.ci_95_low === 'number' ? row.ci_95_low.toFixed(4) : row.ci_95_low},{' '}
                      {typeof row.ci_95_high === 'number' ? row.ci_95_high.toFixed(4) : row.ci_95_high}]
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {typeof row.p_value === 'number' ? row.p_value.toFixed(3) : row.p_value}
                    </td>
                    <td className="py-2.5 px-3 font-sans">
                      {isSig ? (
                        <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold">
                          Significant ({row.significance})
                        </span>
                      ) : (
                        <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                          ns (p ≥ 0.05)
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
