import React, { useEffect, useState } from 'react';
import { fetchResearchOverview } from '../../services/api';
import { Loader2, Award, AlertTriangle } from 'lucide-react';

export const ResearchOverview: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchResearchOverview()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading benchmark overview from saved reports...</span>
      </div>
    );
  }

  const table = data?.model_comparison_table || [];

  return (
    <div className="space-y-6 animate-fadeIn">
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Award className="text-amber-400" size={20} />
          Executive Benchmark Overview
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Headline test split performance metrics read directly at runtime from{' '}
          <code className="text-emerald-400 font-mono">reports/model_comparison.csv</code>.
        </p>
      </div>

      {/* Cross-Model Comparison Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
            <tr>
              <th className="py-3 px-4">Model ID</th>
              <th className="py-3 px-3">Macro F1</th>
              <th className="py-3 px-3">Macro Precision</th>
              <th className="py-3 px-3">Macro Recall</th>
              <th className="py-3 px-3">Micro F1</th>
              <th className="py-3 px-3">Macro PR-AUC</th>
              <th className="py-3 px-3">Mean ROC-AUC</th>
              <th className="py-3 px-3">Exact Match</th>
              <th className="py-3 px-3">Latency (ms)</th>
              <th className="py-3 px-3">Throughput (cps)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {table.map((row: any, idx: number) => {
              const isServing = row.model_id === '3_distilroberta';
              const isDetox = row.model_id === 'detoxify';

              return (
                <tr
                  key={idx}
                  className={`hover:bg-slate-800/40 transition-colors ${
                    isServing ? 'bg-emerald-500/10 font-semibold' : ''
                  }`}
                >
                  <td className="py-3 px-4 font-sans font-medium text-slate-200 flex items-center gap-2">
                    <span>{row.model_id}</span>
                    {isServing && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-mono">
                        SERVING DEFAULT
                      </span>
                    )}
                    {isDetox && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 font-mono">
                        EXTERNAL BASELINE
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">
                    {typeof row.overall_macro_f1 === 'number' ? row.overall_macro_f1.toFixed(4) : row.overall_macro_f1}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {typeof row.overall_macro_precision === 'number' ? row.overall_macro_precision.toFixed(4) : row.overall_macro_precision}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {typeof row.overall_macro_recall === 'number' ? row.overall_macro_recall.toFixed(4) : row.overall_macro_recall}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {typeof row.overall_micro_f1 === 'number' ? row.overall_micro_f1.toFixed(4) : row.overall_micro_f1}
                  </td>
                  <td className="py-3 px-3 text-cyan-400 font-bold">
                    {typeof row.overall_macro_pr_auc === 'number' ? row.overall_macro_pr_auc.toFixed(4) : row.overall_macro_pr_auc}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {typeof row.overall_mean_roc_auc === 'number' ? row.overall_mean_roc_auc.toFixed(4) : row.overall_mean_roc_auc}
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    {typeof row.exact_match_acc === 'number' ? (row.exact_match_acc * 100).toFixed(2) + '%' : row.exact_match_acc}
                  </td>
                  <td className="py-3 px-3 text-amber-300">
                    {typeof row.latency_ms === 'number' ? row.latency_ms.toFixed(2) : row.latency_ms}
                  </td>
                  <td className="py-3 px-3 text-teal-300">
                    {typeof row.throughput_cps === 'number' ? Number(row.throughput_cps).toLocaleString() : row.throughput_cps}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Detoxify Baseline Methodological Footnote */}
      <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-500/30 text-xs text-purple-200/90 space-y-1.5">
        <div className="flex items-center gap-1.5 font-semibold text-purple-300">
          <AlertTriangle size={15} />
          <span>Detoxify Baseline Methodological Disclosure</span>
        </div>
        <p className="text-[11px] leading-relaxed text-purple-300/80">
          {data?.detoxify_note}
        </p>
      </div>
    </div>
  );
};
