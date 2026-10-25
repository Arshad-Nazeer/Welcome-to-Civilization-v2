import React, { useEffect, useState } from 'react';
import { fetchResearchResults } from '../../services/api';
import { Loader2, TrendingUp, BarChart3, Image as ImageIcon, ExternalLink } from 'lucide-react';

export const ResearchResults: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('3_distilroberta');
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  useEffect(() => {
    fetchResearchResults()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading test benchmark results and bootstrap intervals...</span>
      </div>
    );
  }

  const perLabelRows = (data?.per_label_metrics || []).filter(
    (r: any) => r.model_id === selectedModel
  );

  const modelMetrics = data?.detailed_metrics?.[selectedModel]?.test_evaluation || {};
  const overall = modelMetrics.overall || {};
  const overallCi = modelMetrics.overall_ci || {};
  const perLabelDetailed = modelMetrics.per_label || {};
  const modelVis = (data?.visualizations?.[selectedModel] || []);

  const models = [
    { id: '3_distilroberta', name: '3_distilroberta (Serving)' },
    { id: '2b_bigru_attn_v2', name: '2b_bigru_attn_v2' },
    { id: '1_tfidf_lr', name: '1_tfidf_lr' },
    { id: 'detoxify', name: 'detoxify (External)' },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header and Model Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <TrendingUp className="text-emerald-400" size={20} />
            Test Set Performance & Bootstrap Confidence Intervals
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluated strictly on held-out test split (N = 15,841) with 95% bootstrap confidence bounds (B = 1000).
          </p>
        </div>

        <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
          {models.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                selectedModel === m.id
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>

      {/* Confidence Interval Top Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-4 rounded-xl border border-emerald-900/40 bg-emerald-950/10">
          <span className="text-[11px] font-mono uppercase text-emerald-400">Macro F1 Score</span>
          <p className="text-2xl font-bold font-mono text-emerald-300 mt-1">
            {typeof overall.macro_f1 === 'number' ? overall.macro_f1.toFixed(4) : '--'}
          </p>
          <span className="text-[10px] text-emerald-500/90 font-mono">
            {overallCi.macro_f1_ci
              ? `95% CI: [${overallCi.macro_f1_ci[0].toFixed(4)}, ${overallCi.macro_f1_ci[1].toFixed(4)}]`
              : 'Bootstrap 95% CI'}
          </span>
        </div>

        <div className="p-4 rounded-xl border border-cyan-900/40 bg-cyan-950/10">
          <span className="text-[11px] font-mono uppercase text-cyan-400">Macro PR-AUC</span>
          <p className="text-2xl font-bold font-mono text-cyan-300 mt-1">
            {typeof overall.macro_pr_auc === 'number' ? overall.macro_pr_auc.toFixed(4) : '--'}
          </p>
          <span className="text-[10px] text-cyan-500/90 font-mono">
            {overallCi.macro_pr_auc_ci
              ? `95% CI: [${overallCi.macro_pr_auc_ci[0].toFixed(4)}, ${overallCi.macro_pr_auc_ci[1].toFixed(4)}]`
              : 'Bootstrap 95% CI'}
          </span>
        </div>

        <div className="p-4 rounded-xl border border-blue-900/40 bg-blue-950/10">
          <span className="text-[11px] font-mono uppercase text-blue-400">Mean ROC-AUC</span>
          <p className="text-2xl font-bold font-mono text-blue-300 mt-1">
            {typeof overall.mean_roc_auc === 'number' ? overall.mean_roc_auc.toFixed(4) : '--'}
          </p>
          <span className="text-[10px] text-blue-500/90 font-mono">
            {overallCi.mean_roc_auc_ci
              ? `95% CI: [${overallCi.mean_roc_auc_ci[0].toFixed(4)}, ${overallCi.mean_roc_auc_ci[1].toFixed(4)}]`
              : 'Bootstrap 95% CI'}
          </span>
        </div>

        <div className="p-4 rounded-xl border border-purple-900/40 bg-purple-950/10">
          <span className="text-[11px] font-mono uppercase text-purple-400">Exact Match Accuracy</span>
          <p className="text-2xl font-bold font-mono text-purple-300 mt-1">
            {typeof overall.exact_match_ratio === 'number'
              ? `${(overall.exact_match_ratio * 100).toFixed(2)}%`
              : '--'}
          </p>
          <span className="text-[10px] text-purple-500/90 font-mono">
            {overallCi.exact_match_accuracy_ci
              ? `95% CI: [${(overallCi.exact_match_accuracy_ci[0] * 100).toFixed(2)}%, ${(overallCi.exact_match_accuracy_ci[1] * 100).toFixed(2)}%]`
              : 'Subset Exact Parity'}
          </span>
        </div>
      </div>

      {/* Per-Label Detailed Breakdown Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <BarChart3 size={16} className="text-emerald-400" />
            Per-Label Metrics on Test Split ({selectedModel})
          </h4>
          <span className="text-xs text-slate-400 font-mono">
            Source: reports/model_comparison_per_label.csv
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Toxicity Label</th>
                <th className="py-2.5 px-3">Support</th>
                <th className="py-2.5 px-3">Decision Threshold</th>
                <th className="py-2.5 px-3">F1 Score [95% CI]</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">Recall</th>
                <th className="py-2.5 px-3">PR-AUC</th>
                <th className="py-2.5 px-3">ROC-AUC</th>
                <th className="py-2.5 px-3">TP / FP / FN</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {perLabelRows.map((row: any) => {
                const detailed = perLabelDetailed[row.label] || {};
                const f1Ci = detailed.f1_ci;

                return (
                  <tr key={row.label} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-200 capitalize">
                      {row.label.replace('_', ' ')}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{row.support}</td>
                    <td className="py-2.5 px-3 text-amber-300 font-bold">
                      {typeof row.threshold === 'number' ? row.threshold.toFixed(2) : row.threshold}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      <span>{typeof row.f1 === 'number' ? row.f1.toFixed(4) : row.f1}</span>
                      {f1Ci && (
                        <span className="block text-[10px] text-emerald-500/80 font-normal">
                          [{f1Ci[0].toFixed(3)}, {f1Ci[1].toFixed(3)}]
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {typeof row.precision === 'number' ? row.precision.toFixed(4) : row.precision}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {typeof row.recall === 'number' ? row.recall.toFixed(4) : row.recall}
                    </td>
                    <td className="py-2.5 px-3 text-cyan-400 font-bold">
                      {typeof row.pr_auc === 'number' ? row.pr_auc.toFixed(4) : row.pr_auc}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {typeof row.roc_auc === 'number' ? row.roc_auc.toFixed(4) : row.roc_auc}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px]">
                      <span className="text-emerald-400">{row.tp}</span> /{' '}
                      <span className="text-red-400">{row.fp}</span> /{' '}
                      <span className="text-amber-400">{row.fn}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visualizations Gallery for Model */}
      {modelVis.length > 0 && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <ImageIcon size={16} className="text-cyan-400" />
              Diagnostic Curves & Confusion Matrices ({modelVis.length} plots)
            </h4>
            <span className="text-xs text-slate-500 font-mono">
              Static path: visualizations/{selectedModel}/
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {modelVis.map((img: any, idx: number) => (
              <div
                key={idx}
                onClick={() => setPreviewImage(img.url)}
                className="group cursor-pointer overflow-hidden rounded-xl border border-slate-800 bg-slate-950 hover:border-emerald-500/40 transition-all hover:shadow-lg"
              >
                <div className="aspect-video flex items-center justify-center p-2 bg-slate-950">
                  <img
                    src={img.url}
                    alt={img.title}
                    className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>
                <div className="p-3 border-t border-slate-800/80 flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200 group-hover:text-emerald-300 transition-colors truncate">
                    {img.title}
                  </span>
                  <ExternalLink size={14} className="text-slate-500 group-hover:text-emerald-400 transition-colors shrink-0" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Image Modal */}
      {previewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div
            className="relative max-w-5xl max-h-[90vh] bg-slate-900 border border-slate-800 rounded-xl p-4 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-end pb-2">
              <button
                onClick={() => setPreviewImage(null)}
                className="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded"
              >
                Close [ESC]
              </button>
            </div>
            <img src={previewImage} alt="Diagnostic Curve" className="max-h-[80vh] w-auto mx-auto object-contain rounded" />
          </div>
        </div>
      )}
    </div>
  );
};
