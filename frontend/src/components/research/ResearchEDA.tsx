import React, { useEffect, useState } from 'react';
import { fetchResearchEDA } from '../../services/api';
import { Loader2, Database, BarChart2, FileText, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export const ResearchEDA: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'stats' | 'figures' | 'conclusions'>('stats');
  const [selectedFigureCategory, setSelectedFigureCategory] = useState<string>('all');
  const [previewFigure, setPreviewFigure] = useState<string | null>(null);

  useEffect(() => {
    fetchResearchEDA()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading exploratory data analysis findings...</span>
      </div>
    );
  }

  const findings = data?.findings || {};
  const datasetSizes = findings?.dataset_sizes || {};
  const labelDist = findings?.a_label_distribution || {};
  const labelsPerComment = findings?.b_labels_per_comment || {};
  const figures = data?.figures || [];

  // Filter figures
  const filteredFigures = figures.filter((fig: any) => {
    if (selectedFigureCategory === 'all') return true;
    if (selectedFigureCategory === 'distribution') return fig.filename.startsWith('a_') || fig.filename.startsWith('b_') || fig.filename.startsWith('c_');
    if (selectedFigureCategory === 'length') return fig.filename.startsWith('d_') || fig.filename.startsWith('e_');
    if (selectedFigureCategory === 'ngrams') return fig.filename.startsWith('g_') || fig.filename.startsWith('h_') || fig.filename.startsWith('l_');
    if (selectedFigureCategory === 'validation') return fig.filename.startsWith('j_') || fig.filename.startsWith('k_') || fig.filename.startsWith('f_');
    return true;
  });

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Subnav */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <BarChart2 className="text-cyan-400" size={20} />
            Exploratory Data Analysis (EDA)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Statistical profiling, label distributions, multi-label co-occurrence, and corpus characteristics.
          </p>
        </div>

        <div className="flex bg-slate-900/90 border border-slate-800 rounded-lg p-1 text-xs">
          <button
            onClick={() => setActiveTab('stats')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'stats' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Summary & Metrics
          </button>
          <button
            onClick={() => setActiveTab('figures')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'figures' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Figure Gallery ({figures.length})
          </button>
          <button
            onClick={() => setActiveTab('conclusions')}
            className={`px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'conclusions' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Key Conclusions
          </button>
        </div>
      </div>

      {activeTab === 'stats' && (
        <div className="space-y-6">
          {/* Dataset Size Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
              <span className="text-[11px] font-mono uppercase text-slate-400">Total Processed</span>
              <p className="text-2xl font-bold font-mono text-slate-100 mt-1">
                {datasetSizes.total_processed?.toLocaleString() || '159,571'}
              </p>
              <span className="text-[10px] text-slate-500">Deduplicated Corpus</span>
            </div>
            <div className="p-4 rounded-xl border border-emerald-900/30 bg-emerald-950/10">
              <span className="text-[11px] font-mono uppercase text-emerald-400">Train Split (80%)</span>
              <p className="text-2xl font-bold font-mono text-emerald-300 mt-1">
                {datasetSizes.train?.toLocaleString() || '127,861'}
              </p>
              <span className="text-[10px] text-emerald-500/80">Group-stratified</span>
            </div>
            <div className="p-4 rounded-xl border border-cyan-900/30 bg-cyan-950/10">
              <span className="text-[11px] font-mono uppercase text-cyan-400">Val Split (10%)</span>
              <p className="text-2xl font-bold font-mono text-cyan-300 mt-1">
                {datasetSizes.val?.toLocaleString() || '15,869'}
              </p>
              <span className="text-[10px] text-cyan-500/80">Policy & threshold calibration</span>
            </div>
            <div className="p-4 rounded-xl border border-purple-900/30 bg-purple-950/10">
              <span className="text-[11px] font-mono uppercase text-purple-400">Test Split (10%)</span>
              <p className="text-2xl font-bold font-mono text-purple-300 mt-1">
                {datasetSizes.test?.toLocaleString() || '15,841'}
              </p>
              <span className="text-[10px] text-purple-500/80">Unseen test benchmark</span>
            </div>
          </div>

          {/* Label Distribution Table */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-slate-200">
                Training Split Label Distribution & Class Imbalance (N = {labelDist.total?.toLocaleString()})
              </h4>
              <span className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded font-mono">
                Clean Fraction: {(labelsPerComment.fraction_clean * 100)?.toFixed(2)}%
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Toxicity Category</th>
                    <th className="py-2.5 px-3">Positive Count</th>
                    <th className="py-2.5 px-3">Positive Rate (%)</th>
                    <th className="py-2.5 px-3">BCE pos_weight</th>
                    <th className="py-2.5 px-3">Imbalance Ratio</th>
                    <th className="py-2.5 px-3">Imbalance Severity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {Object.entries(labelDist.counts || {}).map(([lbl, count]: [string, any]) => {
                    const rate = labelDist.rates?.[lbl] ?? 0;
                    const weight = labelDist.pos_weight?.[lbl] ?? 0;
                    const ratio = labelDist.imbalance_ratio?.[lbl] ?? 0;

                    let severityBadge = (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
                        Moderate
                      </span>
                    );
                    if (ratio > 100) {
                      severityBadge = (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/30">
                          Extreme (≥100:1)
                        </span>
                      );
                    } else if (ratio > 20) {
                      severityBadge = (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          High (≥20:1)
                        </span>
                      );
                    }

                    return (
                      <tr key={lbl} className="hover:bg-slate-800/40">
                        <td className="py-2.5 px-3 font-sans font-medium text-slate-200 capitalize">
                          {lbl.replace('_', ' ')}
                        </td>
                        <td className="py-2.5 px-3 text-cyan-300 font-bold">{count.toLocaleString()}</td>
                        <td className="py-2.5 px-3 text-slate-300">{(rate * 100).toFixed(4)}%</td>
                        <td className="py-2.5 px-3 text-emerald-400 font-bold">{weight.toFixed(2)}</td>
                        <td className="py-2.5 px-3 text-slate-300">{ratio.toFixed(1)} : 1</td>
                        <td className="py-2.5 px-3 font-sans">{severityBadge}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Multi-label overlap notes */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Database size={14} className="text-cyan-400" />
                Multi-Label Overlap & Clean Majority
              </span>
              <p className="text-slate-400 leading-relaxed text-[11px]">
                Approximately <span className="text-cyan-300 font-semibold">89.86%</span> of comments contain no toxic labels whatsoever. 
                Of toxic comments, multi-label co-occurrence is heavily concentrated between <code className="text-slate-200 font-mono">toxic</code>, 
                <code className="text-slate-200 font-mono">obscene</code>, and <code className="text-slate-200 font-mono">insult</code>.
              </p>
            </div>
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <FileText size={14} className="text-amber-400" />
                Severe Toxic Hierarchy Constraint
              </span>
              <p className="text-slate-400 leading-relaxed text-[11px]">
                In the training set, <span className="text-amber-300 font-semibold">100.0%</span> of comments labeled <code className="text-slate-200 font-mono">severe_toxic</code> are also labeled <code className="text-slate-200 font-mono">toxic</code>. 
                Our serving policy enforces this hierarchical invariant: if severe_toxic triggers, toxic is automatically enforced.
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'figures' && (
        <div className="space-y-4">
          {/* Category Filter Pills */}
          <div className="flex flex-wrap gap-2 text-xs">
            {[
              { id: 'all', label: 'All Figures' },
              { id: 'distribution', label: 'Distributions & Co-occurrence' },
              { id: 'length', label: 'Length & Truncation' },
              { id: 'ngrams', label: 'N-grams & Term Odds' },
              { id: 'validation', label: 'Sanity & Features' },
            ].map((cat) => (
              <button
                key={cat.id}
                onClick={() => setSelectedFigureCategory(cat.id)}
                className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                  selectedFigureCategory === cat.id
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>

          {/* Grid of figures */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredFigures.map((fig: any, idx: number) => (
              <div
                key={idx}
                onClick={() => setPreviewFigure(fig.url)}
                className="group relative cursor-pointer overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 hover:border-cyan-500/40 transition-all hover:shadow-lg"
              >
                <div className="aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
                  <img
                    src={fig.url}
                    alt={fig.name}
                    className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>
                <div className="p-3 border-t border-slate-800/60 flex items-center justify-between">
                  <div>
                    <h5 className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300 transition-colors">
                      {fig.name}
                    </h5>
                    <span className="text-[10px] font-mono text-slate-500">{fig.filename}</span>
                  </div>
                  <ExternalLink size={14} className="text-slate-500 group-hover:text-cyan-400 transition-colors" />
                </div>
              </div>
            ))}
          </div>

          {/* Modal Preview */}
          {previewFigure && (
            <div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
              onClick={() => setPreviewFigure(null)}
            >
              <div
                className="relative max-w-5xl max-h-[90vh] bg-slate-900 border border-slate-800 rounded-xl p-4 overflow-hidden"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex justify-end pb-2">
                  <button
                    onClick={() => setPreviewFigure(null)}
                    className="text-xs font-mono text-slate-400 hover:text-white px-2 py-1 bg-slate-800 rounded"
                  >
                    Close [ESC]
                  </button>
                </div>
                <img src={previewFigure} alt="Preview" className="max-h-[80vh] w-auto mx-auto object-contain rounded" />
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'conclusions' && (
        <div className="p-6 rounded-xl border border-slate-800 bg-slate-900/60 prose prose-invert prose-slate max-w-none text-xs leading-relaxed">
          <ReactMarkdown>{data?.conclusions_markdown || 'No conclusions markdown found.'}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};
