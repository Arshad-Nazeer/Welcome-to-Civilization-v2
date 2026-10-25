import React, { useEffect, useState } from 'react';
import { fetchResearchLimitations } from '../../services/api';
import { Loader2, AlertTriangle, Scissors, HelpCircle, Layers, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export const ResearchLimitations: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchResearchLimitations()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading model limitations and methodological disclosures...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <AlertTriangle className="text-amber-400" size={20} />
          Model Limitations, Truncation & External Baseline Disclosures
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Engineering boundaries, token truncation trade-offs, and critical methodological distinctions regarding Detoxify.
        </p>
      </div>

      {/* Detoxify Disclosure Banner */}
      <div className="p-5 rounded-xl bg-purple-950/20 border border-purple-500/30 text-xs space-y-2">
        <div className="flex items-center gap-2 font-bold text-purple-300 text-sm">
          <AlertTriangle size={16} />
          <span>Detoxify External Pretrained Baseline Footnote</span>
        </div>
        <p className="text-[12px] leading-relaxed text-purple-200/90">
          {data?.detoxify_footnote}
        </p>
      </div>

      {/* Key Limitations Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <Scissors size={14} className="text-amber-400" />
            1. 128-Token Head Truncation
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Input comments are truncated at 128 tokens for throughput and low memory footprint. 
            EDA confirmed that 18.5% of comments exceed 128 tokens, meaning toxic insults appearing late in very long rants may escape transformer attention.
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <Layers size={14} className="text-cyan-400" />
            2. Single-Seed Serving (Seed 42)
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            To prioritize predictable latency (&lt;0.5ms) and lean GPU RAM requirements, we deploy single-seed <code>3_distilroberta</code> (Seed 42) rather than an ensemble of 3 distinct transformer checkpoints.
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <HelpCircle size={14} className="text-rose-400" />
            3. Extreme Threat Sparsity
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            With only 50 positive threat examples in the test split (prevalence: 0.30%), metric confidence intervals for threat are naturally wider. 
            The policy safely caps threat at Orange ("Hide with warning") to prevent false automated removals.
          </p>
        </div>
      </div>

      {/* Full Extracted Markdown from Model Reports */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
        <div className="flex items-center gap-2">
          <FileText size={16} className="text-emerald-400" />
          <h4 className="text-sm font-semibold text-slate-200">
            Formal Limitations Section (From reports/3_distilroberta/model_report.md)
          </h4>
        </div>
        <div className="prose prose-invert prose-slate max-w-none text-xs leading-relaxed max-h-80 overflow-y-auto p-4 bg-slate-950/60 rounded-lg border border-slate-800">
          <ReactMarkdown>{data?.distilroberta_limitations_markdown || 'No markdown extracted.'}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
};
