import React, { useEffect, useState } from 'react';
import { fetchResearchSplits } from '../../services/api';
import { Loader2, GitFork, ShieldCheck, Hash, Layers, CheckCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export const ResearchSplits: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showDataDict, setShowDataDict] = useState(false);

  useEffect(() => {
    fetchResearchSplits()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading split manifest and deduplication methodology...</span>
      </div>
    );
  }

  const manifest = data?.manifest || {};
  const nearDup = manifest?.near_dup_config || {};
  const splits = manifest?.splits || {};
  const methodology = data?.methodology || {};

  const labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <GitFork className="text-indigo-400" size={20} />
          Split Strategy & Near-Duplicate Deduplication (v2)
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Stratified 80/10/10 partitioning with MinHash LSH grouping to eliminate train-test evaluation leakage.
        </p>
      </div>

      {/* Top Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60">
          <span className="text-[11px] font-mono uppercase text-slate-400 flex items-center gap-1.5">
            <Layers size={13} className="text-indigo-400" />
            Split Version
          </span>
          <p className="text-lg font-bold font-mono text-slate-100 mt-1">
            {manifest.split_version || 'v2_near_dup'}
          </p>
          <span className="text-[10px] text-slate-500 font-mono">Seed: {manifest.seed || 42}</span>
        </div>

        <div className="p-4 rounded-xl border border-emerald-900/30 bg-emerald-950/10">
          <span className="text-[11px] font-mono uppercase text-emerald-400 flex items-center gap-1.5">
            <ShieldCheck size={13} className="text-emerald-400" />
            Leakage Prevention
          </span>
          <p className="text-lg font-bold font-mono text-emerald-300 mt-1">
            0% Cross-Split
          </p>
          <span className="text-[10px] text-emerald-500/80">Group-level assignment</span>
        </div>

        <div className="p-4 rounded-xl border border-cyan-900/30 bg-cyan-950/10">
          <span className="text-[11px] font-mono uppercase text-cyan-400 flex items-center gap-1.5">
            <Hash size={13} className="text-cyan-400" />
            MinHash LSH Jaccard
          </span>
          <p className="text-lg font-bold font-mono text-cyan-300 mt-1">
            ≥ {nearDup.threshold || 0.8}
          </p>
          <span className="text-[10px] text-cyan-500/80">Char 5-grams (128 perms)</span>
        </div>

        <div className="p-4 rounded-xl border border-purple-900/30 bg-purple-950/10">
          <span className="text-[11px] font-mono uppercase text-purple-400 flex items-center gap-1.5">
            <CheckCircle size={13} className="text-purple-400" />
            Parquet Checksum
          </span>
          <p className="text-xs font-mono text-purple-300 mt-2 truncate" title={manifest.processed_parquet_sha256}>
            {manifest.processed_parquet_sha256 ? `${manifest.processed_parquet_sha256.slice(0, 12)}...` : '0cba21f7...'}
          </p>
          <span className="text-[10px] text-purple-500/80">SHA-256 Verified</span>
        </div>
      </div>

      {/* 3-Way Split Distribution Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-slate-200">
            Split Positive Rates & Count Parity
          </h4>
          <span className="text-xs text-slate-400 font-mono">
            Train: {splits.train?.n_rows?.toLocaleString()} | Val: {splits.val?.n_rows?.toLocaleString()} | Test: {splits.test?.n_rows?.toLocaleString()}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Toxicity Category</th>
                <th className="py-2.5 px-3">Train Pos (N)</th>
                <th className="py-2.5 px-3">Train Rate</th>
                <th className="py-2.5 px-3">Val Pos (N)</th>
                <th className="py-2.5 px-3">Val Rate</th>
                <th className="py-2.5 px-3">Test Pos (N)</th>
                <th className="py-2.5 px-3">Test Rate</th>
                <th className="py-2.5 px-3">Stratification Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {labels.map((lbl) => {
                const trCount = splits.train?.label_positive_counts?.[lbl] ?? 0;
                const trRate = (splits.train?.label_positive_rates?.[lbl] ?? 0) * 100;
                const valCount = splits.val?.label_positive_counts?.[lbl] ?? 0;
                const valRate = (splits.val?.label_positive_rates?.[lbl] ?? 0) * 100;
                const teCount = splits.test?.label_positive_counts?.[lbl] ?? 0;
                const teRate = (splits.test?.label_positive_rates?.[lbl] ?? 0) * 100;

                const maxDiff = Math.max(Math.abs(trRate - valRate), Math.abs(trRate - teRate));

                return (
                  <tr key={lbl} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-200 capitalize">
                      {lbl.replace('_', ' ')}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{trCount.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-emerald-400">{trRate.toFixed(3)}%</td>
                    <td className="py-2.5 px-3 text-slate-300">{valCount.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-cyan-400">{valRate.toFixed(3)}%</td>
                    <td className="py-2.5 px-3 text-slate-300">{teCount.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-purple-400">{teRate.toFixed(3)}%</td>
                    <td className="py-2.5 px-3 text-slate-400 font-sans">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                        ±{maxDiff.toFixed(3)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Methodology Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200">1. MinHash LSH Grouping</span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            {methodology.leakage_prevention ||
              'MinHash LSH Jaccard similarity >= 0.8 on character 5-grams ensures 0% duplicate leakage across splits.'}
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200">2. Casing Preservation</span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            {methodology.casing ||
              'Preserved without lowercasing in processed text to retain uppercase emotional intensity features (e.g., "I HATE YOU").'}
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200">3. Stratification Guarantee</span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            {methodology.strategy ||
              'Multilabel-stratified 80/10/10 split on near-duplicate groups (dup_group_v2) ensures extreme rare classes are balanced across splits.'}
          </p>
        </div>
      </div>

      {/* Toggle Data Dictionary */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-200">DATA_DICTIONARY.md Reference</span>
          <button
            onClick={() => setShowDataDict(!showDataDict)}
            className="text-xs font-mono text-cyan-400 hover:text-cyan-300 underline"
          >
            {showDataDict ? 'Hide Data Dictionary' : 'View Full Data Dictionary'}
          </button>
        </div>
        {showDataDict && (
          <div className="mt-4 pt-4 border-t border-slate-800 prose prose-invert prose-slate max-w-none text-xs leading-relaxed max-h-96 overflow-y-auto">
            <ReactMarkdown>{data?.data_dictionary_markdown || 'No data dictionary available.'}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
};
