import React, { useEffect, useState } from 'react';
import { fetchResearchImbalance } from '../../services/api';
import { Loader2, AlertOctagon, Scale, ShieldAlert, Cpu, CheckCircle } from 'lucide-react';

export const ResearchImbalance: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchResearchImbalance()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading class imbalance statistics and policy rationale...</span>
      </div>
    );
  }

  const manifest = data?.split_manifest || {};
  const trainRates = manifest?.splits?.train?.label_positive_rates || {};
  const trainCounts = manifest?.splits?.train?.label_positive_counts || {};
  const policy = data?.policy_configuration || {};
  const removeCuts = policy?.remove_cuts || {};

  const labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <AlertOctagon className="text-rose-400" size={20} />
          Extreme Class Imbalance & Asymmetric Policy Engineering
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          How dataset sparsity (threat at 0.30% positive) directly shapes our four-tier moderation policy and corroboration rules.
        </p>
      </div>

      {/* Rationale Banner */}
      <div className="p-5 rounded-xl bg-gradient-to-r from-rose-950/20 via-slate-900 to-amber-950/20 border border-rose-500/30 text-xs space-y-2">
        <span className="font-semibold text-rose-300 flex items-center gap-1.5 text-sm">
          <ShieldAlert size={16} />
          The Imbalance Challenge in Automated Content Moderation
        </span>
        <p className="text-slate-300 leading-relaxed text-[11px]">
          {data?.asymmetric_rationale}
        </p>
      </div>

      {/* Imbalance Severity Visualization Grid */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <h4 className="text-sm font-semibold text-slate-200">
          Positive Class Ratios in Training Split (N = 127,861)
        </h4>

        <div className="space-y-3">
          {labels.map((lbl) => {
            const rate = (trainRates[lbl] || 0) * 100;
            const count = trainCounts[lbl] || 0;
            const isExtreme = rate < 1.0;

            return (
              <div key={lbl} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="font-medium text-slate-300 capitalize flex items-center gap-2">
                    {lbl.replace('_', ' ')}
                    {isExtreme && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30 font-mono">
                        EXTREME SPARSITY (&lt;1%)
                      </span>
                    )}
                  </span>
                  <span className="font-mono text-slate-400 text-[11px]">
                    <strong className="text-slate-200">{count.toLocaleString()}</strong> pos ({rate.toFixed(3)}%)
                  </span>
                </div>
                <div className="h-2.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isExtreme ? 'bg-rose-500' : 'bg-cyan-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(1, rate * 10))}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4 Architectural Policy Mitigations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-cyan-300 flex items-center gap-1.5">
            <Scale size={14} />
            1. Piecewise Linear Score Normalization
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Model thresholds vary widely (e.g., toxic at 0.66 vs threat at 0.68 vs detoxify at 0.34). Raw probabilities are misleading to end users.
            We apply piecewise linear mapping:
            <br />
            <code className="text-cyan-400 font-mono">P ∈ [0, t] → [0.0, 0.5]</code> and{' '}
            <code className="text-cyan-400 font-mono">P ∈ [t, 1.0] → [0.5, 1.0]</code>.
            This ensures that 0.50 consistently denotes the threshold boundary across all models.
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-emerald-300 flex items-center gap-1.5">
            <CheckCircle size={14} />
            2. High-Precision Remove Cuts (Policy Calibration)
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Automated Red ("Remove") requires &ge; 90% validation precision with at least 10 positive cases. 
            Because <code>severe_toxic</code> and <code>threat</code> never achieve 90% precision due to extreme class sparsity, 
            their <code>remove_cut</code> is explicitly set to <code className="text-amber-400 font-mono">null</code>. 
            They safely cap at Orange ("Hide with warning") to prevent devastating false deletions.
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-amber-300 flex items-center gap-1.5">
            <Cpu size={14} />
            3. Multi-Model Corroboration for Red Removal
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            To eliminate single-model hallucinations, no single model can trigger automated hard Red deletion alone. 
            Automated removal mandates consensus from at least two available models (e.g. DistilRoBERTa + BiGRU or Detoxify).
          </p>
        </div>

        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-purple-300 flex items-center gap-1.5">
            <ShieldAlert size={14} />
            4. Watch Zone [0.30, 0.50) & Flagging
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Comments with normalized scores between 0.30 and 0.50 are placed in the "Watch Zone" and routed to human moderator review (<code className="text-purple-400 font-mono">needs_review: true</code>) 
            rather than silently allowed.
          </p>
        </div>
      </div>

      {/* Live Calibrated Remove Cuts Table */}
      <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-3">
        <h4 className="text-xs font-mono uppercase text-slate-400 font-semibold">
          Calibrated Asymmetric Remove Cuts in Production (backend/policy.yaml)
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          {Object.entries(removeCuts).map(([mId, cuts]: [string, any]) => (
            <div key={mId} className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
              <span className="font-bold text-slate-200 block text-[11px] truncate">{mId}</span>
              {Object.entries(cuts || {}).map(([cLabel, cutVal]: [string, any]) => (
                <div key={cLabel} className="flex justify-between text-[10px]">
                  <span className="text-slate-500 capitalize">{cLabel.replace('_', ' ')}</span>
                  <span className={cutVal !== null ? 'text-emerald-400 font-bold' : 'text-slate-600'}>
                    {cutVal !== null ? cutVal : 'null (capped)'}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
