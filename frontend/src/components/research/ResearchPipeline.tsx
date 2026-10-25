import React, { useState } from 'react';
import {
  Database,
  Cpu,
  Layers,
  Workflow,
  ArrowRight,
  Sliders,
  Server,
  CheckCircle2,
  FileCheck,
} from 'lucide-react';

interface PipelineStep {
  id: string;
  number: string;
  title: string;
  subtitle: string;
  icon: any;
  color: string;
  borderColor: string;
  bgColor: string;
  inputs: string[];
  outputs: string[];
  details: string[];
  codeSnippet?: string;
}

export const ResearchPipeline: React.FC = () => {
  const [selectedStep, setSelectedStep] = useState<string>('split');

  const steps: PipelineStep[] = [
    {
      id: 'ingestion',
      number: '01',
      title: 'Raw Data & MinHash LSH',
      subtitle: 'Deduplication & Leakage Prevention',
      icon: Database,
      color: 'text-blue-400',
      borderColor: 'border-blue-500/30',
      bgColor: 'bg-blue-950/20',
      inputs: ['Kaggle Toxic Comment Classification Corpus (159,571 comments)', '6 Multi-Label Binary Flags'],
      outputs: ['dup_group_v2 near-duplicate cluster IDs', 'Zero cross-split leakage guarantee'],
      details: [
        'Character 5-gram shingling across comments ≥ 20 characters.',
        'MinHash LSH with 128 permutations at Jaccard similarity threshold ≥ 0.80.',
        'Ensures near-verbatim spam comments are clustered into unified duplicate groups rather than split between train and test.',
      ],
      codeSnippet: `MinHashLSH(threshold=0.8, num_perm=128)\n# Exact near-duplicates assigned identical dup_group_v2 IDs`,
    },
    {
      id: 'split',
      number: '02',
      title: 'Group-Stratified Partitioning',
      subtitle: '80% Train / 10% Val / 10% Test',
      icon: Layers,
      color: 'text-indigo-400',
      borderColor: 'border-indigo-500/30',
      bgColor: 'bg-indigo-950/20',
      inputs: ['159,571 comments with dup_group_v2 IDs'],
      outputs: ['Train (127,861)', 'Validation (15,869)', 'Test (15,841)'],
      details: [
        'Iterative multi-label group-stratified split on dup_group_v2 clusters.',
        'Preserves exact positive ratios for extreme rare classes (threat 0.30%, identity_hate 0.88%).',
        'Text casing is preserved without lowercasing to retain emotional intensity signals (ALL-CAPS rage).',
      ],
      codeSnippet: `StratifiedGroupKFold(n_splits=10, random_state=42)\n# Train: 80% | Val: 10% (calibration) | Test: 10% (final benchmark)`,
    },
    {
      id: 'models',
      number: '03',
      title: 'Multi-Model Training',
      subtitle: 'Linear, RNN, Transformer & Baseline',
      icon: Cpu,
      color: 'text-cyan-400',
      borderColor: 'border-cyan-500/30',
      bgColor: 'bg-cyan-950/20',
      inputs: ['Tokenized train splits', 'Class-balanced BCE loss with pos_weight'],
      outputs: ['Model 1: TF-IDF + LogReg', 'Model 2b: BiGRU + Attention v2', 'Model 3: DistilRoBERTa (Serving)'],
      details: [
        'Model 1: Word + Char n-grams (1-2) with sublinear term-frequency scaling (CPU fast linear).',
        'Model 2b: Bidirectional GRU with self-attention, spatial dropout, and layer normalization on CUDA.',
        'Model 3: DistilRoBERTa-base fine-tuned with Optuna-tuned hyperparameters (Trial 4: lr 3.86e-5, pos_weight cap 14).',
        'Detoxify: Pretrained multi-label transformer by Unitary AI evaluated as external reference.',
      ],
      codeSnippet: `BCEWithLogitsLoss(pos_weight=torch.clamp(neg/pos, max=14.0))\n# Tuned with Optuna 5-trial sweep (winner: Trial 4 macro PR-AUC 0.600)`,
    },
    {
      id: 'calibration',
      number: '04',
      title: 'Validation Policy Calibration',
      subtitle: 'Zero Test Leakage Rule Derivation',
      icon: Sliders,
      color: 'text-amber-400',
      borderColor: 'border-amber-500/30',
      bgColor: 'bg-amber-950/20',
      inputs: ['Validation probabilities (df_val, 15,869 samples)', 'F1-maximizing grid thresholds'],
      outputs: ['policy.yaml with calibrated remove_cut values', 'Piecewise linear normalization curves'],
      details: [
        'Asymmetric remove cuts derived where validation precision ≥ 90% with ≥ 10 positive predictions.',
        'High-sparsity labels (threat, severe_toxic) cap at Orange ("Hide with warning") to prevent false removals.',
        'Piecewise linear normalization maps model threshold t to 0.50, ensuring consistent UI interpretability.',
        'Severe toxic hierarchy constraint: P(severe_toxic) > t triggers toxic flag automatically.',
      ],
      codeSnippet: `python backend/scripts/calibrate_policy.py\n# Evaluates df_val ONLY. Calibrated cuts: identity_hate 0.900, threat null`,
    },
    {
      id: 'serving',
      number: '05',
      title: 'Production Serving & Moderator API',
      subtitle: 'FastAPI Lifespan + MongoDB Atlas',
      icon: Server,
      color: 'text-emerald-400',
      borderColor: 'border-emerald-500/30',
      bgColor: 'bg-emerald-950/20',
      inputs: ['Incoming comment payload (<5000 chars)', 'policy.yaml + curse_words.txt'],
      outputs: ['Moderation Action (Allow / Flag / Hide / Remove)', '0-10 Severity Score', 'Word-level Censor'],
      details: [
        'Models preloaded into CUDA memory during FastAPI lifespan startup (sub-millisecond inference).',
        'Multi-model corroboration: automated Red "Remove" requires consensus from ≥ 2 models.',
        'Case-insensitive profanity censor with wildcard and leetspeak translation (@, $, 0, 1, !).',
        'All moderated comments logged to MongoDB Atlas with full prediction metadata and reasoning traces.',
      ],
      codeSnippet: `POST /api/predict -> {\n  "headline": { "action": "Hide with warning", "severity": 6.8 },\n  "censored_segments": [...]\n}`,
    },
  ];

  const current = steps.find((s) => s.id === selectedStep) || steps[0];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Workflow className="text-cyan-400" size={20} />
          End-to-End System Pipeline & Lifecycle
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          From raw multi-label text deduplication through transformer fine-tuning to high-precision calibrated serving.
        </p>
      </div>

      {/* Horizontal Step Navigation Flow */}
      <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
        {steps.map((s) => {
          const isSelected = s.id === selectedStep;
          const Icon = s.icon;

          return (
            <button
              key={s.id}
              onClick={() => setSelectedStep(s.id)}
              className={`p-3 rounded-xl border text-left transition-all relative ${
                isSelected
                  ? `${s.bgColor} ${s.borderColor} shadow-lg ring-1 ring-cyan-500/30`
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-mono font-bold text-slate-400">{s.number}</span>
                <Icon size={16} className={s.color} />
              </div>
              <h4 className={`text-xs font-semibold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                {s.title}
              </h4>
              <p className="text-[10px] text-slate-400 mt-0.5 truncate">{s.subtitle}</p>
            </button>
          );
        })}
      </div>

      {/* Detailed Step Inspector */}
      <div className={`p-6 rounded-2xl border ${current.borderColor} ${current.bgColor} space-y-6`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl bg-slate-900/80 border ${current.borderColor}`}>
              <current.icon size={22} className={current.color} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                  PHASE {current.number}
                </span>
                <h4 className="text-base font-bold text-white">{current.title}</h4>
              </div>
              <p className="text-xs text-slate-300 mt-0.5">{current.subtitle}</p>
            </div>
          </div>
        </div>

        {/* Inputs vs Outputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-2">
            <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
              <ArrowRight size={13} className="text-blue-400" />
              Artifact Inputs
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {current.inputs.map((inp, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-slate-500">•</span>
                  <span>{inp}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-2">
            <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
              <CheckCircle2 size={13} className="text-emerald-400" />
              Produced Artifacts & Guarantees
            </span>
            <ul className="space-y-1.5 text-xs text-slate-300">
              {current.outputs.map((out, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-emerald-500">•</span>
                  <span className="text-emerald-200/90 font-medium">{out}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Core Principles */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-2.5">
          <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
            <FileCheck size={13} className="text-amber-400" />
            Engineering Details & Methodology
          </span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {current.details.map((det, idx) => (
              <div key={idx} className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                {det}
              </div>
            ))}
          </div>
        </div>

        {/* Code Snippet if applicable */}
        {current.codeSnippet && (
          <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 p-4 font-mono text-xs text-cyan-300">
            <span className="text-[10px] text-slate-500 block mb-1 uppercase tracking-wider">
              Implementation Signature
            </span>
            <pre className="whitespace-pre-wrap">{current.codeSnippet}</pre>
          </div>
        )}
      </div>
    </div>
  );
};
