import React, { useEffect, useState } from 'react';
import { fetchResearchModels } from '../../services/api';
import { Loader2, Cpu, Sliders, Award, AlertCircle, Sparkles } from 'lucide-react';

export const ResearchModels: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeModel, setActiveModel] = useState<string>('3_distilroberta');

  useEffect(() => {
    fetchResearchModels()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 text-slate-400 gap-2">
        <Loader2 className="animate-spin" size={20} />
        <span>Loading model configurations and tuning trials...</span>
      </div>
    );
  }

  const metas = data?.metadata_by_model || {};
  const tuningTrials = data?.tuning_trials || [];
  const blend = data?.blend_summary || {};

  const modelsList = [
    {
      id: '3_distilroberta',
      name: 'DistilRoBERTa-base (Serving Default)',
      type: 'Transformer',
      device: 'CUDA (RTX 3060)',
      role: 'Headline Production Model',
      tagColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      description:
        'Fine-tuned DistilRoBERTa with damped positive class weighting, layerwise learning rate decay, and log-odds bias initialization.',
    },
    {
      id: '2b_bigru_attn_v2',
      name: 'BiGRU + Self-Attention v2',
      type: 'Recurrent Neural Net',
      device: 'CUDA (RTX 3060)',
      role: 'Deep Learning Corroborator',
      tagColor: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
      description:
        'Bidirectional GRU with multi-head self-attention, spatial dropout, layer normalization, and gradient clipping to resolve v1 gradient instability.',
    },
    {
      id: '1_tfidf_lr',
      name: 'TF-IDF + Logistic Regression',
      type: 'Linear Baseline',
      device: 'CPU',
      role: 'Ultra-fast Linear Baseline',
      tagColor: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      description:
        'Word & character n-gram TF-IDF vectorizer (1-2 words, 2-6 chars) paired with OvR Logistic Regression using sublinear term-frequency scaling.',
    },
    {
      id: 'detoxify',
      name: 'Detoxify (Unitary AI)',
      type: 'External Baseline',
      device: 'CUDA (RTX 3060)',
      role: 'Pretrained Upper Bound',
      tagColor: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      description:
        'External off-the-shelf toxic comment classifier trained on the raw 2018 Kaggle set without near-duplicate deduplication.',
    },
  ];

  const currentMeta = metas[activeModel] || {};

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div>
        <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Cpu className="text-emerald-400" size={20} />
          Model Architecture, Tuning & Ensembling
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Detailed specifications for all 4 evaluated architectures, Optuna Bayesian hyperparameter search, and blend analysis.
        </p>
      </div>

      {/* Model Selection Tabs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {modelsList.map((m) => {
          const isSelected = m.id === activeModel;
          return (
            <button
              key={m.id}
              onClick={() => setActiveModel(m.id)}
              className={`p-4 rounded-xl border text-left transition-all ${
                isSelected
                  ? 'bg-slate-900 border-emerald-500/50 shadow-lg ring-1 ring-emerald-500/30'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${m.tagColor}`}>
                  {m.type}
                </span>
                <span className="text-[10px] text-slate-500 font-mono">{m.device}</span>
              </div>
              <h4 className="text-xs font-bold text-slate-100">{m.name}</h4>
              <p className="text-[10px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                {m.description}
              </p>
            </button>
          );
        })}
      </div>

      {/* Selected Model Deep Dive Card */}
      <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div>
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <span>{modelsList.find((m) => m.id === activeModel)?.name}</span>
            </h4>
            <span className="text-[11px] font-mono text-slate-400">
              Artifact: models/{activeModel}/meta.json
            </span>
          </div>
          {currentMeta.seed && (
            <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-1 rounded border border-slate-800">
              Random Seed: {currentMeta.seed}
            </span>
          )}
        </div>

        {/* Hyperparameter / Config Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-500">Input Column</span>
            <p className="font-mono text-slate-200 mt-0.5">{currentMeta.input_column || 'text_clean'}</p>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-500">Max Length</span>
            <p className="font-mono text-slate-200 mt-0.5">
              {currentMeta.config?.tokenizer?.max_len ? `${currentMeta.config.tokenizer.max_len} tokens` : 'Adaptive'}
            </p>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-500">Optimizer / Loss</span>
            <p className="font-mono text-slate-200 mt-0.5">
              {currentMeta.config?.training?.optimizer?.toUpperCase() || 'AdamW'} (Weighted BCE)
            </p>
          </div>
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-500">Mixed Precision</span>
            <p className="font-mono text-slate-200 mt-0.5">
              {currentMeta.config?.training?.mixed_precision || 'fp32 / native'}
            </p>
          </div>
        </div>

        {/* Optimal Thresholds Table */}
        {currentMeta.optimal_thresholds && (
          <div className="pt-2">
            <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold block mb-2">
              Validation-Tuned Decision Thresholds (F1-Optimal)
            </span>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
              {Object.entries(currentMeta.optimal_thresholds).map(([lbl, thresh]: [string, any]) => (
                <div key={lbl} className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-center font-mono">
                  <span className="text-[10px] text-slate-500 block capitalize">{lbl.replace('_', ' ')}</span>
                  <span className="text-xs font-bold text-emerald-400">
                    {typeof thresh === 'number' ? thresh.toFixed(2) : thresh}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Optuna Hyperparameter Trials Section */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sliders size={16} className="text-cyan-400" />
              DistilRoBERTa Optuna Hyperparameter Optimization (5 Trials)
            </h4>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Bayesian optimization optimizing validation Macro PR-AUC on <code>3_distilroberta</code>.
            </p>
          </div>
          <span className="text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-md font-mono flex items-center gap-1">
            <Award size={13} />
            Winner: Trial 4 (0.5996)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] font-mono border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Trial</th>
                <th className="py-2.5 px-3">Val Macro PR-AUC</th>
                <th className="py-2.5 px-3">Learning Rate</th>
                <th className="py-2.5 px-3">pos_weight Cap</th>
                <th className="py-2.5 px-3">Focal Gamma</th>
                <th className="py-2.5 px-3">Dropout</th>
                <th className="py-2.5 px-3">LLRD</th>
                <th className="py-2.5 px-3">Duration</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {tuningTrials.map((t: any) => {
                const isWinner = t.number === 4 || t.number === '4';
                return (
                  <tr
                    key={t.number}
                    className={`hover:bg-slate-800/40 ${isWinner ? 'bg-emerald-500/10 font-semibold' : ''}`}
                  >
                    <td className="py-2.5 px-3 text-slate-200 flex items-center gap-1.5 font-sans">
                      <span>Trial {t.number}</span>
                      {isWinner && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                          SELECTED
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">
                      {typeof t.value === 'number' ? t.value.toFixed(4) : t.value}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {typeof t.params_lr === 'number' ? t.params_lr.toExponential(2) : t.params_lr}
                    </td>
                    <td className="py-2.5 px-3 text-cyan-300">
                      {typeof t.params_pos_weight_cap === 'number'
                        ? t.params_pos_weight_cap.toFixed(1)
                        : t.params_pos_weight_cap}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">{t.params_focal_gamma}</td>
                    <td className="py-2.5 px-3 text-slate-300">{t.params_head_dropout}</td>
                    <td className="py-2.5 px-3 text-slate-300">{t.params_layerwise_lr_decay}</td>
                    <td className="py-2.5 px-3 text-slate-400 text-[11px] font-sans">
                      {String(t.duration || '').replace('0 days ', '')}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 font-sans">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {t.state}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model 2 Setback & TF-IDF Blend Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model 2 vs 2b story */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <AlertCircle size={14} className="text-amber-400" />
            Model 2 Setback & Model 2b Resolution
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Model 2 (BiGRU + Attention v1) suffered from recurrent gradient explosion and severe under-recall on rare categories (threat, severe_toxic). 
            In Model 2b (v2), we introduced <strong>spatial dropout (0.25)</strong>, <strong>layer normalization</strong> across recurrent states, 
            and capped gradient clipping at 1.0, raising threat F1 from 0.4567 to <strong>0.5652</strong> (+10.8%).
          </p>
        </div>

        {/* Blend analysis */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs space-y-2">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            <Sparkles size={14} className="text-cyan-400" />
            Linear Blend Analysis: TF-IDF + DistilRoBERTa
          </span>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            Grid search evaluated ensembling Model 1 (TF-IDF) with Model 3 at various weights:
            <br />
            Optimal blend weight: <code className="text-cyan-300 font-mono">α = {blend.best_alpha || '0.55'}</code> yielding validation score <code className="text-emerald-400 font-mono">{blend.best_blend_val_score || '0.7401'}</code> vs single seed <code className="text-slate-300 font-mono">{blend.best_single_val_macro_pr_auc || '0.7228'}</code> (+1.7%).
            For production serving, single-seed <code>3_distilroberta</code> is chosen to avoid dual-model CPU/GPU inference latency and cold-start overhead.
          </p>
        </div>
      </div>
    </div>
  );
};
