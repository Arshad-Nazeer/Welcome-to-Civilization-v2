import React, { useState } from 'react';
import type {
  PredictResponse,
  CommentRecord,
  ModelInfo,
} from '../types/api';
import { postPredict } from '../services/api';
import { ActionBadge } from './ActionBadge';
import { SeverityMeter } from './SeverityMeter';
import { ProbabilityBar } from './ProbabilityBar';
import { OverviewComparison } from './OverviewComparison';
import { CommentsHistory } from './CommentsHistory';
import { CensoredText } from './CensoredText';
import {
  Send,
  Loader2,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  AlertCircle,
  Cpu,
  Zap,
  Info,
} from 'lucide-react';

interface ModerateTabProps {
  comments: CommentRecord[];
  totalComments: number;
  currentPage: number;
  totalPages: number;
  activeFilter: string;
  onPageChange: (page: number) => void;
  onFilterChange: (action: string) => void;
  onNewPrediction: (pred: PredictResponse) => void;
  onDeleteComment?: (id: string) => Promise<void>;
  globalCensor: boolean;
  onToggleGlobalCensor: () => void;
  modelsInfo: ModelInfo[];
  isLoadingHistory: boolean;
}

const MODEL_TAB_IDS = [
  { id: 'overview', name: 'Overview Matrix', short: 'Overview' },
  { id: '3_distilroberta', name: 'Model 3: DistilRoBERTa', short: 'DistilRoBERTa' },
  { id: '2b_bigru_attn_v2', name: 'Model 2b: BiGRU + Attn', short: 'BiGRU v2' },
  { id: '1_tfidf_lr', name: 'Model 1: TF-IDF + LR', short: 'TF-IDF' },
  { id: 'detoxify', name: 'Detoxify (External Pretrained)', short: 'Detoxify' },
];

const PRESET_COMMENTS = [
  { label: 'Benign Comment', text: 'Thank you for explaining the article structure. I appreciate your thoughtful feedback!' },
  { label: 'Overt Insult', text: 'You are an absolute idiot and completely incompetent. Get off Wikipedia.' },
  { label: 'Severe Threat', text: 'I know where you live and I am going to find you and kill you.' },
  { label: 'Compound / Masked Profanity', text: 'What the f*ck is wrong with you, you dumb a$$hole and bitch.' },
];

export const ModerateTab: React.FC<ModerateTabProps> = ({
  comments,
  totalComments,
  currentPage,
  totalPages,
  activeFilter,
  onPageChange,
  onFilterChange,
  onNewPrediction,
  onDeleteComment,
  globalCensor,
  onToggleGlobalCensor,
  modelsInfo,
  isLoadingHistory,
}) => {
  const [inputText, setInputText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<PredictResponse | null>(null);
  const [activeModelTab, setActiveModelTab] = useState<string>('3_distilroberta');

  const charCount = inputText.length;
  const isOverLimit = charCount > 5000;
  const canSubmit = inputText.trim().length > 0 && !isOverLimit && !isSubmitting;

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!canSubmit) return;

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const result = await postPredict(inputText.trim());
      setLatestResult(result);
      onNewPrediction(result);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to get moderation prediction.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePresetClick = (text: string) => {
    setInputText(text);
    setErrorMessage(null);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Input Moderation Form */}
      <section className="glass-panel p-6 rounded-2xl shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Sparkles className="text-emerald-400" size={18} />
              Real-Time Content Moderation
            </h2>
            <p className="text-xs text-slate-400">
              Submit text to evaluate across all 4 machine learning models and policy thresholds.
            </p>
          </div>

          {/* Global Profanity Censor Toggle */}
          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-300 font-medium">Censor profanity:</span>
            <button
              type="button"
              onClick={onToggleGlobalCensor}
              className="flex items-center text-xs font-semibold cursor-pointer"
            >
              {globalCensor ? (
                <span className="flex items-center gap-1 text-emerald-400">
                  <ToggleRight size={22} />
                  <span>ON</span>
                </span>
              ) : (
                <span className="flex items-center gap-1 text-slate-500">
                  <ToggleLeft size={22} />
                  <span>OFF</span>
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Text Input Area */}
        <form onSubmit={handlePredict} className="space-y-3">
          <div className="relative">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Paste or type a user comment to test toxicity detection (up to 5,000 characters)..."
              rows={4}
              className="w-full bg-slate-950/90 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all font-sans resize-y"
            />
            {/* Character counter */}
            <div className="absolute right-3 bottom-3 text-[11px] font-mono text-slate-500 bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">
              <span className={isOverLimit ? 'text-rose-400 font-bold' : ''}>
                {charCount}
              </span>
              /5000
            </div>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
            <span className="text-slate-500 text-[11px]">Test examples:</span>
            {PRESET_COMMENTS.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handlePresetClick(preset.text)}
                className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3 rounded-lg bg-rose-500/15 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle size={16} className="shrink-0 text-rose-400" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Submit Action */}
          <div className="flex items-center justify-between pt-2">
            <span className="text-[11px] text-slate-500">
              Evaluates asynchronously against Model 1, Model 2b, Model 3 & Detoxify.
            </span>
            <button
              type="submit"
              disabled={!canSubmit}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 hover:from-emerald-400 hover:to-teal-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-md shadow-emerald-500/20"
            >
              {isSubmitting ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Evaluating Models...</span>
                </>
              ) : (
                <>
                  <Send size={16} />
                  <span>Predict Toxicity</span>
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {/* Latest Evaluation Result */}
      {latestResult && (
        <section className="glass-panel p-6 rounded-2xl shadow-xl space-y-6 animate-fadeIn">
          {/* Headline Recommendation Card */}
          <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-semibold">
                    Headline Recommendation (Model 3 Serving)
                  </span>
                  <span className="text-xs text-slate-500 font-mono">
                    ID: {latestResult.id}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-slate-100 mt-1">
                  Overall Decision: {latestResult.headline.action}
                </h3>
              </div>

              <div className="flex items-center gap-4">
                <ActionBadge action={latestResult.headline.action} size="lg" />
                <div className="w-32">
                  <SeverityMeter severity={latestResult.headline.severity} size="md" />
                </div>
              </div>
            </div>

            {/* Input Echo with Word-level Censor Toggle */}
            <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 text-sm">
              <span className="text-xs text-slate-500 block mb-1">Original Comment Input:</span>
              <CensoredText
                segments={latestResult.censored_segments}
                originalText={latestResult.original_text}
                globalCensor={globalCensor}
                className="text-slate-200"
              />
            </div>

            {/* Headline Reasoning */}
            {latestResult.headline.reasoning?.length > 0 && (
              <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-xs space-y-1.5">
                <div className="flex items-center gap-1.5 font-semibold text-slate-300">
                  <Info size={14} className="text-emerald-400" />
                  <span>Policy Rule Reasoning:</span>
                </div>
                <ul className="list-disc pl-5 space-y-1 text-slate-400 font-sans">
                  {latestResult.headline.reasoning.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Model Sub-Tabs Navigation */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 overflow-x-auto pb-1 border-b border-slate-800">
              {MODEL_TAB_IDS.map((tab) => {
                const isActive = activeModelTab === tab.id;
                const m = latestResult.models[tab.id];
                const isAvail = tab.id === 'overview' ? true : m?.available;

                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveModelTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs md:text-sm font-medium transition-all whitespace-nowrap ${
                      isActive
                        ? 'bg-slate-800 text-white border border-slate-700 shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                    }`}
                  >
                    <span>{tab.name}</span>
                    {tab.id === 'detoxify' && (
                      <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.2 rounded border border-purple-500/30">
                        External
                      </span>
                    )}
                    {tab.id !== 'overview' && m && isAvail && (
                      <span
                        className={`w-2 h-2 rounded-full ${
                          m.action_color === 'red'
                            ? 'bg-rose-500'
                            : m.action_color === 'orange'
                            ? 'bg-amber-500'
                            : m.action_color === 'yellow'
                            ? 'bg-yellow-500'
                            : 'bg-emerald-500'
                        }`}
                      />
                    )}
                  </button>
                );
              })}
            </div>

            {/* Tab Body: Overview Matrix vs Specific Model Breakdown */}
            {activeModelTab === 'overview' ? (
              <OverviewComparison
                models={latestResult.models}
                headlineModelId={latestResult.headline.model_id}
              />
            ) : (
              (() => {
                const m = latestResult.models[activeModelTab];
                if (!m) {
                  return (
                    <div className="text-center py-8 text-slate-500 text-sm">
                      Model information not available.
                    </div>
                  );
                }

                if (!m.available) {
                  return (
                    <div className="p-8 rounded-xl bg-slate-900/60 border border-slate-800 text-center space-y-3">
                      <AlertCircle size={32} className="mx-auto text-yellow-500" />
                      <h4 className="font-semibold text-slate-200 text-base">Model Unavailable</h4>
                      <p className="text-xs text-slate-400 max-w-md mx-auto">
                        {m.error_message || 'Artifacts or dependencies for this model could not be loaded.'}
                      </p>
                    </div>
                  );
                }

                const thresholds = modelsInfo.find((x) => x.id === activeModelTab)?.thresholds || {};

                return (
                  <div className="space-y-6 animate-fadeIn">
                    {/* Model Header */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/60 border border-slate-800">
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-bold text-base text-slate-100">{m.model_name}</h4>
                          {activeModelTab === 'detoxify' && (
                            <span className="text-[11px] font-semibold text-purple-400 bg-purple-500/15 border border-purple-500/30 px-2 py-0.5 rounded-full">
                              External Pretrained Baseline (Unitary AI)
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">{m.role}</p>
                      </div>

                      <div className="flex items-center gap-4">
                        <ActionBadge action={m.action} size="md" />
                        <div className="w-32">
                          <SeverityMeter severity={m.severity} size="sm" />
                        </div>
                      </div>
                    </div>

                    {/* Six Probability Bars */}
                    <div className="space-y-2.5">
                      <div className="text-xs font-semibold text-slate-300 font-mono flex items-center justify-between px-1">
                        <span>Category Predictions & Thresholds</span>
                        <span className="text-slate-500 font-normal">
                          Dashed line indicates calibrated decision threshold
                        </span>
                      </div>

                      {['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'].map((lbl) => (
                        <ProbabilityBar
                          key={lbl}
                          label={lbl}
                          probability={m.probabilities[lbl] ?? 0.0}
                          threshold={thresholds[lbl] ?? 0.5}
                          normalizedScore={m.normalized_scores[lbl] ?? 0.0}
                          isFired={m.per_label_flags[lbl] ?? false}
                        />
                      ))}
                    </div>

                    {/* Model Hardware Performance & Latency */}
                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-950/70 border border-slate-800/80 text-xs text-slate-400">
                      <span className="flex items-center gap-1.5">
                        <Cpu size={14} className="text-slate-400" />
                        Inference Hardware: <strong className="text-slate-200 uppercase">{m.device}</strong>
                      </span>
                      <span className="flex items-center gap-1.5 font-mono">
                        <Zap size={14} className="text-amber-400" />
                        Inference Latency: <strong className="text-slate-200">{m.execution_time_ms.toFixed(1)} ms</strong>
                      </span>
                    </div>

                    {/* Reasoning Details */}
                    {m.reasoning && m.reasoning.length > 0 && (
                      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
                        <span className="font-semibold text-slate-300">Policy Evaluation Reasoning:</span>
                        <ul className="list-disc pl-5 space-y-1 text-slate-400">
                          {m.reasoning.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                );
              })()
            )}
          </div>
        </section>
      )}

      {/* History Beneath Form */}
      <section className="glass-panel p-6 rounded-2xl shadow-xl">
        <CommentsHistory
          comments={comments}
          total={totalComments}
          currentPage={currentPage}
          totalPages={totalPages}
          activeFilter={activeFilter}
          onPageChange={onPageChange}
          onFilterChange={onFilterChange}
          onDeleteComment={onDeleteComment}
          globalCensor={globalCensor}
          isLoading={isLoadingHistory}
        />
      </section>
    </div>
  );
};
