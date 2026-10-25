import React, { useState } from 'react';
import type { CommentRecord } from '../types/api';
import { ActionBadge } from './ActionBadge';
import { SeverityMeter } from './SeverityMeter';
import { CensoredText } from './CensoredText';
import {
  Clock,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  ShieldAlert,
  Filter,
} from 'lucide-react';

interface CommentsHistoryProps {
  comments: CommentRecord[];
  total: number;
  currentPage: number;
  totalPages: number;
  activeFilter: string;
  onPageChange: (page: number) => void;
  onFilterChange: (action: string) => void;
  globalCensor: boolean;
  isLoading: boolean;
}

export const CommentsHistory: React.FC<CommentsHistoryProps> = ({
  comments,
  total,
  currentPage,
  totalPages,
  activeFilter,
  onPageChange,
  onFilterChange,
  globalCensor,
  isLoading,
}) => {
  // Track expanded rows for per-model breakdown
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Track explicit reveals for Orange warning comments
  const [revealedOrange, setRevealedOrange] = useState<Record<string, boolean>>({});

  // Track explicit reveals for Red removed comments (Moderator view)
  const [revealedRed, setRevealedRed] = useState<Record<string, boolean>>({});

  const toggleExpand = (id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  const toggleOrangeView = (id: string) => {
    setRevealedOrange((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleRedView = (id: string) => {
    setRevealedRed((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filterOptions = [
    { label: 'All Actions', value: 'all' },
    { label: 'Remove', value: 'Remove' },
    { label: 'Hide with warning', value: 'Hide with warning' },
    { label: 'Flag for review', value: 'Flag for review' },
    { label: 'Allow', value: 'Allow' },
  ];

  return (
    <div className="space-y-4">
      {/* Header and Filter Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            Moderation Activity History
            <span className="text-xs font-mono font-normal text-slate-400 bg-slate-900 px-2 py-0.5 rounded-full border border-slate-800">
              {total} comments logged
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Chronological audit log persisted to MongoDB Atlas.
          </p>
        </div>

        {/* Filter Badges */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 sm:pb-0">
          <Filter size={14} className="text-slate-500 mr-1 shrink-0" />
          {filterOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => onFilterChange(opt.value)}
              className={`text-xs px-2.5 py-1 rounded-lg border transition-all whitespace-nowrap ${
                activeFilter === opt.value
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-medium'
                  : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* History Items List */}
      {comments.length === 0 ? (
        <div className="text-center py-12 glass-panel rounded-xl text-slate-400 text-sm">
          {isLoading ? 'Loading comments history...' : 'No comments found for this filter.'}
        </div>
      ) : (
        <div className="space-y-3">
          {comments.map((item) => {
            const headline = item.headline || { action: 'Allow', action_color: 'green' };
            const actionNorm = (headline.action || '').toLowerCase();
            const isRed = actionNorm.includes('remove') || actionNorm.includes('red');
            const isOrange = actionNorm.includes('hide') || actionNorm.includes('orange');
            const isExpanded = expandedId === item.id;
            const isOrangeRevealed = !!revealedOrange[item.id];
            const isRedRevealed = !!revealedRed[item.id];

            // Border color by action
            let borderStyle = 'border-slate-800 hover:border-slate-700';
            if (isRed) borderStyle = 'border-rose-500/30 bg-rose-950/10';
            else if (isOrange) borderStyle = 'border-amber-500/30 bg-amber-950/10';
            else if (actionNorm.includes('flag')) borderStyle = 'border-yellow-500/30 bg-yellow-950/10';
            else borderStyle = 'border-emerald-500/20 bg-slate-900/30';

            return (
              <div
                key={item.id}
                className={`rounded-xl border transition-all overflow-hidden ${borderStyle}`}
              >
                {/* Main Card Header / Summary Row */}
                <div className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
                  <div className="flex-1 min-w-0 space-y-2">
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <ActionBadge action={headline.action} size="sm" />
                      <span className="flex items-center gap-1 font-mono text-slate-500 text-[11px]">
                        <Clock size={12} />
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                      <span className="text-[11px] text-slate-500 font-mono">
                        ID: {item.id}
                      </span>
                    </div>

                    {/* Text Rendering based on Red/Orange/Allow status */}
                    <div className="text-sm text-slate-200">
                      {isRed ? (
                        /* Red: Removed Card */
                        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-1.5 font-semibold text-rose-400">
                              <ShieldAlert size={14} /> Removed by moderation
                            </span>
                            <button
                              type="button"
                              onClick={() => toggleRedView(item.id)}
                              className="text-[11px] underline text-rose-400/80 hover:text-rose-300 cursor-pointer"
                            >
                              {isRedRevealed ? 'Hide original text' : 'Show original (moderator view)'}
                            </button>
                          </div>
                          <p className="text-[11px] text-rose-300/80">
                            {headline.reasoning?.[0] || 'Violates community standards.'}
                          </p>
                          {isRedRevealed && (
                            <div className="mt-2 pt-2 border-t border-rose-500/20 text-slate-200 font-mono text-xs">
                              <CensoredText
                                segments={item.censored_segments}
                                originalText={item.original_text}
                                globalCensor={globalCensor}
                              />
                            </div>
                          )}
                        </div>
                      ) : isOrange ? (
                        /* Orange: Warning Card */
                        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-1.5 font-semibold text-amber-400">
                              <AlertTriangle size={14} /> Content hidden with warning
                            </span>
                            <button
                              type="button"
                              onClick={() => toggleOrangeView(item.id)}
                              className="px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-300 hover:bg-amber-500/30 text-xs font-medium cursor-pointer"
                            >
                              {isOrangeRevealed ? 'Hide' : 'View comment'}
                            </button>
                          </div>
                          {isOrangeRevealed && (
                            <div className="mt-2 pt-2 border-t border-amber-500/20 text-slate-200 text-sm">
                              <CensoredText
                                segments={item.censored_segments}
                                originalText={item.original_text}
                                globalCensor={globalCensor}
                              />
                            </div>
                          )}
                        </div>
                      ) : (
                        /* Normal / Allowed Text */
                        <div className="line-clamp-3">
                          <CensoredText
                            segments={item.censored_segments}
                            originalText={item.original_text}
                            globalCensor={globalCensor}
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right side: Severity score & Expand Toggle */}
                  <div className="flex md:flex-col items-center md:items-end justify-between gap-3 shrink-0">
                    <div className="w-28 sm:w-32">
                      <SeverityMeter severity={headline.severity} size="sm" />
                    </div>

                    <button
                      type="button"
                      onClick={() => toggleExpand(item.id)}
                      className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700 transition-colors"
                    >
                      <span>{isExpanded ? 'Less' : 'Details'}</span>
                      {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>
                </div>

                {/* Expanded Per-Model Breakdown Drawer */}
                {isExpanded && (
                  <div className="p-4 bg-slate-950/80 border-t border-slate-800 space-y-4">
                    <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                      Architectural Comparison Breakdown:
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                      {['1_tfidf_lr', '2b_bigru_attn_v2', '3_distilroberta', 'detoxify'].map((mId) => {
                        const m = item.models?.[mId];
                        if (!m) return null;
                        return (
                          <div
                            key={mId}
                            className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs space-y-2"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-semibold text-slate-300">
                                {m.model_name || mId}
                              </span>
                              <ActionBadge action={m.action} size="sm" />
                            </div>
                            <div className="text-[11px] font-mono text-slate-400">
                              Severity: <strong className="text-slate-200">{m.severity?.score.toFixed(1)}</strong> ({m.severity?.band})
                            </div>
                            <div className="space-y-1">
                              {Object.entries(m.probabilities || {}).map(([lbl, p]) => (
                                <div key={lbl} className="flex justify-between text-[10px] font-mono text-slate-400">
                                  <span>{lbl}:</span>
                                  <span className={m.per_label_flags?.[lbl] ? 'text-rose-400 font-bold' : ''}>
                                    {p.toFixed(3)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Reasoning list */}
                    {headline.reasoning && headline.reasoning.length > 0 && (
                      <div className="pt-2 border-t border-slate-800 text-xs text-slate-400">
                        <span className="font-semibold text-slate-300">Policy Reasoning:</span>
                        <ul className="list-disc pl-4 mt-1 space-y-0.5 text-[11px]">
                          {headline.reasoning.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Pagination Bar */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-4 border-t border-slate-800 text-xs">
          <span className="text-slate-400 font-mono">
            Page {currentPage} of {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage <= 1}
              className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages}
              className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
