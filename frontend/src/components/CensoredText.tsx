import React, { useState } from 'react';
import type { CensorSegment } from '../types/api';

interface CensoredTextProps {
  segments?: CensorSegment[];
  originalText: string;
  globalCensor?: boolean;
  globalCensorEnabled?: boolean;
  className?: string;
}

export const CensoredText: React.FC<CensoredTextProps> = ({
  segments,
  originalText,
  globalCensor = true,
  globalCensorEnabled,
  className = '',
}) => {
  const isCensorActive = globalCensorEnabled !== undefined ? globalCensorEnabled : globalCensor;
  // Track per-word reveal toggle state
  const [revealedIndices, setRevealedIndices] = useState<Record<number, boolean>>({});

  if (!isCensorActive || !segments || segments.length === 0) {
    return <span className={`whitespace-pre-wrap ${className}`}>{originalText}</span>;
  }

  const toggleWord = (index: number) => {
    setRevealedIndices((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <span className={`whitespace-pre-wrap ${className}`}>
      {segments.map((seg, idx) => {
        if (!seg.censored) {
          return <React.Fragment key={idx}>{seg.text}</React.Fragment>;
        }

        const isRevealed = !!revealedIndices[idx];

        if (isRevealed) {
          return (
            <button
              key={idx}
              type="button"
              onClick={() => toggleWord(idx)}
              className="inline px-1 py-0.5 rounded bg-rose-500/20 text-rose-300 font-mono text-xs border border-rose-500/30 hover:bg-rose-500/30 transition-colors cursor-pointer"
              title="Click to re-censor"
            >
              {seg.text}
            </button>
          );
        }

        return (
          <button
            key={idx}
            type="button"
            onClick={() => toggleWord(idx)}
            className="inline px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono text-xs border border-slate-700 hover:border-slate-500 hover:text-slate-200 transition-colors cursor-pointer select-none"
            title="Click to reveal profanity"
          >
            {'*'.repeat(Math.max(seg.text.length, 4))}
          </button>
        );
      })}
    </span>
  );
};
