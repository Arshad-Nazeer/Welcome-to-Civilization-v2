import React from 'react';
import { Shield, Sparkles, Cpu, Database, BookOpen, Layers } from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface NavbarProps {
  activeTab: 'moderate' | 'research';
  onSelectTab: (tab: 'moderate' | 'research') => void;
  health: HealthResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  onSelectTab,
  health,
}) => {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 p-0.5 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Shield className="text-emerald-400" size={20} />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-white text-base md:text-lg">
                Welcome to Civilization
              </span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                v2.0
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Multi-Label Toxicity Classification & Policy Engine
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center gap-1.5 p-1 bg-slate-900/90 rounded-xl border border-slate-800">
          <button
            onClick={() => onSelectTab('moderate')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'moderate'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <Layers size={16} />
            <span>Moderate</span>
          </button>
          <button
            onClick={() => onSelectTab('research')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'research'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <BookOpen size={16} />
            <span>Research</span>
          </button>
        </nav>

        {/* System Status Indicators */}
        <div className="hidden lg:flex items-center gap-3 text-xs">
          {/* Models */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800 text-slate-300">
            <Sparkles size={13} className="text-emerald-400" />
            <span>
              Models: <strong className="text-slate-100">{health?.models_available ?? '4'}/{health?.total_models ?? '4'}</strong>
            </span>
          </div>

          {/* Hardware */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800 text-slate-300">
            <Cpu size={13} className={health?.gpu_available ? 'text-teal-400' : 'text-slate-400'} />
            <span>{health?.gpu_available ? 'RTX 3060 (CUDA)' : 'CPU Mode'}</span>
          </div>

          {/* Database */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800 text-slate-300">
            <Database size={13} className={health?.mongodb_connected ? 'text-emerald-400' : 'text-yellow-400'} />
            <span>{health?.mongodb_connected ? 'MongoDB Atlas' : 'In-Memory Store'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
