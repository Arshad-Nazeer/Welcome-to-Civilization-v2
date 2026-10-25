import React, { useState } from 'react';
import {
  Award,
  BarChart2,
  GitFork,
  Workflow,
  Cpu,
  TrendingUp,
  Scale,
  AlertOctagon,
  AlertTriangle,
} from 'lucide-react';

import { ResearchOverview } from './ResearchOverview';
import { ResearchEDA } from './ResearchEDA';
import { ResearchSplits } from './ResearchSplits';
import { ResearchPipeline } from './ResearchPipeline';
import { ResearchModels } from './ResearchModels';
import { ResearchResults } from './ResearchResults';
import { ResearchComparisons } from './ResearchComparisons';
import { ResearchImbalance } from './ResearchImbalance';
import { ResearchLimitations } from './ResearchLimitations';

type SubTab =
  | 'overview'
  | 'eda'
  | 'splits'
  | 'pipeline'
  | 'models'
  | 'results'
  | 'comparisons'
  | 'imbalance'
  | 'limitations';

export const ResearchTab: React.FC = () => {
  const [subTab, setSubTab] = useState<SubTab>('overview');

  const subTabs = [
    { id: 'overview', label: 'Benchmark', icon: Award },
    { id: 'eda', label: 'EDA & Findings', icon: BarChart2 },
    { id: 'splits', label: 'Deduplicated Splits', icon: GitFork },
    { id: 'pipeline', label: 'Architecture Pipeline', icon: Workflow },
    { id: 'models', label: 'Model Tuning', icon: Cpu },
    { id: 'results', label: 'Test Results (CIs)', icon: TrendingUp },
    { id: 'comparisons', label: 'Significance Tests', icon: Scale },
    { id: 'imbalance', label: 'Imbalance & Policy', icon: AlertOctagon },
    { id: 'limitations', label: 'Limitations', icon: AlertTriangle },
  ];

  return (
    <div className="space-y-6">
      {/* Sub-navigation bar */}
      <div className="overflow-x-auto pb-2 scrollbar-none">
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900/80 border border-slate-800 w-max min-w-full">
          {subTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = subTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => setSubTab(tab.id as SubTab)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon size={14} className={isActive ? 'text-emerald-400' : 'text-slate-400'} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* View Container */}
      <div className="min-h-[600px]">
        {subTab === 'overview' && <ResearchOverview />}
        {subTab === 'eda' && <ResearchEDA />}
        {subTab === 'splits' && <ResearchSplits />}
        {subTab === 'pipeline' && <ResearchPipeline />}
        {subTab === 'models' && <ResearchModels />}
        {subTab === 'results' && <ResearchResults />}
        {subTab === 'comparisons' && <ResearchComparisons />}
        {subTab === 'imbalance' && <ResearchImbalance />}
        {subTab === 'limitations' && <ResearchLimitations />}
      </div>
    </div>
  );
};
