import React, { useState, useEffect, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { ModerateTab } from './components/ModerateTab';
import { ResearchTab } from './components/research/ResearchTab';
import type {
  HealthResponse,
  ModelInfo,
  CommentRecord,
  PredictResponse,
} from './types/api';
import {
  fetchHealth,
  fetchModels,
  fetchComments,
} from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'moderate' | 'research'>('moderate');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelsInfo, setModelsInfo] = useState<ModelInfo[]>([]);

  // Comments history state
  const [comments, setComments] = useState<CommentRecord[]>([]);
  const [totalComments, setTotalComments] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [isLoadingHistory, setIsLoadingHistory] = useState<boolean>(false);

  // Global profanity masking toggle
  const [globalCensor, setGlobalCensor] = useState<boolean>(true);

  // Load health & models info
  const loadSystemInfo = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchModels().catch(() => []),
      ]);
      if (h) setHealth(h);
      if (m) setModelsInfo(m);
    } catch (err) {
      console.error('Failed to load system info', err);
    }
  }, []);

  // Load comments
  const loadComments = useCallback(
    async (page = 1, filter = 'all') => {
      setIsLoadingHistory(true);
      try {
        const res = await fetchComments(page, 10, filter);
        setComments(res.items || []);
        setTotalComments(res.total || 0);
        setCurrentPage(res.page || 1);
        setTotalPages(res.total_pages || 1);
      } catch (err) {
        console.error('Failed to fetch comments', err);
      } finally {
        setIsLoadingHistory(false);
      }
    },
    []
  );

  useEffect(() => {
    loadSystemInfo();
    loadComments(1, activeFilter);

    // Refresh health every 20 seconds
    const interval = setInterval(loadSystemInfo, 20000);
    return () => clearInterval(interval);
  }, [loadSystemInfo, loadComments, activeFilter]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    loadComments(page, activeFilter);
  };

  const handleFilterChange = (filter: string) => {
    setActiveFilter(filter);
    setCurrentPage(1);
    loadComments(1, filter);
  };

  const handleNewPrediction = (pred: PredictResponse) => {
    // Optimistically update comments history
    setComments((prev) => [pred, ...prev]);
    setTotalComments((prev) => prev + 1);
  };

  const handleToggleGlobalCensor = () => {
    setGlobalCensor((prev) => !prev);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-white">
      {/* Sticky Navigation */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        health={health}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'moderate' ? (
          <ModerateTab
            comments={comments}
            totalComments={totalComments}
            currentPage={currentPage}
            totalPages={totalPages}
            activeFilter={activeFilter}
            onPageChange={handlePageChange}
            onFilterChange={handleFilterChange}
            onNewPrediction={handleNewPrediction}
            globalCensor={globalCensor}
            onToggleGlobalCensor={handleToggleGlobalCensor}
            modelsInfo={modelsInfo}
            isLoadingHistory={isLoadingHistory}
          />
        ) : (
          <ResearchTab />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">Welcome to Civilization v2</span>
            <span>•</span>
            <span>Multi-Label Jigsaw Toxicity Moderation Platform</span>
          </div>
          <div className="flex items-center gap-4 text-slate-500 text-[11px] font-mono">
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>MongoDB Atlas</span>
            <span>•</span>
            <span>PyTorch 2.6 CUDA</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
