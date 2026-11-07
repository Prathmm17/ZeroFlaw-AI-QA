import React, { useEffect, useState } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  Camera,
  CheckCircle2,
  Clock,
  Cpu,
  Film,
  History,
  Layers,
  LayoutDashboard,
  Shield,
  Video,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemHealth } from '../types';

interface LayoutProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, activeTab, onTabChange }) => {
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    apiClient
      .getHealth()
      .then(setHealth)
      .catch((err) => console.error('Failed to fetch system health:', err));
  }, []);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'inspector', label: 'Image Inspector', icon: Camera },
    { id: 'video', label: 'Video Inspector', icon: Film },
    { id: 'stream', label: 'Live Stream', icon: Video },
    { id: 'history', label: 'Inspection Audit', icon: History },
    { id: 'analytics', label: 'Quality Analytics', icon: BarChart3 },
    { id: 'glossary', label: 'Defect Glossary', icon: BookOpen },
  ];

  return (
    <div className="min-h-screen bg-[#06040c] text-slate-100 flex flex-col relative selection:bg-violet-500/40 selection:text-white">
      {/* Exact Electric Violet Diagonal Aurora Light Streak (Replicated from user reference image) */}
      {/* 1. Upper-Right Radiant Violet Flare Hotspot */}
      <div className="fixed -top-12 right-[8%] w-[38rem] h-[38rem] rounded-full bg-violet-600/35 blur-[110px] pointer-events-none -z-0" />
      <div className="fixed top-[4%] right-[14%] w-[22rem] h-[22rem] rounded-full bg-purple-500/45 blur-[75px] pointer-events-none -z-0" />

      {/* 2. Main Diagonal Electric Purple Light Beam running from upper-right down across center */}
      <div className="fixed top-[14%] -left-[10%] w-[130%] h-[28rem] -rotate-[36deg] bg-gradient-to-r from-violet-950/20 via-violet-600/35 to-purple-500/45 blur-[85px] pointer-events-none -z-0" />

      {/* 3. Mid-Center Deep Indigo / Purple Ambient Dissolve */}
      <div className="fixed top-[40%] left-[15%] w-[42rem] h-[42rem] rounded-full bg-indigo-700/15 blur-[140px] pointer-events-none -z-0" />

      {/* 4. Bottom Deep Obsidian Pitch Base */}
      <div className="fixed -bottom-28 left-[5%] w-[65rem] h-[35rem] rounded-full bg-black/95 blur-[130px] pointer-events-none -z-0" />

      {/* Top Glassmorphic Navigation Bar */}
      <header className="sticky top-0 z-50 bg-[#0a0716]/65 backdrop-blur-3xl border-b border-violet-500/20 shadow-[0_8px_32px_rgba(0,0,0,0.65)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Brand Logo & Title */}
            <div className="flex items-center space-x-3 cursor-pointer group" onClick={() => onTabChange('dashboard')}>
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-violet-500 via-purple-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-600/40 ring-1 ring-violet-300/40 transition-transform duration-300 group-hover:scale-105">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-extrabold tracking-tight text-white text-lg drop-shadow-sm">ZeroFlaw</span>
                  <span className="text-[11px] px-2.5 py-0.5 rounded-full font-mono font-bold bg-violet-500/20 text-violet-200 border border-violet-400/35 shadow-inner backdrop-blur-md">
                    AI-QA
                  </span>
                </div>
                <p className="text-[11px] text-violet-200/70 font-medium">Real-Time Industrial Defect Detection</p>
              </div>
            </div>

            {/* Navigation Tabs - Violet Aurora Capsule Bar */}
            <nav className="hidden md:flex items-center p-1 rounded-full bg-white/[0.04] border border-white/[0.1] backdrop-blur-2xl shadow-[inset_0_1px_1px_rgba(255,255,255,0.1)]">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onTabChange(item.id)}
                    className={`flex items-center space-x-2 px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all duration-200 ${
                      isActive
                        ? 'bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 text-white shadow-lg shadow-violet-600/45 ring-1 ring-violet-300/50 font-bold'
                        : 'text-violet-200/70 hover:text-white hover:bg-white/[0.08]'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>

            {/* System Status Pill - Frosted Glass with Violet Accent */}
            <div className="flex items-center space-x-3">
              <div className="hidden sm:flex items-center space-x-2 bg-[#120d24]/80 px-3.5 py-1.5 rounded-full border border-violet-500/30 text-xs font-mono backdrop-blur-2xl shadow-inner">
                <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse" />
                <span className="text-violet-300/70">AI:</span>
                <span className="text-white font-semibold">{health?.model_name || 'best.pt'}</span>
                <span className="text-violet-400/40">|</span>
                <Cpu className="w-3.5 h-3.5 text-violet-400" />
                <span className="text-white/90 uppercase">{health?.device || 'CPU'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="md:hidden flex overflow-x-auto px-4 py-2 border-t border-violet-500/20 gap-2 bg-[#0a0716]/90 backdrop-blur-2xl">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex shrink-0 items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/40 font-bold'
                    : 'text-violet-200/70 bg-white/[0.04] border border-white/[0.08]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 relative z-10">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] bg-slate-950/50 backdrop-blur-xl py-4 text-center text-xs text-slate-500 font-mono relative z-10">
        <p>ZeroFlaw AI-QA &bull; Industrial Edge Quality Assurance System &bull; Powered by YOLOv8 &amp; FastAPI</p>
      </footer>
    </div>
  );
};

