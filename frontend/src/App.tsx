import React, { useState } from 'react';
import { Layout } from './components/Layout';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { DashboardPage } from './pages/DashboardPage';
import { HistoryPage } from './pages/HistoryPage';
import { InspectorPage } from './pages/InspectorPage';
import { LiveStreamPage } from './pages/LiveStreamPage';
import { VideoInspectionPage } from './pages/VideoInspectionPage';
import { GlossaryPage } from './pages/GlossaryPage';
import { Inspection } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [currentInspection, setCurrentInspection] = useState<Inspection | null>(null);
  const [historyKey, setHistoryKey] = useState<number>(0);

  const handleSelectInspection = (inspection: Inspection) => {
    setCurrentInspection(inspection);
    setActiveTab('inspector');
  };

  const handleTabChange = (tab: string) => {
    if (tab === 'history') {
      setHistoryKey((k) => k + 1);
    }
    setActiveTab(tab);
  };

  return (
    <Layout activeTab={activeTab} onTabChange={handleTabChange}>
      {activeTab === 'dashboard' && (
        <DashboardPage
          onNavigate={setActiveTab}
          onSelectInspection={handleSelectInspection}
        />
      )}
      {activeTab === 'inspector' && (
        <InspectorPage
          currentInspection={currentInspection}
          onInspectionLoaded={setCurrentInspection}
        />
      )}
      {activeTab === 'video' && <VideoInspectionPage />}
      {activeTab === 'stream' && (
        <LiveStreamPage onSelectInspection={handleSelectInspection} />
      )}
      {activeTab === 'history' && (
        <HistoryPage key={historyKey} onSelectInspection={handleSelectInspection} />
      )}
      {activeTab === 'analytics' && <AnalyticsPage />}
      {activeTab === 'glossary' && <GlossaryPage />}
    </Layout>
  );
}

export default App;

