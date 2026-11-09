import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Camera,
  CheckCircle,
  Clock,
  Layers,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { apiClient, isColdStartError, resolveMediaUrl, FALLBACK_IMAGE_DATA_URI } from '../api/client';
import { AlertTicker } from '../components/AlertTicker';
import { SeverityBadge } from '../components/SeverityBadge';
import { StatsCard } from '../components/StatsCard';
import { AnalyticsOverview, Inspection, QualityAlert } from '../types';

const INITIAL_SAMPLES = [
  'benchmark_2048_hairline.png',
  'sample_crack_000.png',
  'sample_dent_002.png',
  'sample_scratch_001.png',
  'sample_chip_004.png',
  'sample_stain_003.png',
];

interface DashboardPageProps {
  onNavigate: (tab: string) => void;
  onSelectInspection: (inspection: Inspection) => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  onNavigate,
  onSelectInspection,
}) => {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [recentInspections, setRecentInspections] = useState<Inspection[]>([]);
  const [alerts, setAlerts] = useState<QualityAlert[]>([]);
  const [samples, setSamples] = useState<string[]>(INITIAL_SAMPLES);
  const [inspectingSample, setInspectingSample] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isWakingUp, setIsWakingUp] = useState(false);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      setIsWakingUp(false);

      const [overviewRes, historyRes, alertsRes, samplesRes] = await Promise.allSettled([
        apiClient.getAnalyticsOverview(),
        apiClient.getHistory(1, 5),
        apiClient.getAlerts(false),
        apiClient.getSamples(),
      ]);

      let anySuccess = false;
      if (overviewRes.status === 'fulfilled') {
        setAnalytics(overviewRes.value);
        anySuccess = true;
      }
      if (historyRes.status === 'fulfilled') {
        setRecentInspections(historyRes.value.items || []);
        anySuccess = true;
      }
      if (alertsRes.status === 'fulfilled') {
        setAlerts(alertsRes.value || []);
        anySuccess = true;
      }
      if (samplesRes.status === 'fulfilled') {
        setSamples((samplesRes.value || []).slice(0, 6));
        anySuccess = true;
      }

      if (!anySuccess) {
        const reason = (overviewRes as PromiseRejectedResult).reason;
        if (isColdStartError(reason)) {
          setIsWakingUp(true);
          setErrorMsg(
            'The AI backend is warming up from sleep (Render free tier cold start ~30–50s). Click Retry to reload.'
          );
        } else {
          setErrorMsg('Unable to retrieve dashboard telemetry from AI server. Please retry.');
        }
      }
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      setErrorMsg('Failed to load dashboard telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const handleQuickInspect = async (sampleName: string) => {
    try {
      setInspectingSample(sampleName);
      const res = await apiClient.inspectSample(sampleName);
      onSelectInspection(res);
      onNavigate('inspector');
    } catch (err) {
      console.error('Quick inspect failed:', err);
    } finally {
      setInspectingSample(null);
    }
  };

  const yieldRate =
    analytics && analytics.total_inspections > 0
      ? (
          ((analytics.defect_free_inspections || 0) /
            analytics.total_inspections) *
          100
        ).toFixed(1)
      : '100.0';

  return (
    <div className="space-y-6">
      {/* Real-time Alert Ticker */}
      <AlertTicker
        alerts={alerts}
        onAlertAcknowledged={(id) => {
          setAlerts((prev) => prev.filter((a) => a.id !== id));
        }}
      />

      {/* Cold Start / Error Notification Banner */}
      {errorMsg && (
        <div className="bg-amber-950/70 border border-amber-500/50 p-4 rounded-2xl text-xs text-amber-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 backdrop-blur-xl shadow-lg">
          <div className="flex items-center space-x-2.5">
            <AlertTriangle className={`w-4 h-4 shrink-0 text-amber-400 ${isWakingUp ? 'animate-spin' : ''}`} />
            <span>{errorMsg}</span>
          </div>
          <button
            onClick={() => loadDashboardData()}
            disabled={loading}
            className="self-end sm:self-auto px-4 py-1.5 rounded-full bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-400/40 text-xs font-semibold flex items-center space-x-1.5 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>{loading ? 'Connecting...' : 'Retry Now'}</span>
          </button>
        </div>
      )}

      {/* Header Bar with Manual Refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Factory Quality Operations Center</h2>
          <p className="text-xs text-slate-400 mt-0.5">Real-time edge inspection telemetry &amp; defect capability</p>
        </div>
        <button
          onClick={() => loadDashboardData()}
          disabled={loading}
          className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-xs text-slate-300 hover:text-white transition-all backdrop-blur-xl disabled:opacity-50"
          title="Refresh metrics from Render API"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>{loading ? 'Refreshing...' : 'Refresh Metrics'}</span>
        </button>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Overall Line Yield"
          value={loading && !analytics ? '...' : `${yieldRate}%`}
          subtitle="Target threshold: ≥ 98.0%"
          icon={ShieldCheck}
          variant={parseFloat(yieldRate) >= 95.0 ? 'emerald' : parseFloat(yieldRate) >= 80.0 ? 'amber' : 'rose'}
        />
        <StatsCard
          title="Total Inspections"
          value={loading && !analytics ? '...' : analytics?.total_inspections || 0}
          subtitle={`${analytics?.defect_free_inspections || 0} defect-free components`}
          icon={Activity}
          variant="blue"
        />
        <StatsCard
          title="Defects Detected"
          value={loading && !analytics ? '...' : analytics?.total_defects || 0}
          subtitle={`Defect rate: ${analytics?.defect_rate_percentage || 0}%`}
          icon={AlertTriangle}
          variant={analytics && analytics.critical_defects_count > 0 ? 'rose' : 'amber'}
        />
        <StatsCard
          title="Inference Speed"
          value={loading && !analytics ? '...' : `${analytics?.avg_inference_time_ms.toFixed(0) || 0} ms`}
          subtitle="Real-time edge throughput"
          icon={Zap}
          variant="blue"
        />
      </div>

      {/* Quick Inspection Launchpad */}
      <div className="glass-panel rounded-3xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white flex items-center space-x-2">
              <span>Instant Factory Sample Inspector</span>
              <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-gradient-to-r from-violet-500/25 to-purple-500/25 text-violet-200 border border-violet-400/35">
                YOLOv8 Edge
              </span>
            </h3>
            <p className="text-xs text-violet-200/70 mt-0.5">
              Select any demo specimen from the manufacturing line for instant YOLOv8 defect inference
            </p>
          </div>
          <button
            onClick={() => onNavigate('inspector')}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.15] text-xs font-semibold text-white transition-all shadow-inner"
          >
            <span>Custom Upload Studio</span>
            <ArrowRight className="w-3.5 h-3.5 text-violet-400" />
          </button>
        </div>

        {samples.length === 0 ? (
          <div className="py-10 text-center glass-panel rounded-2xl border border-white/[0.08] p-4 text-xs text-slate-400 font-mono space-y-2">
            <p>{loading ? 'Connecting to AI Engine on Render...' : 'Catalog samples loading...'}</p>
            {!loading && (
              <button
                onClick={() => loadDashboardData()}
                className="px-3 py-1 rounded-full bg-white/[0.05] hover:bg-white/[0.1] text-sky-400 text-xs font-semibold"
              >
                Reload Samples
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3.5">
            {samples.map((sample) => (
              <button
                key={sample}
                onClick={() => handleQuickInspect(sample)}
                disabled={inspectingSample === sample}
                className={`p-3 rounded-2xl border text-left transition-all duration-300 ${
                  inspectingSample === sample
                    ? 'bg-violet-600/30 border-violet-400/60 ring-2 ring-violet-400/50 shadow-lg shadow-violet-500/30'
                    : 'bg-white/[0.03] border-white/[0.08] hover:border-violet-400/40 hover:bg-[#140e28]/70 hover:shadow-xl hover:shadow-violet-500/20 hover:-translate-y-0.5'
                }`}
              >
                <div className="aspect-square rounded-xl bg-black/50 overflow-hidden mb-2.5 flex items-center justify-center ring-1 ring-white/10">
                  <img
                    src={resolveMediaUrl(`/data/samples/${sample}`)}
                    alt={sample}
                    loading="lazy"
                    onError={(e) => {
                      const img = e.currentTarget;
                      img.onerror = null;
                      img.src = FALLBACK_IMAGE_DATA_URI;
                    }}
                    className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
                  />
                </div>
                <p className="text-[11px] font-mono text-slate-200 truncate font-medium">{sample}</p>
                <span className="text-[10px] text-violet-300 font-semibold mt-1 block">
                  {inspectingSample === sample ? 'Analyzing...' : 'Click to Inspect →'}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Recent Inspections Table */}
      <div className="glass-panel rounded-3xl overflow-hidden">
        <div className="p-4.5 px-6 border-b border-white/[0.08] flex items-center justify-between">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white">
            Recent QA Inspection Log
          </h3>
          <button
            onClick={() => onNavigate('history')}
            className="text-xs font-semibold text-white flex items-center space-x-1 px-3.5 py-1.5 rounded-full bg-white/[0.06] border border-white/[0.15] hover:bg-white/[0.12] transition-all"
          >
            <span>View All Records</span>
            <ArrowRight className="w-3.5 h-3.5 text-violet-400" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-white/[0.04] text-white/70 font-mono border-b border-white/[0.08]">
              <tr>
                <th className="py-3.5 px-4 font-semibold">ID</th>
                <th className="py-3.5 px-4 font-semibold">TIMESTAMP</th>
                <th className="py-3.5 px-4 font-semibold">SPECIMEN</th>
                <th className="py-3.5 px-4 font-semibold">SOURCE</th>
                <th className="py-3.5 px-4 font-semibold">DEFECTS</th>
                <th className="py-3.5 px-4 font-semibold">LATENCY</th>
                <th className="py-3.5 px-4 font-semibold">STATUS</th>
                <th className="py-3.5 px-4 text-right font-semibold">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05] font-mono">
              {recentInspections.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-400 font-sans">
                    No inspections recorded yet. Run a sample or upload an image above.
                  </td>
                </tr>
              ) : (
                recentInspections.map((insp) => (
                  <tr key={insp.id} className="hover:bg-white/[0.06] transition-colors">
                    <td className="py-3 px-4 text-white/60">#{insp.id}</td>
                    <td className="py-3 px-4 text-white/90">
                      {new Date(insp.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-3 px-4 text-white font-medium">{insp.filename}</td>
                    <td className="py-3 px-4 text-white/60 uppercase">{insp.source}</td>
                    <td className="py-3 px-4">
                      {insp.total_defects > 0 ? (
                        <span className="text-orange-300 font-bold">{insp.total_defects} found</span>
                      ) : (
                        <span className="text-emerald-300 font-medium">None</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-white/70">{insp.inference_time_ms.toFixed(0)} ms</td>
                    <td className="py-3 px-4">
                      <SeverityBadge level={insp.status} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => {
                          onSelectInspection(insp);
                          onNavigate('inspector');
                        }}
                        className="text-xs text-pink-300 hover:text-pink-200 font-semibold"
                      >
                        Inspect →
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

