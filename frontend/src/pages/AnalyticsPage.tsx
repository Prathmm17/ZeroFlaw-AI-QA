import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  CartesianGrid,
  Legend,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  Award,
  BarChart3,
  CheckCircle,
  Clock,
  Layers,
  PieChart as PieIcon,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { StatsCard } from '../components/StatsCard';
import { AnalyticsOverview, AnalyticsTrends } from '../types';

export const AnalyticsPage: React.FC = () => {
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [trends, setTrends] = useState<AnalyticsTrends | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const [ovRes, trRes] = await Promise.allSettled([
        apiClient.getAnalyticsOverview(),
        apiClient.getAnalyticsTrends(7),
      ]);
      if (ovRes.status === 'fulfilled') setOverview(ovRes.value);
      if (trRes.status === 'fulfilled') setTrends(trRes.value);
      if (ovRes.status === 'rejected' && trRes.status === 'rejected') {
        setErrorMsg('Failed to load quality analytics from AI engine. Please retry.');
      }
    } catch (err) {
      console.error('Failed to load analytics:', err);
      setErrorMsg('Failed to load analytics telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const SEVERITY_COLORS = {
    critical: '#f43f5e', // Rose
    major: '#f59e0b',    // Amber
    minor: '#eab308',    // Yellow
  };

  const STATUS_COLORS = {
    passed: '#10b981',   // Emerald
    flagged: '#f59e0b',  // Amber
    rejected: '#f43f5e', // Rose
  };

  // Format defect types for BarChart
  const defectTypeData = overview
    ? Object.entries(overview.defect_types_distribution).map(([name, count]) => ({
        name: name.toUpperCase(),
        count,
        cumulative: overview.pareto_cumulative?.[name] ?? 0,
      }))
    : [];

  // Format severity for PieChart
  const severityPieData = overview
    ? [
        { name: 'Critical', value: overview.critical_defects_count, color: SEVERITY_COLORS.critical },
        { name: 'Major', value: overview.major_defects_count, color: SEVERITY_COLORS.major },
        { name: 'Minor', value: overview.minor_defects_count, color: SEVERITY_COLORS.minor },
      ].filter((d) => d.value > 0)
    : [];

  // Format status for PieChart
  const statusPieData = overview
    ? [
        { name: 'Passed', value: overview.status_distribution.passed || 0, color: STATUS_COLORS.passed },
        { name: 'Flagged', value: overview.status_distribution.flagged || 0, color: STATUS_COLORS.flagged },
        { name: 'Rejected', value: overview.status_distribution.rejected || 0, color: STATUS_COLORS.rejected },
      ].filter((d) => d.value > 0)
    : [];

  const yieldRate =
    overview && overview.total_inspections > 0
      ? (
          ((overview.total_inspections - (overview.critical_defects_count + overview.major_defects_count)) /
            overview.total_inspections) *
          100
        ).toFixed(1)
      : '100.0';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Executive Quality Intelligence &amp; Analytics</h2>
          <p className="text-xs text-slate-400 mt-1">
            Statistical process control (SPC), defect Pareto distributions, and continuous manufacturing yield trends.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="self-start sm:self-auto flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-xs text-slate-300 hover:text-white transition-all backdrop-blur-xl disabled:opacity-50"
        >
          <span>{loading ? 'Refreshing...' : 'Refresh Telemetry'}</span>
        </button>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/60 border border-rose-500/40 p-4 rounded-2xl text-xs text-rose-200 flex items-center justify-between">
          <span>{errorMsg}</span>
          <button
            onClick={loadData}
            className="px-3 py-1 rounded-full bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 font-semibold text-xs border border-rose-400/30"
          >
            Retry
          </button>
        </div>
      )}

      {/* Top Level KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="First Pass Yield"
          value={`${yieldRate}%`}
          subtitle="Specification limit ≥ 98%"
          icon={Award}
          variant="emerald"
        />
        <StatsCard
          title="Defect Free Specimen"
          value={overview?.defect_free_inspections || 0}
          subtitle={`Out of ${overview?.total_inspections || 0} total inspected`}
          icon={ShieldCheck}
          variant="blue"
        />
        <StatsCard
          title="Overall Defect Rate"
          value={`${overview?.defect_rate_percentage || 0}%`}
          subtitle={`${overview?.total_defects || 0} total defects flagged`}
          icon={AlertTriangle}
          variant={overview && overview.critical_defects_count > 0 ? 'rose' : 'amber'}
        />
        <StatsCard
          title="Mean Processing Time"
          value={`${overview?.avg_inference_time_ms.toFixed(0) || 0} ms`}
          subtitle="Edge AI YOLOv8 inference"
          icon={Clock}
          variant="blue"
        />
      </div>

      {/* Statistical Process Control (SPC) Capability Banner */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] pb-3">
          <div className="flex items-center space-x-2.5">
            <Activity className="w-5 h-5 text-sky-400" />
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                Statistical Process Control (SPC) &amp; Capability Index
              </h3>
              <p className="text-[11px] text-slate-400">
                IATF 16949 / ISO 9001:2015 Continuous Quality Conformance &bull; Specification Limits: USL = 5.0%, LSL = 0.0%
              </p>
            </div>
          </div>
          {overview?.process_capability_status && (
            <span
              className={`px-3 py-1 rounded-full text-xs font-mono font-bold uppercase border ${
                overview.process_capability_status === 'WORLD_CLASS_SIX_SIGMA'
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-400/30'
                  : overview.process_capability_status === 'CAPABLE_PROCESS'
                  ? 'bg-sky-500/20 text-sky-300 border-sky-400/30'
                  : overview.process_capability_status === 'MARGINALLY_CAPABLE'
                  ? 'bg-amber-500/20 text-amber-300 border-amber-400/30'
                  : 'bg-rose-500/20 text-rose-300 border-rose-400/30'
              }`}
            >
              {overview.process_capability_status.replace(/_/g, ' ')}
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          <div className="bg-white/[0.03] backdrop-blur-xl p-4 rounded-xl border border-white/[0.06] space-y-1">
            <span className="text-slate-400 font-sans text-[11px]">Process Potential (Cp)</span>
            <p className="text-2xl font-bold text-sky-400">
              {overview?.cp != null ? overview.cp.toFixed(2) : '—'}
            </p>
            <span className="text-[10px] text-slate-500 font-sans block">Target ≥ 1.33 (Automotive benchmark)</span>
          </div>

          <div className="bg-white/[0.03] backdrop-blur-xl p-4 rounded-xl border border-white/[0.06] space-y-1">
            <span className="text-slate-400 font-sans text-[11px]">Process Capability (Cpk)</span>
            <p className={`text-2xl font-bold ${
              (overview?.cpk ?? 0) >= 1.33 ? 'text-emerald-400' : (overview?.cpk ?? 0) >= 1.0 ? 'text-amber-400' : 'text-rose-400'
            }`}>
              {overview?.cpk != null ? overview.cpk.toFixed(2) : '—'}
            </p>
            <span className="text-[10px] text-slate-500 font-sans block">Center-adjusted Six-Sigma capability</span>
          </div>

          <div className="bg-white/[0.03] backdrop-blur-xl p-4 rounded-xl border border-white/[0.06] space-y-1">
            <span className="text-slate-400 font-sans text-[11px]">Specification Tolerance Band</span>
            <p className="text-lg font-bold text-slate-200 mt-1">0.0% – 5.0%</p>
            <span className="text-[10px] text-slate-500 font-sans block">USL = 5.0% defect density max</span>
          </div>

          <div className="bg-white/[0.03] backdrop-blur-xl p-4 rounded-xl border border-white/[0.06] space-y-1">
            <span className="text-slate-400 font-sans text-[11px]">Six-Sigma Grade</span>
            <p className="text-lg font-bold text-teal-400 mt-1">
              {(overview?.cpk ?? 0) >= 2.0 ? '6.0 σ' : (overview?.cpk ?? 0) >= 1.67 ? '5.0 σ' : (overview?.cpk ?? 0) >= 1.33 ? '4.0 σ' : '< 3.0 σ'}
            </p>
            <span className="text-[10px] text-slate-500 font-sans block">Zero-Defect Quality Target</span>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Defect Timeline Trend Chart */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-sky-400" />
              <span>7-Day Inspection &amp; Defect Volume Trend</span>
            </h3>
          </div>

          <div className="h-64 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trends?.trends || []}>
                <defs>
                  <linearGradient id="colorInspections" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0284c7" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#0284c7" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="colorDefects" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area
                  type="monotone"
                  dataKey="inspections"
                  stroke="#0284c7"
                  fillOpacity={1}
                  fill="url(#colorInspections)"
                  name="Inspections"
                />
                <Area
                  type="monotone"
                  dataKey="defects"
                  stroke="#f43f5e"
                  fillOpacity={1}
                  fill="url(#colorDefects)"
                  name="Defects Found"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Defect Pareto Type Distribution */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-sky-400" />
              <span>Defect Type Pareto Distribution</span>
            </h3>
          </div>

          <div className="h-64 pt-2">
            {defectTypeData.length === 0 ? (
              <p className="text-xs text-slate-500 text-center py-20 font-sans">
                No defect data yet. Inspect specimens to populate distribution.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={defectTypeData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="count" fill="#0284c7" radius={[4, 4, 0, 0]} name="Defect Count" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {overview?.pareto_cumulative && Object.keys(overview.pareto_cumulative).length > 0 && (
            <div className="pt-3 border-t border-white/[0.06] flex flex-wrap items-center gap-2 text-[11px] font-mono">
              <span className="text-slate-400 font-sans">Cumulative 80/20 Focus:</span>
              {Object.entries(overview.pareto_cumulative).map(([type, pct]) => (
                <span
                  key={type}
                  className={`px-2.5 py-0.5 rounded-full border ${
                    pct <= 80
                      ? 'bg-amber-500/20 text-amber-300 border-amber-400/40 font-bold shadow-[0_0_10px_rgba(245,158,11,0.2)]'
                      : 'bg-white/[0.03] text-slate-400 border-white/[0.06]'
                  }`}
                >
                  {type.toUpperCase()}: {pct.toFixed(1)}%
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Defect Severity Breakdown */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white flex items-center space-x-2">
              <PieIcon className="w-4 h-4 text-sky-400" />
              <span>Defect Severity Classification</span>
            </h3>
          </div>

          <div className="h-64 flex items-center justify-center">
            {severityPieData.length === 0 ? (
              <p className="text-xs text-slate-500 text-center py-20 font-sans">
                Zero defects recorded across inspection history.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={severityPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={85}
                    paddingAngle={4}
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {severityPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Inspection Outcome Yield Distribution */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span>Inspection Outcome Distribution</span>
            </h3>
          </div>

          <div className="h-64 flex items-center justify-center">
            {statusPieData.length === 0 ? (
              <p className="text-xs text-slate-500 text-center py-20 font-sans">
                No inspections recorded yet.
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={85}
                    paddingAngle={4}
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  >
                    {statusPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

