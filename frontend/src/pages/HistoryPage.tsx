import React, { useEffect, useState } from 'react';
import {
  Calendar,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Eye,
  Filter,
  History,
  Search,
  X,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { DefectViewer } from '../components/DefectViewer';
import { SeverityBadge } from '../components/SeverityBadge';
import { Inspection } from '../types';

interface HistoryPageProps {
  onSelectInspection?: (inspection: Inspection) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ onSelectInspection }) => {
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [statusFilter, setStatusFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [selectedInspection, setSelectedInspection] = useState<Inspection | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const res = await apiClient.getHistory(page, 15, statusFilter, severityFilter);
      setInspections(res.items);
      setTotal(res.total);
      setPages(res.pages);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, statusFilter, severityFilter]);

  // Handle ESC key to dismiss modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedInspection(null);
      }
    };
    if (selectedInspection) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedInspection]);

  return (
    <div className="space-y-6">
      {/* Header & Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Manufacturing Inspection Audit Trail</h2>
          <p className="text-xs text-slate-400 mt-1">
            Complete database of inspected components, defect telemetry, and operator decisions.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Status Filter */}
          <div className="flex items-center space-x-1 bg-white/[0.04] border border-white/[0.08] rounded-full p-1 text-xs font-medium backdrop-blur-xl">
            <span className="text-slate-400 px-2.5 flex items-center space-x-1.5 font-medium">
              <Filter className="w-3.5 h-3.5 text-sky-400" />
              <span>Status:</span>
            </span>
            {['all', 'passed', 'flagged', 'rejected'].map((st) => (
              <button
                key={st}
                onClick={() => {
                  setStatusFilter(st);
                  setPage(1);
                }}
                className={`px-3 py-1 rounded-full capitalize transition-all ${
                  statusFilter === st
                    ? 'bg-sky-500/25 text-sky-200 border border-sky-400/40 shadow-[0_0_12px_rgba(56,189,248,0.25)] font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {st}
              </button>
            ))}
          </div>

          {/* Severity Filter */}
          <div className="flex items-center space-x-1 bg-white/[0.04] border border-white/[0.08] rounded-full p-1 text-xs font-medium backdrop-blur-xl">
            <span className="text-slate-400 px-2.5 font-medium">Severity:</span>
            {['all', 'critical', 'major', 'minor'].map((sev) => (
              <button
                key={sev}
                onClick={() => {
                  setSeverityFilter(sev);
                  setPage(1);
                }}
                className={`px-3 py-1 rounded-full capitalize transition-all ${
                  severityFilter === sev
                    ? 'bg-sky-500/25 text-sky-200 border border-sky-400/40 shadow-[0_0_12px_rgba(56,189,248,0.25)] font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-white/[0.08]">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-white/[0.03] text-slate-400 font-mono border-b border-white/[0.06]">
              <tr>
                <th className="py-3.5 px-4 font-semibold">RECORD ID</th>
                <th className="py-3.5 px-4 font-semibold">TIMESTAMP</th>
                <th className="py-3.5 px-4 font-semibold">COMPONENT</th>
                <th className="py-3.5 px-4 font-semibold">SOURCE</th>
                <th className="py-3.5 px-4 font-semibold">DEFECT COUNT</th>
                <th className="py-3.5 px-4 font-semibold">MAX SEVERITY</th>
                <th className="py-3.5 px-4 font-semibold">LATENCY</th>
                <th className="py-3.5 px-4 font-semibold">OUTCOME</th>
                <th className="py-3.5 px-4 text-right font-semibold">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] font-mono">
              {loading ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-slate-400 font-sans">
                    Loading audit records...
                  </td>
                </tr>
              ) : inspections.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-slate-500 font-sans">
                    No inspection records found matching criteria.
                  </td>
                </tr>
              ) : (
                inspections.map((insp) => (
                  <tr key={insp.id} className="hover:bg-slate-850/60 transition-colors">
                    <td className="py-3 px-4 text-slate-400">#{insp.id}</td>
                    <td className="py-3 px-4 text-slate-300">
                      {new Date(insp.timestamp).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-white font-medium">{insp.filename}</td>
                    <td className="py-3 px-4 text-slate-400 uppercase">{insp.source}</td>
                    <td className="py-3 px-4">
                      {insp.total_defects > 0 ? (
                        <span className="text-amber-400 font-bold">{insp.total_defects} defect(s)</span>
                      ) : (
                        <span className="text-emerald-400">0 defects</span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <SeverityBadge level={insp.max_severity} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-slate-400">{insp.inference_time_ms.toFixed(0)} ms</td>
                    <td className="py-3 px-4">
                      <SeverityBadge level={insp.status} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => setSelectedInspection(insp)}
                        className="inline-flex items-center space-x-1 text-xs text-sky-400 hover:text-sky-300 font-semibold"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="p-4 border-t border-white/[0.06] flex items-center justify-between text-xs text-slate-400">
          <span>
            Showing page <strong>{page}</strong> of <strong>{pages}</strong> ({total} total inspections)
          </span>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2 font-mono font-bold text-white">{page}</span>
            <button
              onClick={() => setPage((p) => Math.min(pages, p + 1))}
              disabled={page >= pages}
              className="p-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Modal Dialog for Inspection Inspection Detail */}
      {selectedInspection && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelectedInspection(null);
          }}
          className="fixed inset-0 z-50 bg-[#06040c]/85 backdrop-blur-3xl flex items-center justify-center p-3 sm:p-6"
        >
          <div className="glass-panel border border-violet-500/25 rounded-3xl max-w-6xl w-full max-h-[92vh] flex flex-col shadow-[0_25px_60px_rgba(0,0,0,0.8)] overflow-hidden">
            {/* Sticky Top Header */}
            <div className="flex items-center justify-between border-b border-violet-500/20 px-6 py-4 bg-[#0d091a]/85 backdrop-blur-3xl flex-shrink-0">
              <div className="flex items-center space-x-3">
                <div className="w-9 h-9 rounded-xl bg-sky-500/10 border border-sky-400/20 flex items-center justify-center text-sky-400 font-mono font-bold text-xs shadow-[0_0_12px_rgba(56,189,248,0.2)]">
                  #{selectedInspection.id}
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                    <span>Inspection Audit Detail</span>
                    <span className="text-xs font-mono text-slate-400 font-normal truncate max-w-xs sm:max-w-md">
                      ({selectedInspection.filename})
                    </span>
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    Inspected on {new Date(selectedInspection.timestamp).toLocaleString()} &bull; Source: {selectedInspection.source.toUpperCase()}
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-2">
                {onSelectInspection && (
                  <button
                    onClick={() => {
                      const target = selectedInspection;
                      setSelectedInspection(null);
                      onSelectInspection(target);
                    }}
                    className="px-3.5 py-1.5 rounded-full bg-sky-500/20 hover:bg-sky-500/30 text-sky-200 text-xs font-semibold flex items-center space-x-1.5 transition-all border border-sky-400/30 shadow-[0_0_12px_rgba(56,189,248,0.2)]"
                    title="Open full interactive analysis in Image Inspector tab"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Open in Image Inspector</span>
                  </button>
                )}
                <a
                  href={`/api/detect/${selectedInspection.id}/report/view`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3.5 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-slate-200 text-xs font-medium flex items-center space-x-1.5 border border-white/[0.08] transition-all"
                  title="Open ISO 9001 Inspection Certificate in browser"
                >
                  <ExternalLink className="w-3.5 h-3.5 text-sky-400" />
                  <span className="hidden sm:inline">Certificate</span>
                </a>
                <button
                  onClick={() => setSelectedInspection(null)}
                  className="p-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-slate-400 hover:text-white transition-all border border-white/[0.08] ml-1"
                  title="Close (ESC)"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Scrollable Content Body */}
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
              <DefectViewer inspection={selectedInspection} />
            </div>

            {/* Sticky Bottom Footer */}
            <div className="flex items-center justify-between border-t border-white/[0.08] px-6 py-3.5 bg-slate-950/60 backdrop-blur-2xl flex-shrink-0 text-xs text-slate-400">
              <span className="font-mono text-[11px] text-slate-500">
                Press <kbd className="px-2 py-0.5 rounded-md bg-white/[0.05] border border-white/[0.1] text-slate-300">ESC</kbd> or click backdrop to close
              </span>
              <button
                onClick={() => setSelectedInspection(null)}
                className="px-4 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-slate-200 font-medium transition-all border border-white/[0.08] text-xs"
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

