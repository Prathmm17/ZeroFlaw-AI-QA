import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, X } from 'lucide-react';
import { QualityAlert } from '../types';
import { apiClient } from '../api/client';

interface AlertTickerProps {
  alerts: QualityAlert[];
  onAlertAcknowledged?: (alertId: number) => void;
}

export const AlertTicker: React.FC<AlertTickerProps> = ({ alerts, onAlertAcknowledged }) => {
  const [acknowledgingId, setAcknowledgingId] = useState<number | null>(null);

  const activeAlerts = alerts.filter((a) => !a.acknowledged);
  if (activeAlerts.length === 0) return null;

  const topAlert = activeAlerts[0];
  const isCritical = topAlert.severity === 'critical';

  const handleAcknowledge = async (id: number) => {
    try {
      setAcknowledgingId(id);
      await apiClient.acknowledgeAlert(id, 'Console Operator');
      onAlertAcknowledged?.(id);
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    } finally {
      setAcknowledgingId(null);
    }
  };

  return (
    <div
      className={`rounded-3xl border p-4.5 backdrop-blur-3xl transition-all duration-300 shadow-[0_16px_40px_rgba(0,0,0,0.65),inset_0_1px_1px_rgba(255,255,255,0.12)] ${
        isCritical
          ? 'bg-rose-950/40 border-rose-500/40 ring-1 ring-rose-500/30'
          : 'bg-amber-950/40 border-amber-500/40 ring-1 ring-amber-500/30'
      }`}
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3.5">
        <div className="flex items-center space-x-3.5">
          <div
            className={`p-2.5 rounded-2xl shrink-0 backdrop-blur-xl ${
              isCritical
                ? 'bg-rose-500/25 text-rose-200 border border-rose-400/40 shadow-lg shadow-rose-500/20 animate-pulse'
                : 'bg-amber-500/25 text-amber-200 border border-amber-400/40 shadow-lg shadow-amber-500/20'
            }`}
          >
            {isCritical ? <ShieldAlert className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span
                className={`text-[11px] font-mono font-bold tracking-wider uppercase px-2.5 py-0.5 rounded-full border ${
                  isCritical
                    ? 'bg-rose-500/20 text-rose-200 border-rose-400/35'
                    : 'bg-amber-500/20 text-amber-200 border-amber-400/35'
                }`}
              >
                {isCritical ? 'LINE HALT ALERT' : 'QA ROUTE ALERT'}
              </span>
              <span className="text-xs text-violet-200/60 font-mono">
                {new Date(topAlert.timestamp).toLocaleTimeString()}
              </span>
              {activeAlerts.length > 1 && (
                <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-white/[0.08] text-violet-200 border border-white/[0.12]">
                  +{activeAlerts.length - 1} more pending
                </span>
              )}
            </div>
            <p className="text-sm font-semibold text-white mt-1 drop-shadow-sm">{topAlert.message}</p>
            <p className="text-xs text-violet-200/70 mt-0.5 font-medium">{topAlert.action_required}</p>
          </div>
        </div>

        <button
          onClick={() => handleAcknowledge(topAlert.id)}
          disabled={acknowledgingId === topAlert.id}
          className="shrink-0 flex items-center space-x-1.5 px-4 py-2 rounded-full text-xs font-bold transition-all shadow-lg bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white border border-violet-400/40 disabled:opacity-50"
        >
          <CheckCircle className="w-4 h-4 text-emerald-300" />
          <span>{acknowledgingId === topAlert.id ? 'Saving...' : 'Acknowledge & Clear'}</span>
        </button>
      </div>
    </div>
  );
};

