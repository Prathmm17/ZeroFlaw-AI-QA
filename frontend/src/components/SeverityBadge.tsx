import React from 'react';
import { SeverityLevel, InspectionStatus } from '../types';

interface SeverityBadgeProps {
  level: SeverityLevel | InspectionStatus;
  size?: 'sm' | 'md' | 'lg';
}

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({ level, size = 'md' }) => {
  const norm = level.toLowerCase();

  let styles = 'bg-slate-800 text-slate-300 border-slate-700';
  let dotColor = 'bg-slate-400';
  let label = level.toUpperCase();

  if (norm === 'critical' || norm === 'rejected') {
    styles = 'bg-rose-950/70 text-rose-300 border-rose-800/80 shadow-rose-950/50 shadow-sm';
    dotColor = 'bg-rose-500 animate-pulse';
    label = norm === 'rejected' ? 'REJECTED' : 'CRITICAL';
  } else if (norm === 'major' || norm === 'flagged') {
    styles = 'bg-amber-950/70 text-amber-300 border-amber-800/80 shadow-amber-950/50 shadow-sm';
    dotColor = 'bg-amber-500';
    label = norm === 'flagged' ? 'FLAGGED' : 'MAJOR';
  } else if (norm === 'minor') {
    styles = 'bg-yellow-950/50 text-yellow-300 border-yellow-800/60';
    dotColor = 'bg-yellow-400';
    label = 'MINOR';
  } else if (norm === 'passed' || norm === 'none') {
    styles = 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60';
    dotColor = 'bg-emerald-400';
    label = norm === 'passed' ? 'PASSED' : 'NORMAL';
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 space-x-1.5',
    md: 'text-xs font-medium px-2.5 py-1 space-x-2',
    lg: 'text-sm font-semibold px-3 py-1.5 space-x-2',
  }[size];

  return (
    <span
      className={`inline-flex items-center rounded-full border font-mono tracking-wider ${styles} ${sizeClasses}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      <span>{label}</span>
    </span>
  );
};

