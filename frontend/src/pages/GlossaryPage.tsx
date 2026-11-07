import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  AlertTriangle,
  ShieldAlert,
  CheckCircle2,
  Wrench,
  Eye,
  Info,
  Sparkles,
  Layers,
  ArrowRight,
  Filter,
} from 'lucide-react';
import { DefectKnowledgeItem, SeverityLevel } from '../types';
import { SeverityBadge } from '../components/SeverityBadge';
import { apiClient } from '../api/client';

// Static fallback in case backend API is warming up
const DEFAULT_GLOSSARY: Record<string, DefectKnowledgeItem> = {
  crazing: {
    name: 'crazing',
    display_name: 'Crazing',
    definition: 'Network of fine, interconnected micro-surface fissures in the surface oxide or metal skin.',
    visual_look: 'Web-like, mosaic, or spider-web pattern of shallow hairline fissures on the steel finish.',
    cause: 'Extreme thermal shock during rapid water quench, uneven cooling rates, or excessive surface residual tensile stress.',
    typical_cause: 'Extreme thermal shock during rapid water quench, uneven cooling rates, or excessive surface residual tensile stress.',
    risk: 'Micro-cracks act as stress concentrations that rapidly propagate under cyclic dynamic load, causing catastrophic fatigue fracture.',
    action: 'Lower quench cooling gradient; verify annealing furnace temperature uniformity; relieve surface tensile residual stress.',
    recommended_action: 'Lower quench cooling gradient; verify annealing furnace temperature uniformity; relieve surface tensile residual stress.',
    default_severity: 'critical',
    calibrated_threshold: 0.245,
    confidence_label: '≥ 24.5% conf',
  },
  inclusion: {
    name: 'inclusion',
    display_name: 'Inclusion',
    definition: 'Non-metallic solid foreign matter, entrapped slag particles, or deoxidation oxides within the metal matrix.',
    visual_look: 'Dark, granular, or glassy irregular spots, clusters, or streaks contrasting sharply with the substrate.',
    cause: 'Tundish or ladle refractory lining erosion, re-oxidation during molten steel pouring, or entrained mold flux powder.',
    typical_cause: 'Tundish or ladle refractory lining erosion, re-oxidation during molten steel pouring, or entrained mold flux powder.',
    risk: 'Discontinuities trigger localized stress risers, inducing laminar tearing, internal void growth, and rupture during stamping.',
    action: 'Inspect tundish argon gas shrouding; purge ladle slag; replace worn submerged entry nozzles; verify tundish cleanliness.',
    recommended_action: 'Inspect tundish argon gas shrouding; purge ladle slag; replace worn submerged entry nozzles; verify tundish cleanliness.',
    default_severity: 'major',
    calibrated_threshold: 0.136,
    confidence_label: '≥ 13.6% conf',
  },
  patches: {
    name: 'patches',
    display_name: 'Patches',
    definition: 'Localized superficial surface tears, roll adhesion marks, or oxide crust accumulation on the sheet.',
    visual_look: 'Discontinuous rough darkened or textured patches, irregular scabs, and localized metal build-up.',
    cause: 'Work roll surface spalling or wear, excessive mill chatter, roll lubrication breakdown, or localized pass reduction imbalance.',
    typical_cause: 'Work roll surface spalling or wear, excessive mill chatter, roll lubrication breakdown, or localized pass reduction imbalance.',
    risk: 'Degrades zinc galvanic coating and paint adhesion; creates surface roughness and cosmetic rejection in visible panels.',
    action: 'Regrind or replace work rolls; balance mill roll-bending hydraulic pressure; optimize cooling lubricant emulsion delivery.',
    recommended_action: 'Regrind or replace work rolls; balance mill roll-bending hydraulic pressure; optimize cooling lubricant emulsion delivery.',
    default_severity: 'minor',
    calibrated_threshold: 0.432,
    confidence_label: '≥ 43.2% conf',
  },
  pitted_surface: {
    name: 'pitted_surface',
    display_name: 'Pitted Surface',
    definition: 'Dense localized micropores, cavity depressions, or pinhole indentations across the metal surface.',
    visual_look: 'Mottled, rough cratered surface covered with dense pinholes, cavities, or localized chemical corrosion pockets.',
    cause: 'Over-pickling acid corrosion attacking grain boundaries, cavitation erosion, or mechanical impression from pitted work rolls.',
    typical_cause: 'Over-pickling acid corrosion attacking grain boundaries, cavitation erosion, or mechanical impression from pitted work rolls.',
    risk: 'Accelerates localized crevice and pitting corrosion; degrades seal mating surfaces and decreases structural fatigue limit.',
    action: 'Reduce pickling bath acid concentration and bath temperature; increase line speed; inspect and redress pitted rolls.',
    recommended_action: 'Reduce pickling bath acid concentration and bath temperature; increase line speed; inspect and redress pitted rolls.',
    default_severity: 'major',
    calibrated_threshold: 0.412,
    confidence_label: '≥ 41.2% conf',
  },
  rolled_in_scale: {
    name: 'rolled_in_scale',
    display_name: 'Rolled-in Scale',
    definition: 'Iron oxide mill scale mechanically pressed and embedded into the steel substrate during hot or cold rolling passes.',
    visual_look: 'Dark bluish or black fish-scale streaks, embedded elongated bands, or flaked trails aligned with rolling direction.',
    cause: 'Deficient high-pressure descaling header pressure, clogged water descaling spray nozzles, or furnace atmosphere control failure.',
    typical_cause: 'Deficient high-pressure descaling header pressure, clogged water descaling spray nozzles, or furnace atmosphere control failure.',
    risk: 'Causes subsequent surface delamination, seam weld porosity, poor electroplating bond, and subsurface blisters during forming.',
    action: 'Clean high-pressure descaling nozzles; maintain descaling water pressure at minimum 150 bar; balance roughing mill blowers.',
    recommended_action: 'Clean high-pressure descaling nozzles; maintain descaling water pressure at minimum 150 bar; balance roughing mill blowers.',
    default_severity: 'major',
    calibrated_threshold: 0.345,
    confidence_label: '≥ 34.5% conf',
  },
  scratches: {
    name: 'scratches',
    display_name: 'Scratches',
    definition: 'Linear abrasive furrows or mechanical gouges scored into the metallic surface by dragging contact.',
    visual_look: 'Straight, elongated continuous or intermittent reflective grooves and sharp score marks parallel to strip movement.',
    cause: 'Mechanical friction against misaligned or worn entry guide shoes, foreign abrasive debris on table rollers, or handling slips.',
    typical_cause: 'Mechanical friction against misaligned or worn entry guide shoes, foreign abrasive debris on table rollers, or handling slips.',
    risk: 'Linear stress concentration grooves act as sharp notch initiators, inducing tearing during deep drawing and stamping operations.',
    action: 'Inspect and polish conveyor guide shoes; clean and replace damaged roller bearings; calibrate strip tension control.',
    recommended_action: 'Inspect and polish conveyor guide shoes; clean and replace damaged roller bearings; calibrate strip tension control.',
    default_severity: 'minor',
    calibrated_threshold: 0.601,
    confidence_label: '≥ 60.1% conf',
  },
  unclassified_anomaly: {
    name: 'unclassified_anomaly',
    display_name: 'Unclassified Surface Anomaly',
    definition: 'Surface texture or structural anomaly deviating from nominal baseline that does not match standard NEU-DET training classes.',
    visual_look: 'Atypical contrast variance, localized geometric discoloration, fluid stains, or novel texture discrepancies.',
    cause: 'Foreign lubricant contamination, electrical/tooling contact burns, out-of-domain metallurgical flaw, or novel tooling failure.',
    typical_cause: 'Foreign lubricant contamination, electrical/tooling contact burns, out-of-domain metallurgical flaw, or novel tooling failure.',
    risk: 'Unknown mechanical defect escape; potential violation of surface roughness and critical dimensional tolerances.',
    action: 'Quarantine component for human QA inspection; execute SEM/EDX metallurgical review and optical surface profilometry.',
    recommended_action: 'Quarantine component for human QA inspection; execute SEM/EDX metallurgical review and optical surface profilometry.',
    default_severity: 'major',
    calibrated_threshold: 0.200,
    confidence_label: '≥ 20.0% conf',
  },
};

export const GlossaryPage: React.FC = () => {
  const [glossary, setGlossary] = useState<Record<string, DefectKnowledgeItem>>(DEFAULT_GLOSSARY);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  useEffect(() => {
    apiClient
      .getGlossary()
      .then((data) => {
        if (data && Object.keys(data).length > 0) {
          setGlossary(data);
        }
      })
      .catch((err) => {
        console.warn('Using default glossary fallback:', err);
      });
  }, []);

  const items = Object.values(glossary);

  const filteredItems = items.filter((item) => {
    const q = searchQuery.toLowerCase().trim();
    const confStr = item.confidence_label ? item.confidence_label.toLowerCase() : '';
    const matchesSearch =
      !q ||
      item.name.toLowerCase().includes(q) ||
      item.display_name.toLowerCase().includes(q) ||
      confStr.includes(q) ||
      item.definition.toLowerCase().includes(q) ||
      item.cause.toLowerCase().includes(q) ||
      item.action.toLowerCase().includes(q) ||
      item.visual_look.toLowerCase().includes(q);

    const matchesSeverity =
      severityFilter === 'all' || item.default_severity.toLowerCase() === severityFilter.toLowerCase();

    return matchesSearch && matchesSeverity;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-brand-blue flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-black text-white tracking-tight flex items-center space-x-3">
                <span>Defect Knowledge Glossary</span>
                <span className="text-xs px-2.5 py-0.5 rounded-full font-mono bg-sky-950 text-sky-400 border border-sky-800">
                  7 Classes Documented
                </span>
              </h1>
              <p className="text-sm text-slate-400 mt-0.5">
                Physical definitions, visual signatures, root manufacturing causes, and calibrated detection thresholds.
              </p>
            </div>
          </div>
        </div>

        {/* Global Confidence Rule Pill */}
        <div className="glass-panel px-4 py-2 rounded-full flex items-center space-x-3 text-xs font-mono">
          <Sparkles className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span className="text-slate-300">
            Rule Applied: <strong className="text-white">Every defect name displayed with calibrated confidence</strong>
          </span>
        </div>
      </div>

      {/* Metric Highlights */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs text-slate-400 font-medium">NEU-DET Benchmark Classes</span>
          <p className="text-xl font-bold font-mono text-white mt-1">6 Supervised</p>
          <span className="text-[11px] text-sky-400 font-mono">YOLOv8n Active Weights</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs text-slate-400 font-medium">Out-of-Domain Detection</span>
          <p className="text-xl font-bold font-mono text-amber-400 mt-1">1 Unsupervised</p>
          <span className="text-[11px] text-amber-400 font-mono">Blackhat + CV Saliency</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs text-slate-400 font-medium">Calibrated Confidence Range</span>
          <p className="text-xl font-bold font-mono text-emerald-400 mt-1">13.6% – 60.1%</p>
          <span className="text-[11px] text-emerald-400 font-mono">Max F1-Score Optimized</span>
        </div>
        <div className="glass-panel p-5 rounded-2xl">
          <span className="text-xs text-slate-400 font-medium">Critical Routing Gate</span>
          <p className="text-xl font-bold font-mono text-rose-400 mt-1">1 Auto-Reject Class</p>
          <span className="text-[11px] text-rose-400 font-mono">Crazing micro-fractures</span>
        </div>
      </div>

      {/* Search and Filters Bar */}
      <div className="glass-panel p-4 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-96">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search defects, causes, actions, or confidence..."
            className="w-full bg-white/[0.04] border border-white/[0.08] rounded-full pl-10 pr-4 py-2 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-sky-400/60 font-sans transition-colors"
          />
        </div>

        <div className="flex items-center space-x-2 w-full md:w-auto overflow-x-auto pb-1 md:pb-0 bg-white/[0.04] border border-white/[0.08] rounded-full p-1 backdrop-blur-xl">
          <span className="text-xs text-slate-400 font-medium px-2.5 flex items-center space-x-1.5">
            <Filter className="w-3.5 h-3.5 text-sky-400" />
            <span>Severity:</span>
          </span>
          {['all', 'critical', 'major', 'minor'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSeverityFilter(sev)}
              className={`px-3 py-1 rounded-full text-xs font-semibold capitalize transition-all ${
                severityFilter === sev
                  ? 'bg-sky-500/25 text-sky-200 border border-sky-400/40 shadow-[0_0_12px_rgba(56,189,248,0.25)]'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {sev}
            </button>
          ))}
        </div>
      </div>

      {/* Defect Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredItems.map((item) => {
          const isSelected = selectedKey === item.name;
          const confLabel = item.confidence_label || `≥ ${(Number(item.calibrated_threshold || 0.40) * 100).toFixed(1)}% conf`;

          return (
            <div
              key={item.name}
              onClick={() => setSelectedKey(isSelected ? null : item.name)}
              className={`glass-panel rounded-2xl transition-all cursor-pointer flex flex-col justify-between overflow-hidden ${
                isSelected
                  ? 'border-sky-400/80 shadow-[0_0_30px_rgba(56,189,248,0.2)] ring-1 ring-sky-400/50'
                  : 'hover:border-white/[0.16] hover:shadow-[0_8px_32px_rgba(0,0,0,0.4)]'
              }`}
            >
              <div className="p-5 space-y-4">
                {/* Header with Name & Confidence strictly attached */}
                <div className="flex items-start justify-between gap-2 border-b border-white/[0.06] pb-3">
                  <div>
                    <h3 className="text-base font-bold text-white font-mono flex items-center flex-wrap gap-2">
                      <span>{item.display_name}</span>
                      <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-sky-500/15 text-sky-300 border border-sky-400/30">
                        {confLabel}
                      </span>
                    </h3>
                    <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                      Class ID: #{item.name} | Model: {item.name === 'unclassified_anomaly' ? 'Anomaly CV Engine' : 'YOLOv8 Edge'}
                    </p>
                  </div>
                  <SeverityBadge level={item.default_severity} size="sm" />
                </div>

                {/* Plain Definition */}
                <div className="space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1">
                    <Info className="w-3 h-3 text-sky-400" />
                    <span>Plain Definition</span>
                  </span>
                  <p className="text-xs text-slate-200 leading-relaxed bg-white/[0.03] backdrop-blur-md p-3 rounded-xl border border-white/[0.06]">
                    {item.definition}
                  </p>
                </div>

                {/* Visual Appearance */}
                <div className="space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-purple-400 flex items-center space-x-1">
                    <Eye className="w-3 h-3 text-purple-400" />
                    <span>Visual Appearance</span>
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed bg-purple-950/20 backdrop-blur-md p-3 rounded-xl border border-purple-500/20">
                    {item.visual_look}
                  </p>
                </div>

                {/* Manufacturing Cause */}
                <div className="space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-amber-400 flex items-center space-x-1">
                    <AlertTriangle className="w-3 h-3 text-amber-400" />
                    <span>Typical Manufacturing Cause</span>
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed bg-amber-950/20 backdrop-blur-md p-3 rounded-xl border border-amber-500/20">
                    {item.cause}
                  </p>
                </div>

                {/* Engineering Failure Risk */}
                <div className="space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-rose-400 flex items-center space-x-1">
                    <ShieldAlert className="w-3 h-3 text-rose-400" />
                    <span>Failure Risk</span>
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed bg-rose-950/20 backdrop-blur-md p-3 rounded-xl border border-rose-500/20">
                    {item.risk}
                  </p>
                </div>

                {/* Recommended Corrective Action */}
                <div className="space-y-1">
                  <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-emerald-400 flex items-center space-x-1">
                    <Wrench className="w-3 h-3 text-emerald-400" />
                    <span>Recommended Action</span>
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed bg-emerald-950/20 backdrop-blur-md p-3 rounded-xl border border-emerald-500/20">
                    {item.action}
                  </p>
                </div>
              </div>

              {/* Card Footer */}
              <div className="bg-slate-950/60 backdrop-blur-xl px-5 py-3 border-t border-white/[0.06] rounded-b-2xl flex items-center justify-between text-xs font-mono text-slate-400">
                <span className="flex items-center space-x-1">
                  <span>Threshold:</span>
                  <strong className="text-white">{confLabel}</strong>
                </span>
                <span className="text-sky-400 font-sans text-[11px] font-medium flex items-center space-x-1">
                  <span>{item.display_name} ({confLabel})</span>
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {filteredItems.length === 0 && (
        <div className="text-center py-16 glass-panel rounded-2xl space-y-3">
          <BookOpen className="w-12 h-12 text-slate-500 mx-auto" />
          <h3 className="text-lg font-bold text-white">No Defect Classes Found</h3>
          <p className="text-xs text-slate-400">
            No defect classes matched &quot;{searchQuery}&quot; under the selected severity filter.
          </p>
        </div>
      )}
    </div>
  );
};

