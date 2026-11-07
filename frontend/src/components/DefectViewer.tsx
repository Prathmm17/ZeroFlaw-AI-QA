import React, { useState, useRef, useEffect } from 'react';
import {
  Eye,
  EyeOff,
  Layers,
  ZoomIn,
  ZoomOut,
  Info,
  ShieldCheck,
  AlertCircle,
  Sparkles,
  FileText,
  Wrench,
  Ruler,
  RotateCcw,
  Check,
  X,
  Download,
  ExternalLink,
} from 'lucide-react';
import { Inspection, Defect, EightDReport } from '../types';
import { SeverityBadge } from './SeverityBadge';
import { apiClient } from '../api/client';

interface DefectViewerProps {
  inspection: Inspection;
}

interface Point {
  x: number;
  y: number;
}

export const DefectViewer: React.FC<DefectViewerProps> = ({ inspection }) => {
  const [currentInspection, setCurrentInspection] = useState<Inspection>(inspection);
  const [showAnnotated, setShowAnnotated] = useState(true);
  const [selectedDefectIdx, setSelectedDefectIdx] = useState<number | null>(null);
  const [auditReport, setAuditReport] = useState<EightDReport | null>(inspection.audit_report || null);
  const [isAuditing, setIsAuditing] = useState(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1.0);

  // Calibration Helper State
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [lineStart, setLineStart] = useState<Point | null>(null);
  const [lineEnd, setLineEnd] = useState<Point | null>(null);
  const [measuredPx, setMeasuredPx] = useState<number | null>(null);
  const [knownLengthMm, setKnownLengthMm] = useState<string>('10.0');
  const [calibMessage, setCalibMessage] = useState<string | null>(null);
  const [isSavingCalib, setIsSavingCalib] = useState(false);

  // Overlay & image bounds for defect box highlight
  const [naturalDimensions, setNaturalDimensions] = useState<{ width: number; height: number }>({ width: 640, height: 640 });
  const [imageContentBounds, setImageContentBounds] = useState<{ left: number; top: number; width: number; height: number }>({ left: 0, top: 0, width: 0, height: 0 });

  const imgRef = useRef<HTMLImageElement>(null);
  const svgOverlayRef = useRef<SVGSVGElement>(null);

  const updateImageBounds = () => {
    if (!imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const nw = imgRef.current.naturalWidth || 640;
    const nh = imgRef.current.naturalHeight || 640;
    setNaturalDimensions({ width: nw, height: nh });

    if (rect.width === 0 || rect.height === 0) return;

    const imgAspect = nw / nh;
    const elemAspect = rect.width / rect.height;

    let renderW = rect.width;
    let renderH = rect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (imgAspect > elemAspect) {
      renderW = rect.width;
      renderH = rect.width / imgAspect;
      offsetY = (rect.height - renderH) / 2;
    } else {
      renderH = rect.height;
      renderW = rect.height * imgAspect;
      offsetX = (rect.width - renderW) / 2;
    }

    setImageContentBounds({ left: offsetX, top: offsetY, width: renderW, height: renderH });
  };

  useEffect(() => {
    updateImageBounds();
    window.addEventListener('resize', updateImageBounds);
    return () => window.removeEventListener('resize', updateImageBounds);
  }, [currentInspection]);

  useEffect(() => {
    setCurrentInspection(inspection);
    setAuditReport(inspection.audit_report || null);
  }, [inspection]);

  const handleRunAudit = async () => {
    setIsAuditing(true);
    try {
      const res = await fetch(`/api/detect/${currentInspection.id}/audit`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setAuditReport(data);
        setCurrentInspection((prev) => ({
          ...prev,
          audit_report: data,
        }));
      } else {
        const errText = await res.text().catch(() => '');
        console.error('Audit failed with status:', res.status, errText);
      }
    } catch (e) {
      console.error('Audit failed', e);
    } finally {
      setIsAuditing(false);
    }
  };

  // Convert SVG coordinates to actual image pixel coordinates
  const computeImagePixelDistance = (p1: Point, p2: Point): number => {
    if (!imgRef.current) return Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const rect = imgRef.current.getBoundingClientRect();
    const naturalW = imgRef.current.naturalWidth || rect.width;
    const naturalH = imgRef.current.naturalHeight || rect.height;

    const scaleX = naturalW / (rect.width || 1);
    const scaleY = naturalH / (rect.height || 1);

    const dx = (p2.x - p1.x) * scaleX;
    const dy = (p2.y - p1.y) * scaleY;
    return Math.hypot(dx, dy);
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isCalibrating || !svgOverlayRef.current) return;
    const rect = svgOverlayRef.current.getBoundingClientRect();
    const pt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setLineStart(pt);
    setLineEnd(pt);
    setIsDrawingLine(true);
    setCalibMessage(null);
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isCalibrating || !isDrawingLine || !svgOverlayRef.current) return;
    const rect = svgOverlayRef.current.getBoundingClientRect();
    const pt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setLineEnd(pt);
    if (lineStart) {
      const pxDist = computeImagePixelDistance(lineStart, pt);
      setMeasuredPx(pxDist);
    }
  };

  const handleMouseUp = () => {
    if (!isCalibrating || !isDrawingLine) return;
    setIsDrawingLine(false);
    if (lineStart && lineEnd) {
      const pxDist = computeImagePixelDistance(lineStart, lineEnd);
      if (pxDist > 3) {
        setMeasuredPx(pxDist);
      }
    }
  };

  const computedScale =
    measuredPx && parseFloat(knownLengthMm) > 0
      ? measuredPx / parseFloat(knownLengthMm)
      : null;

  const handleApplyCalibration = async () => {
    if (!measuredPx || !parseFloat(knownLengthMm)) return;
    setIsSavingCalib(true);
    setCalibMessage(null);
    try {
      const res = await apiClient.setCalibration({
        line_length_px: measuredPx,
        known_length_mm: parseFloat(knownLengthMm),
      });
      // Recalculate metrology for the active inspection
      const updatedInsp = await apiClient.recalculateMetrology(currentInspection.id);
      setCurrentInspection(updatedInsp);
      setCalibMessage(`Calibrated: ${res.px_per_mm?.toFixed(2)} px/mm applied.`);
    } catch (e) {
      console.error('Failed to set calibration', e);
      setCalibMessage('Failed to apply calibration.');
    } finally {
      setIsSavingCalib(false);
    }
  };

  const handleResetCalibration = async () => {
    setIsSavingCalib(true);
    setCalibMessage(null);
    try {
      await apiClient.setCalibration({ px_per_mm: null });
      const updatedInsp = await apiClient.recalculateMetrology(currentInspection.id);
      setCurrentInspection(updatedInsp);
      setMeasuredPx(null);
      setLineStart(null);
      setLineEnd(null);
      setCalibMessage('Reset to uncalibrated (px) mode.');
    } catch (e) {
      console.error('Failed to reset calibration', e);
      setCalibMessage('Failed to reset scale.');
    } finally {
      setIsSavingCalib(false);
    }
  };

  const currentImageUrl = showAnnotated ? currentInspection.result_url : currentInspection.original_url;
  const isSpecimenCalibrated = Boolean(
    currentInspection.is_calibrated ||
      currentInspection.defects.some((d) => d.metrology?.is_calibrated && d.metrology.surface_area_mm2 != null)
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Image Preview Canvas */}
      <div className="lg:col-span-2 space-y-3">
        <div className="flex items-center justify-between glass-panel px-5 py-3 rounded-t-2xl border-b-0">
          <div className="flex items-center space-x-2.5">
            <div className="w-6 h-6 rounded-lg bg-sky-500/10 border border-sky-400/20 flex items-center justify-center">
              <Layers className="w-3.5 h-3.5 text-sky-400" />
            </div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              Inspection Viewport
            </span>
          </div>
          <div className="flex items-center space-x-2">
            {/* Interactive Zoom Controls */}
            <div className="flex items-center space-x-1 bg-white/[0.04] rounded-full px-2 py-0.5 border border-white/[0.08] backdrop-blur-xl">
              <button
                onClick={() => setZoomLevel((z) => Math.max(1.0, Number((z - 0.5).toFixed(1))))}
                disabled={zoomLevel <= 1.0}
                className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] font-mono text-sky-400 px-1 font-bold min-w-[2.6rem] text-center">
                {(zoomLevel * 100).toFixed(0)}%
              </span>
              <button
                onClick={() => setZoomLevel((z) => Math.min(4.0, Number((z + 0.5).toFixed(1))))}
                disabled={zoomLevel >= 4.0}
                className="p-1 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              {zoomLevel > 1.0 && (
                <button
                  onClick={() => setZoomLevel(1.0)}
                  className="text-[10px] text-slate-400 hover:text-white px-1.5 py-0.5 font-mono border-l border-white/[0.08]"
                  title="Reset 1x Zoom"
                >
                  1x
                </button>
              )}
            </div>

            <button
              onClick={() => {
                setIsCalibrating(!isCalibrating);
                setCalibMessage(null);
              }}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                isCalibrating
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-400/40 shadow-[0_0_15px_rgba(245,158,11,0.25)]'
                  : isSpecimenCalibrated
                  ? 'bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 border border-emerald-400/20'
                  : 'bg-white/[0.04] text-slate-300 hover:bg-white/[0.08] border border-white/[0.08]'
              }`}
              title="Calibrate optical scale (px/mm) using an object of known length"
            >
              <Ruler className="w-3.5 h-3.5" />
              <span>{isCalibrating ? 'Close Ruler' : isSpecimenCalibrated ? 'Calibrated Scale' : 'Calibrate Scale'}</span>
            </button>
            <button
              onClick={() => setShowAnnotated(!showAnnotated)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                showAnnotated
                  ? 'bg-sky-500/20 text-sky-200 border border-sky-400/40 shadow-[0_0_15px_rgba(56,189,248,0.25)]'
                  : 'bg-white/[0.04] text-slate-300 hover:bg-white/[0.08] border border-white/[0.08]'
              }`}
            >
              {showAnnotated ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span>{showAnnotated ? 'AI Overlay Active' : 'Raw Original'}</span>
            </button>
          </div>
        </div>

        {/* Calibration Interactive Helper Tool */}
        {isCalibrating && (
          <div className="glass-panel p-4 rounded-2xl flex flex-wrap items-center justify-between gap-3 text-xs font-mono border-amber-500/30 shadow-[0_8px_32px_rgba(245,158,11,0.15)]">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-amber-300 font-bold flex items-center space-x-1.5">
                <Ruler className="w-4 h-4 text-amber-400" />
                <span>Optical Scale Calibration:</span>
              </span>
              <span className="text-slate-300">
                Line:{' '}
                <strong className="text-amber-400">
                  {measuredPx ? `${measuredPx.toFixed(1)} px` : 'Click & drag on image'}
                </strong>
              </span>
              <div className="flex items-center space-x-1.5">
                <label className="text-slate-400">Known Length:</label>
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  value={knownLengthMm}
                  onChange={(e) => setKnownLengthMm(e.target.value)}
                  className="w-16 px-2 py-0.5 bg-slate-950/80 border border-white/[0.1] rounded-lg text-amber-300 text-center font-bold"
                />
                <span className="text-slate-400">mm</span>
              </div>
              <span className="text-slate-300">
                Computed Scale:{' '}
                <strong className="text-sky-300 font-bold">
                  {computedScale ? `${computedScale.toFixed(2)} px/mm` : '—'}
                </strong>
              </span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleApplyCalibration}
                disabled={!measuredPx || !parseFloat(knownLengthMm) || isSavingCalib}
                className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded font-medium flex items-center space-x-1 transition-colors"
              >
                <Check className="w-3.5 h-3.5" />
                <span>{isSavingCalib ? 'Applying...' : 'Apply Scale'}</span>
              </button>
              <button
                onClick={handleResetCalibration}
                disabled={isSavingCalib}
                className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded font-medium flex items-center space-x-1 transition-colors"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset (px mode)</span>
              </button>
              <button
                onClick={() => setIsCalibrating(false)}
                className="p-1 text-slate-400 hover:text-white transition-colors"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {calibMessage && (
          <div className="bg-sky-950/60 border border-sky-800/80 px-3 py-1.5 rounded-lg text-xs font-mono text-sky-200 flex items-center justify-between">
            <span>ℹ️ {calibMessage}</span>
            <button onClick={() => setCalibMessage(null)} className="text-sky-400 hover:text-sky-200">
              <X className="w-3 h-3" />
            </button>
          </div>
        )}

        <div className="relative aspect-video sm:aspect-[4/3] rounded-b-2xl overflow-auto border border-violet-500/20 bg-[#090614]/85 backdrop-blur-3xl flex items-center justify-center group select-none shadow-[0_16px_40px_rgba(0,0,0,0.7)]">
          <div
            className="relative w-full h-full flex items-center justify-center transition-transform duration-200"
            style={{
              transform: `scale(${zoomLevel})`,
              transformOrigin: 'center center',
            }}
          >
            <img
              ref={imgRef}
              src={currentImageUrl}
              alt={currentInspection.filename}
              onLoad={updateImageBounds}
              className="w-full h-full object-contain pointer-events-none"
            />

          {/* Highlighted Defect Bounding Box SVG Overlay */}
          {imageContentBounds.width > 0 && (
            <svg
              style={{
                position: 'absolute',
                left: `${imageContentBounds.left}px`,
                top: `${imageContentBounds.top}px`,
                width: `${imageContentBounds.width}px`,
                height: `${imageContentBounds.height}px`,
                pointerEvents: isCalibrating ? 'none' : 'auto',
              }}
              viewBox={`0 0 ${naturalDimensions.width} ${naturalDimensions.height}`}
              className="z-10"
            >
              <defs>
                <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
                  <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#06b6d4" floodOpacity="0.9" />
                  <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#38bdf8" floodOpacity="1" />
                </filter>
              </defs>

              {currentInspection.defects.map((defect, idx) => {
                const isSelected = selectedDefectIdx === idx;
                const [x1, y1, x2, y2] = defect.bbox;
                const bw = Math.max(2, x2 - x1);
                const bh = Math.max(2, y2 - y1);
                const strokeWidth = Math.max(2.5, Math.round(naturalDimensions.width / 220));

                if (isSelected) {
                  return (
                    <g key={idx} filter="url(#neon-glow)" onClick={() => setSelectedDefectIdx(null)} className="cursor-pointer">
                      {/* Highlighted Glowing Bounding Box */}
                      <rect
                        x={x1}
                        y={y1}
                        width={bw}
                        height={bh}
                        fill="rgba(6, 182, 212, 0.28)"
                        stroke="#06b6d4"
                        strokeWidth={strokeWidth}
                        strokeDasharray="6 3"
                      />
                      {/* Corner Accents */}
                      <line x1={x1} y1={y1} x2={x1 + Math.min(24, bw * 0.4)} y2={y1} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x1} y1={y1} x2={x1} y2={y1 + Math.min(24, bh * 0.4)} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x2} y1={y1} x2={x2 - Math.min(24, bw * 0.4)} y2={y1} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x2} y1={y1} x2={x2} y2={y1 + Math.min(24, bh * 0.4)} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x1} y1={y2} x2={x1 + Math.min(24, bw * 0.4)} y2={y2} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x1} y1={y2} x2={x1} y2={y2 - Math.min(24, bh * 0.4)} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x2} y1={y2} x2={x2 - Math.min(24, bw * 0.4)} y2={y2} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />
                      <line x1={x2} y1={y2} x2={x2} y2={y2 - Math.min(24, bh * 0.4)} stroke="#ffffff" strokeWidth={strokeWidth * 1.5} />

                      {/* Polygon highlight if available */}
                      {defect.polygon && defect.polygon.length >= 3 && (
                        <polygon
                          points={defect.polygon.map(([px, py]) => `${px * naturalDimensions.width},${py * naturalDimensions.height}`).join(' ')}
                          fill="rgba(6, 182, 212, 0.35)"
                          stroke="#38bdf8"
                          strokeWidth={strokeWidth}
                        />
                      )}

                      {/* Floating Label with Defect Name and Confidence */}
                      <g transform={`translate(${x1}, ${Math.max(26, y1 - 8)})`}>
                        <rect
                          x="0"
                          y="-22"
                          width={Math.max(170, (defect.class_name.length + 15) * 11)}
                          height="24"
                          rx="4"
                          fill="#080c14"
                          stroke="#06b6d4"
                          strokeWidth="1.5"
                        />
                        <text
                          x="8"
                          y="-6"
                          fill="#38bdf8"
                          fontSize={Math.max(11, Math.round(naturalDimensions.width / 50))}
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          #{idx + 1} {defect.class_name.toUpperCase()} ({(defect.confidence * 100).toFixed(1)}% conf)
                        </text>
                      </g>
                    </g>
                  );
                }

                return (
                  <rect
                    key={idx}
                    x={x1}
                    y={y1}
                    width={bw}
                    height={bh}
                    fill="transparent"
                    stroke="rgba(255, 255, 255, 0.25)"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    className="cursor-pointer hover:stroke-brand-blue hover:fill-brand-blue/10 transition-colors"
                    onClick={() => setSelectedDefectIdx(idx)}
                  />
                );
              })}
            </svg>
          )}

          {/* Interactive Line-drawing SVG overlay */}
          {isCalibrating && (
            <svg
              ref={svgOverlayRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              className="absolute inset-0 w-full h-full cursor-crosshair z-20"
            >
              {lineStart && lineEnd && (
                <g>
                  <line
                    x1={lineStart.x}
                    y1={lineStart.y}
                    x2={lineEnd.x}
                    y2={lineEnd.y}
                    stroke="#f59e0b"
                    strokeWidth="3"
                    strokeDasharray="4 2"
                  />
                  <circle cx={lineStart.x} cy={lineStart.y} r="5" fill="#f59e0b" stroke="#ffffff" strokeWidth="1.5" />
                  <circle cx={lineEnd.x} cy={lineEnd.y} r="5" fill="#f59e0b" stroke="#ffffff" strokeWidth="1.5" />
                  {measuredPx && (
                    <text
                      x={(lineStart.x + lineEnd.x) / 2 + 8}
                      y={(lineStart.y + lineEnd.y) / 2 - 8}
                      fill="#fef08a"
                      fontSize="12"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {measuredPx.toFixed(1)} px
                    </text>
                  )}
                </g>
              )}
            </svg>
          )}
          </div>

          {/* HUD Overlay in corners */}
          <div className="absolute top-3 left-3 bg-[#0d091a]/85 backdrop-blur-3xl px-3.5 py-2 rounded-xl border border-violet-500/25 text-xs font-mono space-y-0.5 pointer-events-none shadow-xl">
            <p className="text-violet-200/70">FILE: <span className="text-white font-medium">{currentInspection.filename}</span></p>
            <p className="text-violet-200/70">LATENCY: <span className="text-emerald-400 font-bold">{currentInspection.inference_time_ms.toFixed(1)}ms</span></p>
          </div>

          <div className="absolute bottom-3 right-3 bg-[#0d091a]/85 backdrop-blur-3xl px-3.5 py-2 rounded-xl border border-violet-500/25 flex items-center space-x-2 pointer-events-none shadow-xl">
            {isSpecimenCalibrated ? (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-emerald-400/20 text-emerald-200 border border-emerald-300/40 font-bold">
                calibrated
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-amber-400/20 text-amber-200 border border-amber-300/40 font-bold">
                uncalibrated (px)
              </span>
            )}
            {currentInspection.arbitration_status === 'VLM_VERIFIED' || (currentInspection.arbitration_tier === 'tier2_vlm' && currentInspection.arbitration_status === 'VLM_CONFIRMED') ? (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-purple-400/20 text-purple-200 border border-purple-300/40 flex items-center space-x-1.5 shadow-[0_0_12px_rgba(192,132,252,0.25)]">
                <Sparkles className="w-3 h-3 text-purple-300" />
                <span>Tier 2 VLM Verified</span>
              </span>
            ) : currentInspection.arbitration_status === 'ANOMALY_ESCALATED' || currentInspection.defects.some((d) => d.source === 'anomaly_cv') ? (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-amber-400/20 text-amber-200 border border-amber-300/40 flex items-center space-x-1">
                <span>Anomaly (CV)</span>
              </span>
            ) : (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-blue-400/20 text-blue-200 border border-blue-300/40 flex items-center space-x-1">
                <span>⚡ Tier 1 Edge Fast Gate</span>
              </span>
            )}
            <span className="text-xs text-white/70 font-mono">RESULT:</span>
            <SeverityBadge level={currentInspection.status} size="sm" />
          </div>
        </div>
      </div>

      {/* Defect Diagnostics Panel */}
      <div className="space-y-4">
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <h4 className="text-sm font-semibold text-white flex items-center space-x-2">
              <Info className="w-4 h-4 text-sky-400" />
              <span>Inspection Summary</span>
            </h4>
            <SeverityBadge level={currentInspection.max_severity} size="sm" />
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-white/[0.03] backdrop-blur-xl p-3 rounded-xl border border-white/[0.06]">
              <span className="text-slate-400 font-medium">Defects Found</span>
              <p className="text-xl font-bold font-mono text-white mt-0.5">{currentInspection.total_defects}</p>
            </div>
            <div className="bg-white/[0.03] backdrop-blur-xl p-3 rounded-xl border border-white/[0.06]">
              <span className="text-slate-400 font-medium">Inference Time</span>
              <p className="text-xl font-bold font-mono text-white mt-0.5">{currentInspection.inference_time_ms.toFixed(0)} ms</p>
            </div>
            <div className="bg-white/[0.03] backdrop-blur-xl p-3 rounded-xl border border-white/[0.06]">
              <span className="text-slate-400 font-medium">Critical Alerts</span>
              <p className="text-xl font-bold font-mono text-rose-400 mt-0.5">{currentInspection.critical_count}</p>
            </div>
            <div className="bg-white/[0.03] backdrop-blur-xl p-3 rounded-xl border border-white/[0.06]">
              <span className="text-slate-400 font-medium">Major Defects</span>
              <p className="text-xl font-bold font-mono text-amber-400 mt-0.5">{currentInspection.major_count}</p>
            </div>
          </div>
        </div>

        {/* Defect List */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white">Detected Defects List</h4>
            <span className="text-xs font-mono text-slate-400 bg-white/[0.04] px-2.5 py-0.5 rounded-full border border-white/[0.06]">{currentInspection.defects.length} detected</span>
          </div>

          {currentInspection.defects.length === 0 ? (
            <div className="text-center py-8 space-y-2">
              {currentInspection.status.toLowerCase().includes('flagged') ? (
                <>
                  <AlertCircle className="w-10 h-10 text-amber-400 mx-auto" />
                  <p className="text-sm font-semibold text-amber-300">Flagged For Manual Review</p>
                  <p className="text-xs text-slate-400">Specimen flagged due to domain ambiguity or low trust; manual verification required.</p>
                </>
              ) : (
                <>
                  <ShieldCheck className="w-10 h-10 text-emerald-400 mx-auto" />
                  <p className="text-sm font-semibold text-white">Zero Defects Detected</p>
                  <p className="text-xs text-slate-400">Component meets quality inspection thresholds.</p>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-3 max-h-[34rem] overflow-y-auto pr-1">
              {currentInspection.defects.map((defect, idx) => (
                <div
                  key={idx}
                  onClick={() => setSelectedDefectIdx(idx === selectedDefectIdx ? null : idx)}
                  className={`p-4 rounded-2xl border cursor-pointer transition-all ${
                    selectedDefectIdx === idx
                      ? 'bg-sky-500/10 border-sky-400/80 shadow-[0_0_25px_rgba(56,189,248,0.2)] ring-1 ring-sky-400/50'
                      : 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.05] hover:border-white/[0.12]'
                  }`}
                >
                  {/* Card Header: Defect Name + Confidence, Source, Trust Level, and Severity */}
                  <div className="flex items-start justify-between gap-2 border-b border-slate-800/80 pb-2.5">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                        <span className="text-xs font-bold font-mono uppercase text-white tracking-wide">
                          #{idx + 1} {defect.class_name === 'unclassified_anomaly' ? 'Unclassified surface anomaly' : defect.class_name}{' '}
                          <span className="text-sky-300">({(defect.confidence * 100).toFixed(1)}% conf)</span>
                        </span>

                        {/* Source Badge */}
                        {defect.source === 'vlm' || (defect.vlm_verified && defect.source !== 'anomaly_cv') ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-900/60 text-purple-200 border border-purple-700 font-mono flex items-center space-x-1">
                            <Sparkles className="w-2.5 h-2.5 text-purple-400" />
                            <span>VLM</span>
                          </span>
                        ) : defect.source === 'anomaly_cv' || defect.class_name === 'unclassified_anomaly' ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-950/70 text-amber-300 border border-amber-700 font-mono">
                            Anomaly (CV)
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-sky-950/70 text-sky-300 border border-sky-700 font-mono">
                            YOLO
                          </span>
                        )}

                        {/* Domain Trust Level */}
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${
                            currentInspection.domain_trust === 'LOW'
                              ? 'bg-amber-950/70 text-amber-300 border-amber-700'
                              : 'bg-emerald-950/70 text-emerald-300 border-emerald-700'
                          }`}
                        >
                          Trust: {currentInspection.domain_trust || 'HIGH'}
                        </span>

                        {selectedDefectIdx === idx && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-700 animate-pulse">
                            Box Highlighted
                          </span>
                        )}
                      </div>

                      <div className="flex items-center space-x-3 text-[11px] font-mono text-slate-400">
                        <span>Confidence: <strong className="text-white">{(defect.confidence * 100).toFixed(1)}%</strong></span>
                        <span>Area: <strong className="text-white">{(defect.area_ratio * 100).toFixed(2)}%</strong></span>
                      </div>
                    </div>
                    <SeverityBadge level={defect.severity} size="sm" />
                  </div>

                  {defect.failure_mechanism && (
                    <p className="text-[11px] text-purple-300 bg-purple-950/40 px-2 py-1 rounded border border-purple-900/50 mt-2 font-mono">
                      🔬 {defect.failure_mechanism}
                    </p>
                  )}

                  {/* Defect Knowledge Base: Definition, Cause, Action */}
                  <div className="mt-2.5 space-y-1.5 text-xs bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/80 font-sans">
                    <div>
                      <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-400 block">Definition:</span>
                      <p className="text-slate-200 text-xs leading-relaxed mt-0.5">
                        {defect.definition || 'Manufacturing surface anomaly identified on inspected steel component.'}
                      </p>
                    </div>
                    <div className="pt-1.5 border-t border-slate-800/70">
                      <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-amber-400 block">Typical Cause:</span>
                      <p className="text-slate-300 text-xs leading-relaxed mt-0.5">
                        {defect.cause || 'Rolling mill mechanical wear, thermal quench shock, or refractory lining degradation.'}
                      </p>
                    </div>
                    <div className="pt-1.5 border-t border-slate-800/70">
                      <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-sky-400 block">Recommended Action:</span>
                      <p className="text-slate-300 text-xs leading-relaxed mt-0.5">
                        {defect.action || 'Quarantine component for rework station and inspect upstream tooling tolerances.'}
                      </p>
                    </div>
                  </div>

                  {defect.description && !defect.definition && (
                    <p className="text-xs text-slate-400 mt-2 italic">{defect.description}</p>
                  )}
                  {defect.metrology && (
                    <div className="mt-2 bg-slate-950/60 p-2.5 rounded border border-slate-800/80 font-mono text-[11px] space-y-1.5">
                      <div className="flex items-center justify-between pb-1 border-b border-slate-800/60">
                        <span className="text-slate-400 font-sans text-[10px] tracking-wide uppercase">Metrology Measurements</span>
                        {defect.metrology.is_calibrated && defect.metrology.surface_area_mm2 != null ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950/80 text-emerald-300 border border-emerald-700/80 font-mono">
                            calibrated {defect.metrology.px_per_mm ? `(${defect.metrology.px_per_mm.toFixed(1)} px/mm)` : ''}
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-950/80 text-amber-300 border border-amber-700/80 font-mono">
                            uncalibrated (px)
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 gap-1.5">
                        {defect.metrology.is_calibrated && defect.metrology.surface_area_mm2 != null ? (
                          <>
                            <div>
                              <span className="text-slate-400">Area:</span>{' '}
                              <span className="text-sky-300 font-bold">{defect.metrology.surface_area_mm2} mm²</span>
                            </div>
                            <div>
                              <span className="text-slate-400">Length:</span>{' '}
                              <span className="text-amber-300 font-bold">{defect.metrology.length_mm} mm</span>
                            </div>
                            <div>
                              <span className="text-slate-400">Width:</span>{' '}
                              <span className="text-amber-300 font-bold">{defect.metrology.width_mm} mm</span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div>
                              <span className="text-slate-400">Area:</span>{' '}
                              <span className="text-sky-300 font-bold">{defect.metrology.area_px ?? defect.metrology.pixel_area} px</span>
                            </div>
                            <div>
                              <span className="text-slate-400">Length:</span>{' '}
                              <span className="text-amber-300 font-bold">{defect.metrology.length_px ?? '—'} px</span>
                            </div>
                            <div>
                              <span className="text-slate-400">Width:</span>{' '}
                              <span className="text-amber-300 font-bold">{defect.metrology.width_px ?? '—'} px</span>
                            </div>
                          </>
                        )}
                        <div>
                          <span className="text-slate-400">Ratio:</span>{' '}
                          <span className="text-emerald-300">{defect.metrology.aspect_ratio}:1</span>
                        </div>
                        <div>
                          <span className="text-slate-400">Solidity:</span>{' '}
                          <span className="text-purple-300">{defect.metrology.solidity}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {selectedDefectIdx === idx && (
                    <div className="mt-2 pt-2 border-t border-slate-700/60 text-[11px] font-mono text-slate-400">
                      BBox: [{defect.bbox.map((n) => Math.round(n)).join(', ')}]
                      {defect.polygon && (
                        <span className="ml-2 text-sky-400">({defect.polygon.length} polygon vertices)</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quality Conformance Verification Matrix (Truth Evaluation) */}
        {currentInspection.conformance_report && (
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <h4 className="text-sm font-semibold text-white">Quality Conformance Matrix</h4>
              </div>
              <span
                className={`px-2.5 py-1 rounded-full text-[10px] font-mono font-bold border ${
                  currentInspection.conformance_report.lot_release_authorized
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-400/30'
                    : 'bg-rose-500/15 text-rose-300 border-rose-400/30'
                }`}
              >
                {currentInspection.conformance_report.lot_release_authorized
                  ? 'LOT RELEASE AUTHORIZED [TRUE]'
                  : 'LOT RELEASE QUARANTINED [FALSE]'}
              </span>
            </div>

            <div className="space-y-2">
              {currentInspection.conformance_report.items.map((it) => (
                <div
                  key={it.criterion_id}
                  className={`flex items-center justify-between p-3 rounded-xl border text-xs backdrop-blur-md transition-colors ${
                    it.passed
                      ? 'bg-emerald-950/20 border-emerald-500/20'
                      : 'bg-rose-950/20 border-rose-500/20'
                  }`}
                >
                  <div className="space-y-0.5">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-[10px] font-bold text-slate-400">
                        {it.criterion_id}
                      </span>
                      <span className="font-medium text-slate-200">{it.name}</span>
                    </div>
                    <p className="text-[10px] text-slate-400 leading-tight">{it.description}</p>
                  </div>
                  <span
                    className={`font-mono text-[10px] font-bold px-2.5 py-0.5 rounded-full shrink-0 ml-2 border ${
                      it.passed
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-400/30'
                        : 'bg-rose-500/15 text-rose-300 border-rose-400/30'
                    }`}
                  >
                    {it.status_label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* AI Root-Cause Diagnostic & 8D Report */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <h4 className="text-sm font-semibold text-white">AI Root-Cause Auditor (8D)</h4>
              {auditReport && (
                auditReport.source === 'vlm' ? (
                  <span className="px-2 py-0.5 rounded-full text-[10px] bg-purple-500/20 text-purple-200 border border-purple-400/30 font-mono">
                    VLM
                  </span>
                ) : auditReport.source === 'anomaly_cv' ? (
                  <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber-500/20 text-amber-300 border border-amber-400/30 font-mono">
                    Anomaly (CV)
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full text-[10px] bg-sky-500/20 text-sky-300 border border-sky-400/30 font-mono">
                    YOLO
                  </span>
                )
              )}
            </div>
            <div className="flex items-center space-x-2">
              <a
                href={`/api/detect/${currentInspection.id}/report/view`}
                target="_blank"
                rel="noreferrer"
                className="px-3 py-1.5 rounded-full bg-sky-500/20 hover:bg-sky-500/30 text-sky-200 text-xs font-semibold flex items-center space-x-1.5 transition-all border border-sky-400/30 shadow-[0_0_12px_rgba(56,189,248,0.2)]"
                title="Open full interactive ISO 9001 inspection certificate directly in a new browser tab"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                <span>View Full Report</span>
              </a>
              {auditReport && (
                <a
                  href={`/api/detect/${currentInspection.id}/8d-report/pdf?inline=true`}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] text-slate-200 text-xs font-medium flex items-center space-x-1 border border-white/[0.08] transition-all"
                  title="Open official vector PDF directly in browser"
                >
                  <Download className="w-3 h-3 text-sky-400" />
                  <span>PDF Report</span>
                </a>
              )}
              <button
                onClick={handleRunAudit}
                disabled={isAuditing}
                className="px-3 py-1.5 rounded-full bg-purple-500/25 hover:bg-purple-500/35 text-purple-200 text-xs font-medium flex items-center space-x-1 transition-all border border-purple-400/40 shadow-[0_0_12px_rgba(168,85,247,0.2)] disabled:opacity-50"
              >
                <Wrench className="w-3 h-3" />
                <span>{isAuditing ? 'Analyzing...' : auditReport ? 'Re-Audit' : 'Run 8D Audit'}</span>
              </button>
            </div>
          </div>

          {auditReport ? (
            <div className="space-y-3 text-xs">
              <div className="bg-purple-950/20 backdrop-blur-xl p-3.5 rounded-xl border border-purple-500/30 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono text-purple-300 uppercase font-bold">
                    {auditReport.primary_defect_type === 'unclassified_anomaly' ? 'Unclassified Surface Anomaly' : auditReport.primary_defect_type} Flaw Mechanism
                  </span>
                  <div className="flex items-center space-x-1.5">
                    {auditReport.source === 'vlm' ? (
                      <span className="px-2 py-0.5 rounded-full text-[10px] bg-purple-500/20 text-purple-200 border border-purple-400/30 font-mono">VLM</span>
                    ) : auditReport.source === 'anomaly_cv' ? (
                      <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber-500/20 text-amber-300 border border-amber-400/30 font-mono">Anomaly (CV)</span>
                    ) : (
                      <span className="px-2 py-0.5 rounded-full text-[10px] bg-sky-500/20 text-sky-300 border border-sky-400/30 font-mono">YOLO</span>
                    )}
                    <span className="px-2 py-0.5 rounded-full text-[10px] bg-white/[0.05] text-purple-200 border border-white/[0.1]">
                      {auditReport.standard_reference}
                    </span>
                  </div>
                </div>
                <p className="text-slate-200 font-medium">{auditReport.defect_mechanism}</p>
                <p className="text-slate-400 text-[11px]"><strong className="text-slate-300">Suspect Tooling:</strong> {auditReport.tooling_station}</p>
              </div>

              {/* D1 - D8 Disciplines Grid */}
              <div className="space-y-2.5">
                <div className="space-y-1">
                  <span className="text-sky-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D1 — Team Formation</span>
                  <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px]">
                    {auditReport.d1_team || 'Cross-Functional Metallurgical & Quality Team'}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-sky-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D2 — Problem Description</span>
                  <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px]">
                    {auditReport.d2_problem_description || auditReport.defect_mechanism}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-amber-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D3 — Interim Containment Actions</span>
                  <p className="text-amber-200/90 bg-amber-950/20 backdrop-blur-md p-2.5 rounded-xl border border-amber-500/20 text-[11px]">
                    {auditReport.d3_containment_action || auditReport.containment_action}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-purple-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D4 — Root Cause Analysis (Ishikawa / 5-Why)</span>
                  <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px] leading-relaxed">
                    {auditReport.d4_root_cause_analysis || auditReport.probable_root_cause}
                  </p>
                </div>

                <div className="space-y-1">
                  <span className="text-emerald-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D5 — Permanent Corrective Actions (PCA)</span>
                  <p className="text-emerald-200/90 bg-emerald-950/20 backdrop-blur-md p-2.5 rounded-xl border border-emerald-500/20 text-[11px] whitespace-pre-line font-mono">
                    {auditReport.d5_corrective_actions || auditReport.corrective_action}
                  </p>
                </div>

                {auditReport.d6_implementation_verification && (
                  <div className="space-y-1">
                    <span className="text-teal-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D6 — Implementation & Verification</span>
                    <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px]">
                      {auditReport.d6_implementation_verification}
                    </p>
                  </div>
                )}

                {auditReport.d7_preventive_action && (
                  <div className="space-y-1">
                    <span className="text-indigo-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D7 — Preventive Actions</span>
                    <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px]">
                      {auditReport.d7_preventive_action}
                    </p>
                  </div>
                )}

                {auditReport.d8_team_recognition && (
                  <div className="space-y-1">
                    <span className="text-rose-400 font-semibold text-[10px] uppercase font-mono tracking-wider">D8 — Team Recognition</span>
                    <p className="text-slate-300 bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-[11px]">
                      {auditReport.d8_team_recognition}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-4 space-y-1">
              <FileText className="w-8 h-8 text-slate-500 mx-auto" />
              <p className="text-xs text-slate-300 font-medium">No 8D Report Generated Yet</p>
              <p className="text-[11px] text-slate-500">Click &apos;Run 8D Audit&apos; for expert root-cause diagnosis & corrective maintenance plan.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

