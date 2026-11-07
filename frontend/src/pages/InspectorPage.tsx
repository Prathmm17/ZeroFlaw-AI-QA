import React, { useEffect, useState, useRef } from 'react';
import {
  UploadCloud,
  FileImage,
  RefreshCw,
  Sliders,
  CheckCircle,
  AlertOctagon,
  Sparkles,
  Info,
  ShieldAlert,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { DefectViewer } from '../components/DefectViewer';
import { Inspection } from '../types';

interface InspectorPageProps {
  currentInspection: Inspection | null;
  onInspectionLoaded: (inspection: Inspection) => void;
}

export const InspectorPage: React.FC<InspectorPageProps> = ({
  currentInspection,
  onInspectionLoaded,
}) => {
  const [samples, setSamples] = useState<string[]>([]);
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [notes, setNotes] = useState('');
  const [deepVerify, setDeepVerify] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    apiClient
      .getSamples()
      .then((data) => {
        setSamples(data);
        if (data.length > 0 && !selectedSample) {
          setSelectedSample(data[0]);
        }
      })
      .catch((err) => console.error('Failed to load samples:', err));
  }, []);

  const handleFileUpload = async (file: File) => {
    try {
      setLoading(true);
      setErrorMsg(null);
      const res = await apiClient.uploadImage(file, notes || undefined, deepVerify);
      onInspectionLoaded(res);
    } catch (err: any) {
      console.error('Upload failed:', err);
      setErrorMsg(err.response?.data?.detail || 'Failed to process and inspect image.');
    } finally {
      setLoading(false);
    }
  };

  const handleSampleInspect = async (sampleName: string) => {
    try {
      setLoading(true);
      setErrorMsg(null);
      setSelectedSample(sampleName);
      const res = await apiClient.inspectSample(sampleName, deepVerify);
      onInspectionLoaded(res);
    } catch (err: any) {
      console.error('Sample inspect failed:', err);
      setErrorMsg(err.response?.data?.detail || 'Failed to inspect sample.');
    } finally {
      setLoading(false);
    }
  };

  const onDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Industrial Image Inspector Studio</h2>
          <p className="text-xs text-slate-400 mt-1">
            Perform high-precision computer vision defect detection on industrial components via custom uploads or catalog samples.
          </p>
        </div>

        {/* Deep Verify Toggle */}
        <div className="flex items-center space-x-3 bg-white/[0.04] border border-white/[0.08] px-4 py-2 rounded-2xl text-xs font-mono backdrop-blur-xl shadow-inner">
          <label className="flex items-center space-x-2.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={deepVerify}
              onChange={(e) => setDeepVerify(e.target.checked)}
              className="rounded-lg bg-slate-950 border-white/20 text-purple-500 focus:ring-0 w-4 h-4 cursor-pointer"
            />
            <span className="text-slate-300 font-sans font-medium">Deep Verify (Tier 2 VLM / Anomaly)</span>
          </label>
          <span
            className={`px-2.5 py-0.5 rounded-full text-[10px] uppercase font-bold border transition-all ${
              deepVerify
                ? 'bg-purple-500/20 text-purple-300 border-purple-400/40 shadow-sm shadow-purple-500/20'
                : 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40 shadow-sm shadow-emerald-500/20'
            }`}
          >
            {deepVerify ? 'ON (VLM Tier 2)' : 'OFF (Fast Edge <150ms)'}
          </span>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/40 border border-rose-500/30 p-4 rounded-2xl text-rose-300 text-xs flex items-center space-x-2.5 backdrop-blur-xl shadow-lg shadow-rose-950/30">
          <AlertOctagon className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Input Controls Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload Dropzone */}
        <div
          onDragEnter={onDrag}
          onDragLeave={onDrag}
          onDragOver={onDrag}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-3xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center space-y-3.5 backdrop-blur-3xl shadow-[0_20px_50px_rgba(0,0,0,0.65),inset_0_1px_1px_rgba(255,255,255,0.12)] ${
            dragActive
              ? 'border-violet-400 bg-violet-600/25 ring-2 ring-violet-400/40 shadow-violet-500/30'
              : 'border-white/[0.12] bg-[#0e0a1c]/60 hover:border-violet-500/50 hover:bg-[#140e28]/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
              }
            }}
          />
          <div className="p-3.5 rounded-2xl bg-gradient-to-br from-violet-600/30 via-purple-600/30 to-indigo-600/30 text-violet-200 border border-violet-400/35 shadow-lg shadow-violet-500/25 transition-transform duration-200 group-hover:scale-105">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white drop-shadow-sm">Upload Specimen Image</p>
            <p className="text-xs text-violet-200/70 mt-0.5">Drag & drop or click to browse (PNG, JPG, TIFF, BMP)</p>
          </div>
          <span className="text-[11px] font-mono text-violet-200/80 bg-white/[0.04] px-3.5 py-1 rounded-full border border-violet-500/25 shadow-inner">
            Max 20MB &bull; 640x640+ optimal resolution
          </span>
        </div>

        {/* Demo Sample Catalog */}
        <div className="glass-panel p-6 rounded-3xl space-y-3.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-100 flex items-center space-x-1.5">
              <Sparkles className="w-4 h-4 text-violet-400" />
              <span>Demo Specimen Catalog (10 Synthetic Samples)</span>
            </span>
          </div>
          <p className="text-xs text-violet-200/70">
            Select a synthetic manufacturing specimen containing procedural cracks, scratches, dents, or chips:
          </p>

          <div className="grid grid-cols-2 gap-2.5 max-h-48 overflow-y-auto pr-1">
            {samples.map((sample) => (
              <button
                key={sample}
                onClick={() => handleSampleInspect(sample)}
                disabled={loading}
                className={`flex items-center space-x-2.5 p-2 rounded-xl border text-left text-xs transition-all duration-200 ${
                  selectedSample === sample
                    ? 'bg-gradient-to-r from-violet-600/30 to-purple-600/30 border-violet-400/60 text-white ring-1 ring-violet-400/50 shadow-md shadow-violet-500/25 font-semibold'
                    : 'bg-white/[0.03] border-white/[0.08] text-slate-300 hover:border-violet-400/40 hover:bg-[#140e28]/60'
                }`}
              >
                <img
                  src={`/data/samples/${sample}`}
                  alt={sample}
                  className="w-8 h-8 rounded-lg object-cover bg-black/60 shrink-0 ring-1 ring-white/10"
                />
                <span className="font-mono truncate">{sample}</span>
              </button>
            ))}
          </div>

          <div className="pt-2">
            <input
              type="text"
              placeholder="Optional inspection notes / batch ID..."
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/[0.08] rounded-xl px-3.5 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-violet-500/60 focus:ring-1 focus:ring-violet-400/40 transition-all shadow-inner"
            />
          </div>
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="bg-slate-900/50 border border-white/[0.08] p-8 rounded-2xl text-center space-y-3 backdrop-blur-2xl shadow-2xl">
          <RefreshCw className="w-8 h-8 text-sky-400 animate-spin mx-auto drop-shadow-md" />
          <p className="text-sm font-semibold text-white">Executing Real YOLOv8 AI Defect Inference...</p>
          <p className="text-xs text-slate-400 font-mono">Running feature pyramid network &amp; bounding box regression</p>
        </div>
      )}

      {/* Inspection Result Viewer */}
      {currentInspection && !loading && (
        <div className="space-y-4">
          {/* Domain Guard Out-of-Domain Notice */}
          {(currentInspection.domain_trust === 'LOW' || currentInspection.domain_warning) && (
            <div className="bg-amber-950/80 border border-amber-600/80 p-4 rounded-xl text-amber-200 text-xs flex items-center space-x-3 shadow-lg shadow-amber-950/40">
              <ShieldAlert className="w-5 h-5 shrink-0 text-amber-400" />
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-amber-300 uppercase tracking-wide px-2 py-0.5 rounded bg-amber-900/80 border border-amber-700 text-[10px]">
                    Domain Guard: Trust LOW
                  </span>
                  <span className="text-[11px] text-amber-400 font-mono">
                    Routing: FLAGGED for human review (never auto-REJECT)
                  </span>
                </div>
                <p className="mt-1 text-slate-200 font-medium">
                  {currentInspection.domain_warning || 'Image differs from the training domain (flat steel surface). Result may be unreliable.'}
                </p>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center space-x-2">
              <FileImage className="w-5 h-5 text-brand-blue" />
              <span>Active Inspection Result (#{currentInspection.id})</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">
              Timestamp: {new Date(currentInspection.timestamp).toLocaleString()}
            </span>
          </div>
          <DefectViewer inspection={currentInspection} />
        </div>
      )}
    </div>
  );
};

