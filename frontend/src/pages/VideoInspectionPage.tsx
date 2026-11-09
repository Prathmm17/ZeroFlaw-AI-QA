import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Clock,
  Film,
  Layers,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Sun,
  UploadCloud,
  Video,
} from 'lucide-react';
import { apiClient, isColdStartError, resolveMediaUrl, FALLBACK_IMAGE_DATA_URI } from '../api/client';
import { SeverityBadge } from '../components/SeverityBadge';
import { SeverityLevel, VideoJob, VideoTimelineItem } from '../types';

export const VideoInspectionPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sampleFps, setSampleFps] = useState<number>(3.0);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.25);
  const [ambientBoost, setAmbientBoost] = useState<boolean>(true);

  const [currentJob, setCurrentJob] = useState<VideoJob | null>(null);
  const [pastJobs, setPastJobs] = useState<VideoJob[]>([]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Video playback states
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [activeTimelineItem, setActiveTimelineItem] = useState<VideoTimelineItem | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>('all');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingTimerRef = useRef<number | null>(null);

  const fetchPastJobs = async () => {
    try {
      const jobs = await apiClient.getVideoJobs();
      setPastJobs(jobs || []);
    } catch (err) {
      console.error('Failed to load past video jobs:', err);
    }
  };

  useEffect(() => {
    fetchPastJobs();
    return () => {
      if (pollingTimerRef.current) clearTimeout(pollingTimerRef.current);
    };
  }, []);

  // Poll video status while processing with adaptive delay
  const startPollingStatus = (videoId: string) => {
    setIsProcessing(true);
    if (pollingTimerRef.current) clearTimeout(pollingTimerRef.current);

    const poll = async () => {
      try {
        const statusData = await apiClient.getVideoStatus(videoId);
        setProgress(statusData.progress || 0);
        setCurrentStage(statusData.current_stage || 'Processing video frames...');

        if (statusData.status === 'completed') {
          setIsProcessing(false);
          pollingTimerRef.current = null;
          const fullResult = await apiClient.getVideoResult(videoId);
          setCurrentJob(fullResult);
          fetchPastJobs();
          return;
        } else if (statusData.status === 'failed') {
          setIsProcessing(false);
          pollingTimerRef.current = null;
          setErrorMessage(statusData.error || 'Video defect processing failed');
          return;
        }
      } catch (err: any) {
        console.error('Polling error:', err);
      }
      pollingTimerRef.current = window.setTimeout(poll, 1200);
    };

    pollingTimerRef.current = window.setTimeout(poll, 800);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setErrorMessage(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setErrorMessage(null);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) return;
    if (selectedFile.size > 50 * 1024 * 1024) {
      setErrorMessage(`Video file is too large (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB). Please select a video under 50MB.`);
      return;
    }
    setIsUploading(true);
    setErrorMessage(null);
    setCurrentJob(null);

    try {
      const res = await apiClient.uploadVideo(
        selectedFile,
        sampleFps,
        confidenceThreshold,
        ambientBoost
      );
      setIsUploading(false);
      setProgress(5);
      setCurrentStage('Video uploaded, starting AI engine...');
      startPollingStatus(res.video_id);
    } catch (err: any) {
      setIsUploading(false);
      if (isColdStartError(err)) {
        setErrorMessage('The AI engine is waking up from sleep (Render free tier cold start ~30–50s). Please wait a few seconds and click Analyze again.');
      } else {
        setErrorMessage(err.response?.data?.detail || err.message || 'Upload failed');
      }
    }
  };

  // Seek video player to specific seconds
  const seekToTimestamp = (sec: number, item?: VideoTimelineItem) => {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      videoRef.current.play();
      setIsPlaying(true);
      if (item) setActiveTimelineItem(item);
    }
  };

  // Video time update listener
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const curr = videoRef.current.currentTime;
      setCurrentTime(curr);

      // Check if current time is near a timeline defect (within 0.5s)
      if (currentJob?.timeline) {
        const match = currentJob.timeline.find(
          (item) => Math.abs(item.timestamp_sec - curr) < 0.6
        );
        if (match) {
          setActiveTimelineItem(match);
        }
      }
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  };

  const filteredTimeline =
    currentJob?.timeline.filter((item) => {
      if (severityFilter === 'all') return true;
      return item.max_severity === severityFilter;
    }) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-5 rounded-2xl">
        <div>
          <div className="flex items-center space-x-2.5">
            <h2 className="text-xl font-bold text-white tracking-tight">Industrial Video Defect Inspection</h2>
            <span className="text-[10px] uppercase font-mono px-2.5 py-0.5 rounded-full font-bold bg-sky-500/15 text-sky-300 border border-sky-400/30">
              NEW
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Upload conveyor or line scan video files (.mp4, .mov, .avi) for automated frame-by-frame defect scanning and timeline playback.
          </p>
        </div>

        {/* Video Parameters Bar */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Ambient Boost CLAHE */}
          <label
            title="Enhance low-contrast surface details under diffuse room or line lighting"
            className={`flex items-center space-x-2 text-xs font-medium cursor-pointer px-3.5 py-2 rounded-full border transition-all backdrop-blur-xl ${
              ambientBoost
                ? 'bg-amber-500/20 border-amber-400/40 text-amber-200 shadow-[0_0_15px_rgba(245,158,11,0.25)]'
                : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:bg-white/[0.08]'
            }`}
          >
            <input
              type="checkbox"
              checked={ambientBoost}
              onChange={(e) => setAmbientBoost(e.target.checked)}
              disabled={isUploading || isProcessing}
              className="rounded bg-slate-900 border-white/[0.2] text-amber-500 focus:ring-0"
            />
            <span className="flex items-center space-x-1.5">
              <Sun className="w-3.5 h-3.5 text-amber-400" />
              <span>Ambient Boost (CLAHE)</span>
            </span>
          </label>

          {/* Min Confidence Slider */}
          <div className="flex items-center space-x-2 bg-white/[0.04] border border-white/[0.08] px-3.5 py-1.5 rounded-full text-xs backdrop-blur-xl">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400 font-medium">Min Conf:</span>
            <input
              type="range"
              min="0.15"
              max="0.65"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              disabled={isUploading || isProcessing}
              className="w-16 accent-sky-500 cursor-pointer"
            />
            <span className="font-mono text-sky-400 font-bold w-8 text-right">
              {Math.round(confidenceThreshold * 100)}%
            </span>
          </div>

          {/* Sample FPS Selector */}
          <div className="flex items-center space-x-1.5 bg-white/[0.04] border border-white/[0.08] px-3.5 py-1.5 rounded-full text-xs backdrop-blur-xl">
            <Film className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400 font-medium">Sample:</span>
            <select
              value={sampleFps}
              onChange={(e) => setSampleFps(parseFloat(e.target.value))}
              disabled={isUploading || isProcessing}
              className="bg-slate-900/90 border border-white/[0.1] rounded-lg text-slate-200 text-xs py-0.5 px-2 font-mono focus:ring-0"
            >
              <option value="1.0">1 FPS</option>
              <option value="2.0">2 FPS</option>
              <option value="3.0">3 FPS (Recommended)</option>
              <option value="5.0">5 FPS (High Def)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Upload & Drag-and-Drop Dropzone */}
      {!currentJob && !isProcessing && (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-white/[0.12] hover:border-sky-400/60 bg-white/[0.02] hover:bg-white/[0.05] backdrop-blur-2xl rounded-3xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-300 group shadow-[0_8px_32px_rgba(0,0,0,0.36)]"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="mx-auto w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-400/20 flex items-center justify-center text-sky-400 group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(56,189,248,0.2)]">
            <UploadCloud className="w-8 h-8" />
          </div>

          <div className="mt-5 space-y-1.5">
            <p className="text-sm font-semibold text-white">
              {selectedFile ? selectedFile.name : 'Click to select or drag and drop industrial video'}
            </p>
            <p className="text-xs text-slate-400">
              Supports MP4, MOV, AVI, WEBM, MKV (Recommended: Conveyor runs & line-scan feeds)
            </p>
          </div>

          {selectedFile && (
            <div className="mt-4 inline-flex items-center space-x-2 bg-white/[0.05] border border-white/[0.1] px-4 py-1.5 rounded-full text-xs text-slate-300 font-mono">
              <Film className="w-3.5 h-3.5 text-sky-400" />
              <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
          )}

          <div className="mt-6">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleUploadAndAnalyze();
              }}
              disabled={!selectedFile || isUploading}
              className="inline-flex items-center space-x-2 bg-sky-500/25 hover:bg-sky-500/35 disabled:opacity-50 text-sky-200 px-7 py-3 rounded-full text-xs font-bold transition-all border border-sky-400/30 shadow-[0_0_20px_rgba(56,189,248,0.25)]"
            >
              <Sparkles className="w-4 h-4" />
              <span>{isUploading ? 'Uploading Video...' : 'Analyze Video with YOLO'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Past Completed Video Analysis Runs */}
      {!currentJob && !isProcessing && pastJobs.length > 0 && (
        <div className="glass-panel p-5 rounded-2xl space-y-3">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center space-x-2">
              <Film className="w-4 h-4 text-sky-400" />
              <h3 className="text-sm font-bold text-white tracking-wider uppercase">
                Past Video Analysis Runs ({pastJobs.length})
              </h3>
            </div>
            <button
              onClick={fetchPastJobs}
              className="text-xs text-slate-400 hover:text-white flex items-center space-x-1"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Refresh</span>
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {pastJobs.map((job) => (
              <button
                key={job.video_id}
                onClick={async () => {
                  try {
                    const fullJob = await apiClient.getVideoResult(job.video_id);
                    setCurrentJob(fullJob);
                  } catch (err: any) {
                    setErrorMessage('Failed to load past video result: file may have expired on ephemeral storage.');
                  }
                }}
                className="p-3.5 rounded-xl border border-white/[0.08] bg-white/[0.03] hover:bg-white/[0.07] hover:border-sky-400/40 text-left transition-all space-y-2 group"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-white font-mono truncate max-w-[12rem]">{job.filename}</span>
                  <SeverityBadge level={job.max_severity || 'none'} size="sm" />
                </div>
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>Defects: <strong className="text-rose-400">{job.total_defects_found || 0}</strong></span>
                  <span>Frames: {job.sampled_frames || 0}</span>
                </div>
                <span className="text-[10px] text-sky-400 group-hover:underline font-semibold block pt-1">
                  View Timeline &amp; Playback →
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Progress & Processing HUD */}
      {isProcessing && (
        <div className="glass-panel p-8 rounded-2xl text-center space-y-4">
          <div className="w-12 h-12 rounded-full border-4 border-sky-400 border-t-transparent animate-spin mx-auto" />
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Scanning Video Frame-by-Frame
            </h3>
            <p className="text-xs text-slate-400 mt-1 font-mono">{currentStage}</p>
          </div>

          <div className="max-w-md mx-auto space-y-2">
            <div className="w-full bg-black/40 rounded-full h-2.5 border border-white/[0.08] overflow-hidden p-0.5">
              <div
                className="bg-gradient-to-r from-sky-400 to-brand-cyan h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] font-mono text-slate-400">
              <span>Extracting & Verifying</span>
              <span className="font-bold text-sky-400">{progress}%</span>
            </div>
          </div>
        </div>
      )}

      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-rose-950 border border-rose-600 p-4 rounded-xl flex items-center justify-between text-xs text-rose-200">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-white hover:text-rose-200 text-xs font-bold"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Inspection Results Dashboard */}
      {currentJob && (
        <div className="space-y-6">
          {/* Top Bar with Re-analyze button */}
          <div className="flex items-center justify-between glass-panel p-4 rounded-2xl">
            <div className="flex items-center space-x-3">
              <span className="text-xs font-mono text-slate-400">Specimen:</span>
              <span className="text-xs font-bold font-mono text-white">{currentJob.filename}</span>
              <span className="text-white/[0.15]">|</span>
              <span className="text-xs font-mono text-slate-400">Duration: {currentJob.duration_seconds}s</span>
              <span className="text-white/[0.15]">|</span>
              <span className="text-xs font-mono text-slate-400">Sampled: {currentJob.sampled_frames} frames</span>
            </div>

            <button
              onClick={() => {
                setCurrentJob(null);
                setSelectedFile(null);
              }}
              className="flex items-center space-x-1.5 text-xs text-slate-300 hover:text-white px-3.5 py-1.5 rounded-full border border-white/[0.08] bg-white/[0.04] hover:bg-white/[0.08] transition-all backdrop-blur-xl"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Inspect Another Video</span>
            </button>
          </div>

          {/* Main Video Viewport & Telemetry Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Video Player + Scrubber (2 Cols) */}
            <div className="lg:col-span-2 space-y-3">
              <div className="relative aspect-video rounded-2xl overflow-hidden border border-violet-500/20 bg-[#090614]/85 backdrop-blur-3xl flex items-center justify-center shadow-[0_16px_40px_rgba(0,0,0,0.7)]">
                <video
                  ref={videoRef}
                  src={resolveMediaUrl(currentJob.video_url)}
                  playsInline
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onError={() => {
                    setErrorMessage('Video streaming error: specimen file may have expired on ephemeral storage or format is incompatible.');
                  }}
                  className="w-full h-full object-contain"
                />

                {/* Overlay Badge for Current Defect Instance */}
                {activeTimelineItem && (
                  <div className="absolute top-3 left-3 bg-[#1e2a3e]/80 backdrop-blur-3xl px-3.5 py-2 rounded-xl border border-white/[0.22] text-xs font-mono space-y-1 shadow-xl z-20 pointer-events-none">
                    <div className="flex items-center space-x-2">
                      <span className="text-slate-400">AT:</span>
                      <span className="text-sky-400 font-bold">{formatTime(activeTimelineItem.timestamp_sec)}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-slate-400">FLAWS:</span>
                      <span className="text-rose-400 font-bold">{activeTimelineItem.defect_count}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Custom Scrubber with Clickable Defect Timeline Markers */}
              <div className="glass-panel p-4 rounded-2xl space-y-3">
                <div className="flex items-center justify-between text-xs font-mono text-slate-400">
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={togglePlay}
                      className="p-2 rounded-full bg-white/[0.05] hover:bg-white/[0.1] text-white border border-white/[0.08] transition-all"
                    >
                      {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                    </button>
                    <span className="text-slate-300 font-medium">{formatTime(currentTime)} / {formatTime(duration || currentJob.duration_seconds)}</span>
                  </div>

                  {currentJob.summary?.peak_timestamp_sec > 0 && (
                    <button
                      onClick={() => seekToTimestamp(currentJob.summary.peak_timestamp_sec)}
                      className="text-[11px] text-amber-400 hover:text-amber-300 underline font-semibold"
                    >
                      Jump to Peak Defect ({currentJob.summary.peak_timestamp_sec}s)
                    </button>
                  )}
                </div>

                {/* Progress Bar with Interactive Timeline Dots */}
                <div className="relative w-full h-5 flex items-center select-none">
                  {/* Base track */}
                  <input
                    type="range"
                    min="0"
                    max={duration || currentJob.duration_seconds || 1}
                    step="0.05"
                    value={currentTime}
                    onChange={(e) => {
                      const t = parseFloat(e.target.value);
                      seekToTimestamp(t);
                    }}
                    className="w-full h-2 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-sky-500 z-10"
                  />

                  {/* Render Defect Markers on Progress Track */}
                  {currentJob.timeline.map((item, idx) => {
                    const videoDuration = duration || currentJob.duration_seconds || 1;
                    const percent = Math.min(100, Math.max(0, (item.timestamp_sec / videoDuration) * 100));
                    const dotColor =
                      item.max_severity === 'critical'
                        ? 'bg-rose-500 border-rose-300'
                        : item.max_severity === 'major'
                        ? 'bg-amber-500 border-amber-300'
                        : 'bg-sky-500 border-sky-300';

                    return (
                      <button
                        key={idx}
                        title={`Defect at ${item.timestamp_sec}s (${item.defect_count} flaws, ${item.max_severity})`}
                        onClick={() => seekToTimestamp(item.timestamp_sec, item)}
                        style={{ left: `${percent}%` }}
                        className={`absolute -top-0.5 w-3 h-3 rounded-full border-2 ${dotColor} -translate-x-1/2 z-20 hover:scale-150 transition-transform cursor-pointer shadow-sm`}
                      />
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Quality Summary & Specimen Verdict Card (1 Col) */}
            <div className="space-y-4">
              <div className="glass-panel p-5 rounded-2xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <h4 className="text-sm font-bold uppercase tracking-wider text-white">Video Quality Verdict</h4>
                  <SeverityBadge level={currentJob.max_severity} size="sm" />
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] backdrop-blur-xl border border-white/[0.06] space-y-2.5 text-xs font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Total Flaws Found:</span>
                    <span className="text-white font-bold text-sm">{currentJob.total_defects_found}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Frames Analyzed:</span>
                    <span className="text-slate-200">{currentJob.sampled_frames}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Defect Density:</span>
                    <span className="text-slate-200">
                      {currentJob.duration_seconds > 0
                        ? (currentJob.total_defects_found / currentJob.duration_seconds).toFixed(2)
                        : 0}{' '}
                      flaws/sec
                    </span>
                  </div>
                </div>

                {/* Class Distribution Breakdown */}
                {currentJob.summary?.by_class && Object.keys(currentJob.summary.by_class).length > 0 && (
                  <div className="space-y-2 pt-1">
                    <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                      Flaw Distribution
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(currentJob.summary.by_class).map(([cName, count]) => (
                        <span
                          key={cName}
                          className="px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs font-mono text-slate-300"
                        >
                          {cName}: <strong className="text-sky-400">{count}</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Defect Occurrence Timeline Strip & Gallery */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.06] pb-3">
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-sky-400" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Defect Timeline Occurrences ({filteredTimeline.length})
                </h3>
              </div>

              {/* Severity Filter Pills */}
              <div className="flex items-center space-x-1.5 bg-white/[0.04] p-1 rounded-full border border-white/[0.08] backdrop-blur-xl">
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

            {/* Gallery Grid of Defect Moments */}
            {filteredTimeline.length === 0 ? (
              <p className="text-xs text-slate-400 italic py-4">No defects matched the selected filter.</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {filteredTimeline.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => seekToTimestamp(item.timestamp_sec, item)}
                    className={`group bg-slate-950 rounded-xl overflow-hidden border p-2 space-y-2 cursor-pointer transition-all hover:scale-[1.02] ${
                      activeTimelineItem?.frame_idx === item.frame_idx
                        ? 'border-sky-500 shadow-md shadow-sky-950'
                        : 'border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="relative aspect-video rounded-lg overflow-hidden bg-black">
                      <img
                        src={resolveMediaUrl(item.thumbnail_url)}
                        alt={`Defect at ${item.timestamp_sec}s`}
                        onError={(e) => {
                          const img = e.currentTarget;
                          img.onerror = null;
                          img.src = FALLBACK_IMAGE_DATA_URI;
                        }}
                        className="w-full h-full object-cover"
                      />
                      <span className="absolute bottom-1.5 right-1.5 bg-black/80 px-2 py-0.5 rounded text-[10px] font-mono font-bold text-white">
                        {formatTime(item.timestamp_sec)}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-1.5">
                        <span className="font-bold text-white capitalize">
                          {item.defects[0]?.class_name || 'Defect'}
                        </span>
                        {item.defect_count > 1 && (
                          <span className="text-[10px] font-mono text-slate-400">
                            +{item.defect_count - 1}
                          </span>
                        )}
                      </div>
                      <SeverityBadge level={item.max_severity} size="sm" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

