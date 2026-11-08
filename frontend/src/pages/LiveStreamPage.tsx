import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Camera,
  Play,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Sparkles,
  Square,
  Sun,
  Target,
  Video,
  Zap,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { SeverityBadge } from '../components/SeverityBadge';
import { Defect, Inspection, SeverityLevel } from '../types';

interface ProductionRouting {
  action: 'PASS' | 'REWORK' | 'REJECT';
  signal: 'CONTINUE' | 'DIVERT_PNEUMATIC' | 'EMERGENCY_HALT';
  target_station: string;
  reason: string;
  state_color: string;
  dispatch_latency_ms: number;
}

interface LiveStreamPageProps {
  onSelectInspection?: (inspection: Inspection) => void;
}

export const LiveStreamPage: React.FC<LiveStreamPageProps> = ({ onSelectInspection }) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [droppedFrames, setDroppedFrames] = useState(0);
  const [totalDefects, setTotalDefects] = useState(0);
  const [maxSeverity, setMaxSeverity] = useState<SeverityLevel>('none');
  const [verdict, setVerdict] = useState<string>('passed');
  const [hasCritical, setHasCritical] = useState(false);
  const [lineHalted, setLineHalted] = useState(false);
  const [activeDefects, setActiveDefects] = useState<Defect[]>([]);
  const [useWebcam, setUseWebcam] = useState(false);
  const [useTargetRoi, setUseTargetRoi] = useState(true);
  const [sensitivity, setSensitivity] = useState(0.20);
  const [ambientBoost, setAmbientBoost] = useState(true);
  const [isSnapping, setIsSnapping] = useState(false);
  const [productionRouting, setProductionRouting] = useState<ProductionRouting>({
    action: 'PASS',
    signal: 'CONTINUE',
    target_station: 'CONVEYOR_CONTINUOUS',
    reason: 'Specimen conforming to tolerances — continuous feed authorized.',
    state_color: 'green',
    dispatch_latency_ms: 0.1,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const conveyorCanvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const viewportContainerRef = useRef<HTMLDivElement>(null);
  const currentFrameDims = useRef<{ width: number; height: number }>({ width: 640, height: 480 });
  const useTargetRoiRef = useRef(true);

  const lastRoiMapping = useRef({
    isRoi: false,
    srcX: 0,
    srcY: 0,
    srcW: 640,
    srcH: 480,
    capW: 640,
    capH: 480,
    videoW: 640,
    videoH: 480,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const isWaitingForResponse = useRef(false);
  const droppedFramesRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const fpsWindowRef = useRef<number[]>([]);

  useEffect(() => {
    useTargetRoiRef.current = useTargetRoi;
  }, [useTargetRoi]);

  // Push updated sensitivity & ambient boost dynamically over WebSocket
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          confidence_threshold: sensitivity,
          ambient_boost: ambientBoost,
        })
      );
    }
  }, [sensitivity, ambientBoost]);

  // Stop stream and clean up media and websocket
  const stopStream = () => {
    setIsStreaming(false);
    isWaitingForResponse.current = false;
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    // Clear overlay canvas
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
      }
    }
  };

  // Draw bounding boxes on canvas overlay with precise object-contain letterbox compensation
  const renderOverlayBoxes = (defects: Defect[], frameW = 640, frameH = 480) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    // Synchronize overlay canvas resolution with its rendered DOM size
    if (overlay.width !== overlay.clientWidth || overlay.height !== overlay.clientHeight) {
      overlay.width = overlay.clientWidth;
      overlay.height = overlay.clientHeight;
    }

    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!defects || defects.length === 0) return;

    // Compute exact rendered dimensions and letterbox/pillarbox offsets within container
    const containerW = overlay.clientWidth;
    const containerH = overlay.clientHeight;
    if (containerW === 0 || containerH === 0) return;

    const rVideo = frameW / frameH;
    const rContainer = containerW / containerH;

    let renderW = containerW;
    let renderH = containerH;
    let offsetX = 0;
    let offsetY = 0;

    if (rContainer > rVideo) {
      // Container wider than video -> pillarboxed (bars on left/right)
      renderH = containerH;
      renderW = containerH * rVideo;
      offsetX = (containerW - renderW) / 2;
    } else {
      // Container taller than video -> letterboxed (bars on top/bottom)
      renderW = containerW;
      renderH = containerW / rVideo;
      offsetY = (containerH - renderH) / 2;
    }

    const scaleX = renderW / frameW;
    const scaleY = renderH / frameH;

    for (const def of defects) {
      const [bx1, by1, bx2, by2] = def.bbox;
      const x1 = Math.round(offsetX + bx1 * scaleX);
      const y1 = Math.round(offsetY + by1 * scaleY);
      const x2 = Math.round(offsetX + bx2 * scaleX);
      const y2 = Math.round(offsetY + by2 * scaleY);
      const boxW = Math.max(10, x2 - x1);
      const boxH = Math.max(10, y2 - y1);

      const color =
        def.severity === 'critical'
          ? '#f43f5e'
          : def.severity === 'major'
          ? '#f59e0b'
          : '#0284c7';

      // 1. Draw outer glowing bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.strokeRect(x1, y1, boxW, boxH);
      ctx.shadowBlur = 0; // reset shadow

      // 2. Draw label pill
      const label = `${def.class_name.toUpperCase()} ${(def.confidence * 100).toFixed(0)}%`;
      ctx.font = 'bold 11px monospace';
      const textMetrics = ctx.measureText(label);
      const badgeW = textMetrics.width + 12;
      const badgeH = 20;
      const badgeY = Math.max(0, y1 - badgeH);

      ctx.fillStyle = color;
      ctx.fillRect(x1, badgeY, badgeW, badgeH);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 6, badgeY + 14);
    }
  };

  // Start live stream
  const startStream = async () => {
    try {
      setLineHalted(false);
      droppedFramesRef.current = 0;
      setDroppedFrames(0);
      isWaitingForResponse.current = false;
      fpsWindowRef.current = [];

      const apiBase = import.meta.env.VITE_API_BASE_URL;
      let wsUrl: string;
      if (apiBase && (apiBase.startsWith('http://') || apiBase.startsWith('https://'))) {
        const urlObj = new URL(apiBase);
        const wsProto = urlObj.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${wsProto}//${urlObj.host}/ws/stream`;
      } else {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//${window.location.host}/ws/stream`;
      }
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket binary stream connected');
        setIsStreaming(true);
        // Send initial dynamic threshold and ambient boost settings
        ws.send(
          JSON.stringify({
            type: 'config',
            confidence_threshold: sensitivity,
            ambient_boost: ambientBoost,
          })
        );
        if (useWebcam) {
          startWebcamCapture();
        } else {
          startSimulatedConveyor();
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'pong' || data.type === 'config_ack') return;

          isWaitingForResponse.current = false; // Release backpressure gate

          // Compute smooth sliding-window FPS
          const now = performance.now();
          const delta = (now - lastFrameTimeRef.current) / 1000.0;
          lastFrameTimeRef.current = now;
          if (delta > 0 && delta < 1.0) {
            const currentFps = 1.0 / delta;
            fpsWindowRef.current.push(currentFps);
            if (fpsWindowRef.current.length > 10) fpsWindowRef.current.shift();
            const avgFps =
              fpsWindowRef.current.reduce((a, b) => a + b, 0) / fpsWindowRef.current.length;
            setFps(Math.round(avgFps * 10) / 10);
          }

          if (data.inference_time_ms) setLatency(data.inference_time_ms);
          if (data.total_defects !== undefined) setTotalDefects(data.total_defects);
          if (data.max_severity) setMaxSeverity(data.max_severity);
          if (data.verdict) setVerdict(data.verdict);
          if (data.production_routing) setProductionRouting(data.production_routing);
          if (data.has_critical || data.production_routing?.action === 'REJECT') {
            setHasCritical(true);
            setLineHalted(true); // Automatic line halt on critical flaw (<10ms broadcast)
          }
          if (data.defects) {
            let mappedDefects: Defect[] = data.defects;
            const roi = lastRoiMapping.current;
            if (roi.isRoi && roi.capW > 0 && roi.capH > 0) {
              const scaleX = roi.srcW / roi.capW;
              const scaleY = roi.srcH / roi.capH;
              mappedDefects = data.defects.map((def: Defect) => {
                const [bx1, by1, bx2, by2] = def.bbox;
                const x1 = Math.round(roi.srcX + bx1 * scaleX);
                const y1 = Math.round(roi.srcY + by1 * scaleY);
                const x2 = Math.round(roi.srcX + bx2 * scaleX);
                const y2 = Math.round(roi.srcY + by2 * scaleY);
                return {
                  ...def,
                  bbox: [x1, y1, x2, y2],
                };
              });
            }
            setActiveDefects(mappedDefects);
            renderOverlayBoxes(
              mappedDefects,
              roi.isRoi ? roi.videoW : currentFrameDims.current.width,
              roi.isRoi ? roi.videoH : currentFrameDims.current.height
            );
          }
        } catch (err) {
          console.error('Error parsing stream message:', err);
          isWaitingForResponse.current = false;
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        stopStream();
      };

      ws.onclose = () => {
        setIsStreaming(false);
        isWaitingForResponse.current = false;
      };
    } catch (err) {
      console.error('Failed to start stream:', err);
      stopStream();
    }
  };

  const startWebcamCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30 },
        },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        runWebcamCaptureLoop();
      }
    } catch (err) {
      console.warn('Webcam inaccessible, falling back to simulated conveyor:', err);
      setUseWebcam(false);
      startSimulatedConveyor();
    }
  };

  // Video capture loop with natural aspect ratio preservation & backpressure
  const runWebcamCaptureLoop = () => {
    const video = videoRef.current;
    const captureCanvas = captureCanvasRef.current;
    if (!video || !captureCanvas) return;
    const captureCtx = captureCanvas.getContext('2d');
    if (!captureCtx) return;

    const tick = () => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      if (video.videoWidth > 0 && video.videoHeight > 0) {
        const videoW = video.videoWidth;
        const videoH = video.videoHeight;

        let srcX = 0;
        let srcY = 0;
        let srcW = videoW;
        let srcH = videoH;
        const isRoi = useTargetRoiRef.current;

        if (isRoi) {
          // Specimen Target Box: central 60% width x 70% height
          srcW = Math.round(videoW * 0.60);
          srcH = Math.round(videoH * 0.70);
          srcX = Math.round((videoW - srcW) / 2);
          srcY = Math.round((videoH - srcH) / 2);
        }

        // Preserve aspect ratio up to max dimension 640
        const maxDim = 640;
        let capW = srcW;
        let capH = srcH;

        if (srcW >= srcH) {
          capW = Math.min(srcW, maxDim);
          capH = Math.round((capW / srcW) * srcH);
        } else {
          capH = Math.min(srcH, maxDim);
          capW = Math.round((capH / srcH) * srcW);
        }

        if (captureCanvas.width !== capW || captureCanvas.height !== capH) {
          captureCanvas.width = capW;
          captureCanvas.height = capH;
          currentFrameDims.current = { width: capW, height: capH };
        }

        // Backpressure: If server has not replied to prior frame, drop stale frame!
        if (isWaitingForResponse.current) {
          droppedFramesRef.current += 1;
          setDroppedFrames(droppedFramesRef.current);
        } else {
          captureCtx.drawImage(video, srcX, srcY, srcW, srcH, 0, 0, capW, capH);
          isWaitingForResponse.current = true;

          lastRoiMapping.current = {
            isRoi,
            srcX,
            srcY,
            srcW,
            srcH,
            capW,
            capH,
            videoW,
            videoH,
          };

          captureCanvas.toBlob(
            (blob) => {
              if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(blob);
              } else {
                isWaitingForResponse.current = false;
              }
            },
            'image/jpeg',
            0.85
          );
        }
      }

      animationFrameId.current = requestAnimationFrame(tick);
    };

    animationFrameId.current = requestAnimationFrame(tick);
  };

  // Simulated industrial conveyor stream loop with procedural flaws
  const startSimulatedConveyor = () => {
    const conveyorCanvas = conveyorCanvasRef.current;
    const captureCanvas = captureCanvasRef.current;
    if (!conveyorCanvas || !captureCanvas) return;
    const ctx = conveyorCanvas.getContext('2d');
    const captureCtx = captureCanvas.getContext('2d');
    if (!ctx || !captureCtx) return;

    conveyorCanvas.width = 640;
    conveyorCanvas.height = 480;
    captureCanvas.width = 640;
    captureCanvas.height = 480;
    currentFrameDims.current = { width: 640, height: 480 };

    let posX = 0;
    const tick = () => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      // 1. Draw synthetic industrial steel conveyor belt moving across canvas
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, 640, 480);

      // Conveyor steel roller marks
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 4;
      for (let y = 0; y < 480; y += 32) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(640, y);
        ctx.stroke();
      }

      // Moving cold-rolled steel specimen
      posX = (posX + 6) % 760;
      const partLeft = posX - 160;
      ctx.fillStyle = '#94a3b8';
      ctx.fillRect(partLeft, 110, 240, 260);

      // Specimen steel grain texture
      ctx.strokeStyle = '#cbd5e1';
      ctx.lineWidth = 1;
      for (let i = 0; i < 240; i += 16) {
        ctx.beginPath();
        ctx.moveTo(partLeft + i, 110);
        ctx.lineTo(partLeft + i + 8, 370);
        ctx.stroke();
      }

      // Procedural surface defect: Longitudinal stress fracture crack on specimen
      ctx.strokeStyle = '#020617';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(partLeft + 40, 160);
      ctx.lineTo(partLeft + 80, 210);
      ctx.lineTo(partLeft + 110, 200);
      ctx.lineTo(partLeft + 150, 270);
      ctx.lineTo(partLeft + 190, 310);
      ctx.stroke();

      // Pitting flaw
      ctx.fillStyle = '#020617';
      ctx.beginPath();
      ctx.arc(partLeft + 160, 180, 14, 0, Math.PI * 2);
      ctx.fill();

      // 2. Backpressure Frame Dispatch
      if (isWaitingForResponse.current) {
        droppedFramesRef.current += 1;
        setDroppedFrames(droppedFramesRef.current);
      } else {
        captureCtx.drawImage(conveyorCanvas, 0, 0, 640, 480);
        isWaitingForResponse.current = true;

        captureCanvas.toBlob(
          (blob) => {
            if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(blob);
            } else {
              isWaitingForResponse.current = false;
            }
          },
          'image/jpeg',
          0.82
        );
      }

      animationFrameId.current = requestAnimationFrame(tick);
    };

    animationFrameId.current = requestAnimationFrame(tick);
  };

  // Snapshot live frame and trigger deep Tier 2 8D inspection
  const handleDeepAuditSnapshot = async () => {
    const captureCanvas = captureCanvasRef.current;
    if (!captureCanvas || isSnapping) return;
    setIsSnapping(true);
    try {
      captureCanvas.toBlob(
        async (blob) => {
          if (!blob) {
            setIsSnapping(false);
            return;
          }
          const file = new File([blob], `live_snapshot_${Date.now()}.jpg`, { type: 'image/jpeg' });
          const inspection = await apiClient.uploadImage(
            file,
            'Live Camera Stream Snapshot — 8D Deep Metrology Audit',
            true
          );
          setIsSnapping(false);
          if (onSelectInspection) {
            onSelectInspection(inspection);
          }
        },
        'image/jpeg',
        0.95
      );
    } catch (err) {
      console.error('Failed to capture snapshot for deep audit:', err);
      setIsSnapping(false);
    }
  };

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Stream Controls Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Real-Time Conveyor Video Inspection</h2>
          <p className="text-xs text-slate-400 mt-1">
            Ultra-low latency streaming defect detection powered by ONNX YOLO runtime, aspect-ratio preservation, and ambient boost.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Webcam vs Simulation toggle */}
          <label className="flex items-center space-x-2 text-xs text-slate-300 font-medium cursor-pointer bg-white/[0.04] border border-white/[0.08] px-3.5 py-2 rounded-full hover:bg-white/[0.08] transition-all backdrop-blur-xl">
            <input
              type="checkbox"
              checked={useWebcam}
              onChange={(e) => setUseWebcam(e.target.checked)}
              disabled={isStreaming}
              className="rounded bg-slate-900 border-white/[0.2] text-sky-500 focus:ring-0"
            />
            <span className="flex items-center space-x-1.5">
              <Camera className="w-3.5 h-3.5 text-sky-400" />
              <span>Use Physical Webcam</span>
            </span>
          </label>

          {/* Ambient CLAHE Boost */}
          <label
            title="Enhance low-contrast surface details (scratches, cracks) under ambient room lighting"
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
              className="rounded bg-slate-900 border-white/[0.2] text-amber-500 focus:ring-0"
            />
            <span className="flex items-center space-x-1.5">
              <Sun className="w-3.5 h-3.5 text-amber-400" />
              <span>Ambient Boost (CLAHE)</span>
            </span>
          </label>

          {/* Target Focus ROI Toggle (Only when webcam is active) */}
          {useWebcam && (
            <label
              title="Focus high-resolution AI optical inspection on specimen center zone and ignore room background"
              className={`flex items-center space-x-2 text-xs font-medium cursor-pointer px-3.5 py-2 rounded-full border transition-all backdrop-blur-xl ${
                useTargetRoi
                  ? 'bg-sky-500/20 border-sky-400/40 text-sky-200 shadow-[0_0_15px_rgba(56,189,248,0.25)]'
                  : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:bg-white/[0.08]'
              }`}
            >
              <input
                type="checkbox"
                checked={useTargetRoi}
                onChange={(e) => setUseTargetRoi(e.target.checked)}
                className="rounded bg-slate-900 border-white/[0.2] text-sky-500 focus:ring-0"
              />
              <span className="flex items-center space-x-1.5">
                <Target className="w-3.5 h-3.5 text-sky-400" />
                <span>Target ROI Focus</span>
              </span>
            </label>
          )}

          {/* Sensitivity Slider */}
          <div
            title="Adjust AI defect detection sensitivity threshold"
            className="flex items-center space-x-2 bg-white/[0.04] border border-white/[0.08] px-3.5 py-1.5 rounded-full text-xs backdrop-blur-xl"
          >
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400 font-medium">Min Conf:</span>
            <input
              type="range"
              min="0.10"
              max="0.75"
              step="0.05"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseFloat(e.target.value))}
              className="w-20 accent-sky-500 cursor-pointer"
            />
            <span className="font-mono text-sky-400 font-bold w-9 text-right">
              {Math.round(sensitivity * 100)}%
            </span>
          </div>

          {/* Freeze & Deep Audit Button (Only visible while streaming) */}
          {isStreaming && (
            <button
              onClick={handleDeepAuditSnapshot}
              disabled={isSnapping}
              title="Capture current frame and execute Tier 2 8D metrology audit"
              className="flex items-center space-x-1.5 bg-emerald-500/25 hover:bg-emerald-500/35 text-emerald-200 px-4 py-2 rounded-full text-xs font-bold transition-all border border-emerald-400/30 shadow-[0_0_15px_rgba(16,185,129,0.25)] disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5 fill-current" />
              <span>{isSnapping ? 'Auditing...' : 'Freeze & Deep Inspect'}</span>
            </button>
          )}

          {isStreaming ? (
            <button
              onClick={stopStream}
              className="flex items-center space-x-2 bg-rose-500/25 hover:bg-rose-500/35 text-rose-200 px-5 py-2 rounded-full text-xs font-bold transition-all border border-rose-400/30 shadow-[0_0_15px_rgba(244,63,94,0.25)]"
            >
              <Square className="w-4 h-4 fill-current" />
              <span>Halt Stream</span>
            </button>
          ) : (
            <button
              onClick={startStream}
              className="flex items-center space-x-2 bg-sky-500/25 hover:bg-sky-500/35 text-sky-200 px-5 py-2 rounded-full text-xs font-bold transition-all border border-sky-400/30 shadow-[0_0_15px_rgba(56,189,248,0.25)]"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Start Live Stream (15+ FPS)</span>
            </button>
          )}
        </div>
      </div>

      {/* Emergency Line Halt Banner */}
      {lineHalted && (
        <div className="bg-rose-950/40 border border-rose-500/50 p-5 rounded-2xl flex items-center justify-between shadow-[0_0_30px_rgba(244,63,94,0.3)] backdrop-blur-2xl animate-pulse">
          <div className="flex items-center space-x-3">
            <ShieldAlert className="w-6 h-6 text-rose-400 shrink-0" />
            <div>
              <p className="text-sm font-bold text-white uppercase tracking-wider">CONVEYOR LINE HALTED BY AI</p>
              <p className="text-xs text-rose-200">Critical flaw detected exceeding manufacturing tolerance threshold. Line quarantined.</p>
            </div>
          </div>
          <button
            onClick={() => {
              setLineHalted(false);
              setHasCritical(false);
            }}
            className="flex items-center space-x-1.5 bg-rose-500/30 hover:bg-rose-500/40 text-white px-4 py-2 rounded-full text-xs font-bold border border-rose-400/40 shrink-0 transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset Line</span>
          </button>
        </div>
      )}

      {/* Stream Viewport & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-3">
          <div
            ref={viewportContainerRef}
            className="relative aspect-video rounded-2xl overflow-hidden border border-violet-500/20 bg-[#090614]/85 backdrop-blur-3xl flex items-center justify-center select-none shadow-[0_16px_40px_rgba(0,0,0,0.7)]"
          >
            {/* Hidden capture scratchpad canvas */}
            <canvas ref={captureCanvasRef} className="hidden" />

            {/* Video stream from physical webcam */}
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              className={`w-full h-full object-contain ${useWebcam && isStreaming ? 'block' : 'hidden'}`}
            />

            {/* Simulated industrial conveyor canvas */}
            <canvas
              ref={conveyorCanvasRef}
              className={`w-full h-full object-contain ${!useWebcam && isStreaming ? 'block' : 'hidden'}`}
            />

            {/* Hardware-accelerated client canvas overlay for real-time bounding boxes */}
            <canvas
              ref={overlayCanvasRef}
              className={`absolute inset-0 w-full h-full pointer-events-none z-10 ${
                isStreaming ? 'block' : 'hidden'
              }`}
            />

            {/* High-Tech Specimen Target Reticle (Only in Webcam ROI Mode) */}
            {useWebcam && isStreaming && useTargetRoi && (
              <div className="absolute inset-0 pointer-events-none z-15 flex items-center justify-center p-6">
                <div className="w-[60%] h-[70%] border border-dashed border-sky-400/50 rounded-2xl relative bg-sky-950/20 backdrop-blur-sm shadow-[0_0_25px_rgba(56,189,248,0.15)] flex flex-col justify-between p-3">
                  {/* Corner Target Brackets */}
                  <div className="absolute -top-1 -left-1 w-5 h-5 border-t-2 border-l-2 border-sky-400" />
                  <div className="absolute -top-1 -right-1 w-5 h-5 border-t-2 border-r-2 border-sky-400" />
                  <div className="absolute -bottom-1 -left-1 w-5 h-5 border-b-2 border-l-2 border-sky-400" />
                  <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-2 border-r-2 border-sky-400" />

                  <div className="flex items-center justify-center space-x-1.5 bg-sky-950/80 backdrop-blur-md border border-sky-400/30 text-sky-200 text-[10px] font-mono font-bold px-3 py-1 rounded-full self-center shadow-lg">
                    <Target className="w-3.5 h-3.5 text-sky-400 animate-pulse" />
                    <span>TARGET INSPECTION ZONE (Place Specimen / Phone Here)</span>
                  </div>

                  <div className="text-[10px] text-sky-300/80 font-mono text-center self-center bg-slate-950/70 backdrop-blur-md px-3 py-1 rounded-full border border-sky-400/20">
                    High-Res Optical Focus Active (Room Background Ignored)
                  </div>
                </div>
              </div>
            )}

            {/* Idle state placeholder */}
            {!isStreaming && (
              <div className="text-center space-y-3 p-8">
                <div className="p-4 rounded-full bg-white/[0.04] border border-white/[0.08] inline-block text-slate-400 backdrop-blur-xl">
                  <Video className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Conveyor Stream Idle</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Click &quot;Start Live Stream&quot; above to activate real-time defect telemetry.
                  </p>
                </div>
              </div>
            )}

            {/* Real-Time Telemetry HUD Overlays */}
            {isStreaming && (
              <>
                <div className="absolute top-3 left-3 bg-slate-950/70 backdrop-blur-2xl px-3.5 py-2 rounded-xl border border-white/[0.08] text-xs font-mono space-y-1 z-20 pointer-events-none shadow-xl">
                  <div className="flex items-center space-x-2">
                    <span className="text-slate-400">FPS:</span>
                    <span className="text-sky-400 font-bold text-sm">{fps.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-slate-400">LATENCY:</span>
                    <span className="text-emerald-400 font-bold">{latency.toFixed(1)} ms</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-slate-400">DROPPED:</span>
                    <span className="text-amber-400 font-bold">{droppedFrames}</span>
                  </div>
                </div>

                <div className="absolute top-3 right-3 bg-slate-950/70 backdrop-blur-2xl px-3.5 py-1.5 rounded-full border border-white/[0.08] flex items-center space-x-2 z-20 pointer-events-none shadow-xl">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                  <span className="text-xs font-mono font-bold text-emerald-300 uppercase">
                    YOLO ONNX STREAM (15+ FPS)
                  </span>
                </div>

                <div className="absolute bottom-3 right-3 bg-slate-950/70 backdrop-blur-2xl px-3.5 py-1.5 rounded-full border border-white/[0.08] flex items-center space-x-2 z-20 pointer-events-none shadow-xl">
                  <span className="text-xs font-mono text-slate-400">VERDICT:</span>
                  <SeverityBadge level={verdict === 'rejected' ? 'critical' : verdict === 'flagged' ? 'major' : 'passed'} size="sm" />
                </div>
              </>
            )}
          </div>
        </div>

        {/* Real-Time Telemetry & Defect Log */}
        <div className="space-y-4">
          {/* Automated Conveyor Production Routing HUD */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-bold uppercase tracking-wider text-white">Production Routing</h4>
              <span className="text-[10px] font-mono bg-white/[0.04] px-2.5 py-1 rounded-full border border-white/[0.08] text-sky-400">
                {productionRouting.dispatch_latency_ms} ms dispatch
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-white/[0.03] backdrop-blur-xl border border-white/[0.06] space-y-2.5 font-mono">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Action & Signal</span>
                <span
                  className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${
                    productionRouting.action === 'REJECT'
                      ? 'bg-rose-500/20 text-rose-300 border-rose-400/40'
                      : productionRouting.action === 'REWORK'
                      ? 'bg-amber-500/20 text-amber-300 border-amber-400/40'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-400/40'
                  }`}
                >
                  {productionRouting.action}: {productionRouting.signal}
                </span>
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Target Station</span>
                <span className="text-slate-200 font-bold">{productionRouting.target_station}</span>
              </div>

              <p className="text-[11px] text-slate-400 pt-2 border-t border-white/[0.06] leading-snug">
                {productionRouting.reason}
              </p>
            </div>
          </div>

          {/* Telemetry Stats Card */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <h4 className="text-sm font-bold uppercase tracking-wider text-white">Specimen Defect Telemetry</h4>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white/[0.03] backdrop-blur-xl p-3.5 rounded-xl border border-white/[0.06]">
                <span className="text-[10px] text-slate-400 uppercase font-mono font-medium">Detected Flaws</span>
                <p className="text-xl font-bold font-mono text-white mt-0.5">{totalDefects}</p>
              </div>

              <div className="bg-white/[0.03] backdrop-blur-xl p-3.5 rounded-xl border border-white/[0.06]">
                <span className="text-[10px] text-slate-400 uppercase font-mono font-medium">Peak Severity</span>
                <div className="mt-1">
                  <SeverityBadge level={maxSeverity} size="sm" />
                </div>
              </div>
            </div>

            {/* Active Defect Instances Strip */}
            <div className="space-y-2 pt-1">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Live Frame Defect Log
              </span>
              <div className="max-h-48 overflow-y-auto space-y-2 pr-1">
                {activeDefects.length === 0 ? (
                  <p className="text-xs text-slate-400 italic py-2">No active defects on continuous feed.</p>
                ) : (
                  activeDefects.map((def, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between bg-white/[0.03] backdrop-blur-md p-2.5 rounded-xl border border-white/[0.06] text-xs"
                    >
                      <div className="flex items-center space-x-2">
                        <AlertTriangle
                          className={`w-3.5 h-3.5 ${
                            def.severity === 'critical'
                              ? 'text-rose-400'
                              : def.severity === 'major'
                              ? 'text-amber-400'
                              : 'text-sky-400'
                          }`}
                        />
                        <span className="font-medium text-slate-200 capitalize">{def.class_name}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-slate-400 text-[11px]">
                          {(def.confidence * 100).toFixed(0)}%
                        </span>
                        <SeverityBadge level={def.severity} size="sm" />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
