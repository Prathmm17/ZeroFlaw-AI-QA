import axios from 'axios';
import {
  AnalyticsOverview,
  AnalyticsTrends,
  DefectKnowledgeItem,
  Inspection,
  InspectionListResponse,
  QualityAlert,
  SystemHealth,
} from '../types';

const apiBase = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');
const baseURL = apiBase ? (apiBase.endsWith('/api') ? apiBase : `${apiBase}/api`) : '/api';

export function resolveMediaUrl(path: string): string {
  if (!path) return '';
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const hostUrl = apiBase ? apiBase.replace(/\/api\/?$/, '') : '';
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return hostUrl ? `${hostUrl}${cleanPath}` : cleanPath;
}

const api = axios.create({
  baseURL,
  timeout: 30000,
});

export const apiClient = {
  // System Health
  async getHealth(): Promise<SystemHealth> {
    const res = await api.get<SystemHealth>('/health');
    return res.data;
  },

  // Detection & Inspections
  async uploadImage(file: File, notes?: string, deepVerify = false): Promise<Inspection> {
    const formData = new FormData();
    formData.append('file', file);
    if (notes) {
      formData.append('notes', notes);
    }
    const res = await api.post<Inspection>('/detect/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: { deep_verify: deepVerify },
    });
    return res.data;
  },

  async getSamples(): Promise<string[]> {
    const res = await api.get<string[]>('/detect/samples');
    return res.data;
  },

  async inspectSample(sampleName: string, deepVerify = false): Promise<Inspection> {
    const res = await api.post<Inspection>(
      `/detect/sample/${encodeURIComponent(sampleName)}`,
      null,
      { params: { deep_verify: deepVerify } }
    );
    return res.data;
  },

  async getHistory(
    page = 1,
    pageSize = 15,
    statusFilter?: string,
    severityFilter?: string
  ): Promise<InspectionListResponse> {
    const params: Record<string, string | number> = { page, page_size: pageSize };
    if (statusFilter && statusFilter !== 'all') params.status = statusFilter;
    if (severityFilter && severityFilter !== 'all') params.severity = severityFilter;
    const res = await api.get<InspectionListResponse>('/detect/history', { params });
    return res.data;
  },

  async getInspection(id: number): Promise<Inspection> {
    const res = await api.get<Inspection>(`/detect/${id}`);
    return res.data;
  },

  async getGlossary(): Promise<Record<string, DefectKnowledgeItem>> {
    const res = await api.get<Record<string, DefectKnowledgeItem>>('/detect/glossary');
    return res.data;
  },

  // Optical Scale Calibration
  async getCalibration(): Promise<{ px_per_mm: number | null; calibrated: boolean; calibration_label: string; message: string }> {
    const res = await api.get('/detect/calibration');
    return res.data;
  },

  async setCalibration(params: {
    px_per_mm?: number | null;
    line_length_px?: number;
    known_length_mm?: number;
  }): Promise<{ px_per_mm: number | null; calibrated: boolean; calibration_label: string; message: string }> {
    const res = await api.post('/detect/calibration', params);
    return res.data;
  },

  async recalculateMetrology(inspectionId: number): Promise<Inspection> {
    const res = await api.post<Inspection>(`/detect/${inspectionId}/recalculate-metrology`);
    return res.data;
  },

  get8DReportPdfUrl(inspectionId: number): string {
    return resolveMediaUrl(`/api/detect/${inspectionId}/8d-report/pdf`);
  },

  // Analytics
  async getAnalyticsOverview(): Promise<AnalyticsOverview> {
    const res = await api.get<AnalyticsOverview>('/analytics/overview');
    return res.data;
  },

  async getAnalyticsTrends(days = 7): Promise<AnalyticsTrends> {
    const res = await api.get<AnalyticsTrends>('/analytics/trends', { params: { days } });
    return res.data;
  },

  // Alerts
  async getAlerts(acknowledged?: boolean, severity?: string): Promise<QualityAlert[]> {
    const params: Record<string, string | boolean> = {};
    if (acknowledged !== undefined) params.acknowledged = acknowledged;
    if (severity) params.severity = severity;
    const res = await api.get<QualityAlert[]>('/alerts', { params });
    return res.data;
  },

  async acknowledgeAlert(alertId: number, acknowledgedBy = 'Operator'): Promise<QualityAlert> {
    const res = await api.post<QualityAlert>(`/alerts/${alertId}/acknowledge`, {
      acknowledged_by: acknowledgedBy,
    });
    return res.data;
  },

  // Video Inspection
  async uploadVideo(
    file: File,
    sampleFps = 3.0,
    confidenceThreshold = 0.25,
    ambientBoost = true
  ): Promise<{ video_id: string; filename: string; status: string; progress: number; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post('/video/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: {
        sample_fps: sampleFps,
        confidence_threshold: confidenceThreshold,
        ambient_boost: ambientBoost,
      },
    });
    return res.data;
  },

  async getVideoJobs(): Promise<any[]> {
    const res = await api.get('/video/jobs');
    return res.data;
  },

  async getVideoStatus(videoId: string): Promise<any> {
    const res = await api.get(`/video/${videoId}/status`);
    return res.data;
  },

  async getVideoResult(videoId: string): Promise<any> {
    const res = await api.get(`/video/${videoId}/result`);
    return res.data;
  },
};

