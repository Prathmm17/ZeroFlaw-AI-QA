export type SeverityLevel = 'critical' | 'major' | 'minor' | 'none';
export type InspectionStatus = 'passed' | 'flagged' | 'rejected';

export interface DefectMetrology {
  pixel_area: number;
  area_px?: number;
  surface_area_mm2?: number | null;
  length_mm?: number | null;
  width_mm?: number | null;
  length_px?: number;
  width_px?: number;
  aspect_ratio: number;
  solidity: number;
  polygon: [number, number][];
  is_calibrated?: boolean;
  calibration_label?: string;
  px_per_mm?: number | null;
}

export interface CalibrationInfo {
  px_per_mm: number | null;
  calibrated: boolean;
  calibration_label: string;
  message?: string;
}

export interface Defect {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  class_id: number;
  class_name: string;
  severity: SeverityLevel;
  area_ratio: number;
  description: string;
  polygon?: [number, number][];
  metrology?: DefectMetrology;
  tier?: 'tier1_edge' | 'tier2_vlm';
  vlm_verified?: boolean;
  failure_mechanism?: string;
  source?: 'yolo' | 'anomaly_cv' | 'vlm';
  definition?: string;
  cause?: string;
  action?: string;
  visual_look?: string;
  risk?: string;
}

export interface DefectKnowledgeItem {
  name: string;
  display_name: string;
  definition: string;
  visual_look: string;
  cause: string;
  typical_cause?: string;
  risk: string;
  action: string;
  recommended_action?: string;
  default_severity: SeverityLevel;
  calibrated_threshold?: number;
  confidence_label?: string;
}

export interface ConformanceCheckItem {
  criterion_id: string;
  name: string;
  description: string;
  passed: boolean;
  status_label: string;
  severity_level: 'pass' | 'warning' | 'violation';
  recommendation: string;
}

export interface ConformanceReport {
  lot_release_authorized: boolean;
  surface_integrity_pass: boolean;
  critical_violation_free: boolean;
  area_tolerance_pass: boolean;
  dimensional_tolerance_pass: boolean;
  material_anomaly_clear: boolean;
  total_criteria: number;
  passed_criteria: number;
  failed_criteria: number;
  items: ConformanceCheckItem[];
}

export interface Inspection {
  id: number;
  timestamp: string;
  source: string;
  filename: string;
  original_url: string;
  result_url: string;
  total_defects: number;
  critical_count: number;
  major_count: number;
  minor_count: number;
  max_severity: SeverityLevel;
  status: InspectionStatus;
  inference_time_ms: number;
  total_time_ms: number;
  notes?: string;
  audit_report?: EightDReport;
  conformance_report?: ConformanceReport;
  arbitration_tier?: 'tier1_edge' | 'tier2_vlm' | 'dual_fused';
  arbitration_status?: 'CLEAN_PASS' | 'EDGE_CONFIRMED' | 'VLM_VERIFIED' | 'VLM_CONFIRMED' | 'ANOMALY_ESCALATED';
  domain_trust?: 'HIGH' | 'LOW';
  domain_warning?: string | null;
  px_per_mm?: number | null;
  is_calibrated?: boolean;
  calibration_label?: string;
  defects: Defect[];
}

export interface EightDReport {
  inspection_id: number;
  timestamp: string;
  primary_defect_type: string;
  defect_mechanism: string;
  probable_root_cause: string;
  tooling_station: string;
  standard_reference: string;
  containment_action: string;
  corrective_action: string;
  engineering_severity: string;
  metrology_summary: Record<string, any>;
  markdown_report: string;
  source?: 'yolo' | 'anomaly_cv' | 'vlm';
  d1_team?: string;
  d2_problem_description?: string;
  d3_containment_action?: string;
  d4_root_cause_analysis?: string;
  d5_corrective_actions?: string;
  d6_implementation_verification?: string;
  d7_preventive_action?: string;
  d8_team_recognition?: string;
}

export interface InspectionListResponse {
  items: Inspection[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface QualityAlert {
  id: number;
  inspection_id: number;
  timestamp: string;
  severity: 'critical' | 'major';
  defect_type: string;
  message: string;
  action_required: string;
  acknowledged: boolean;
  acknowledged_at?: string;
  acknowledged_by?: string;
}

export interface AnalyticsOverview {
  total_inspections: number;
  total_defects: number;
  defect_free_inspections: number;
  defect_rate_percentage: number;
  critical_defects_count: number;
  major_defects_count: number;
  minor_defects_count: number;
  unacknowledged_alerts: number;
  avg_inference_time_ms: number;
  defect_types_distribution: Record<string, number>;
  severity_distribution: Record<string, number>;
  status_distribution: Record<string, number>;
  cp?: number;
  cpk?: number;
  pareto_cumulative?: Record<string, number>;
  process_capability_status?: string;
}

export interface TrendPoint {
  date: string;
  inspections: number;
  defects: number;
  critical: number;
  major: number;
  minor: number;
}

export interface AnalyticsTrends {
  trends: TrendPoint[];
}

export interface SystemHealth {
  status: string;
  app: string;
  version: string;
  model_ready: string;
  model_name: string;
  device: string;
}

export interface VideoTimelineItem {
  timestamp_sec: number;
  frame_idx: number;
  defect_count: number;
  max_severity: SeverityLevel;
  defects: Defect[];
  thumbnail_url: string;
}

export interface VideoJobSummary {
  by_class: Record<string, number>;
  by_severity: Record<string, number>;
  peak_timestamp_sec: number;
}

export interface VideoJob {
  video_id: string;
  filename: string;
  video_path: string;
  video_url: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_stage: string;
  duration_seconds: number;
  fps: number;
  total_frames: number;
  sampled_frames: number;
  total_defects_found: number;
  max_severity: SeverityLevel;
  timeline: VideoTimelineItem[];
  summary: VideoJobSummary;
  error?: string | null;
  created_at?: number;
}

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  status: string;
  progress: number;
  message: string;
}

