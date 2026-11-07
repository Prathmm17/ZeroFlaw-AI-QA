# VisionGuard QA — API Reference

Base URL: `http://localhost:8000`  
Interactive Swagger Docs: `http://localhost:8000/docs`  
ReDoc: `http://localhost:8000/redoc`  

---

## 1. System Health

### `GET /api/health`
Returns the status of the server and ML detector.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "app": "VisionGuard QA",
  "version": "1.0.0",
  "model_ready": "True",
  "model_name": "best.pt",
  "device": "cpu"
}
```

---

## 2. Defect Detection & Inspections

### `GET /api/detect/samples`
Returns a list of synthetic sample images available for instant demo testing.

#### Response `200 OK`
```json
[
  "sample_chip_004.png",
  "sample_crack_000.png",
  "sample_dent_002.png",
  "sample_scratch_001.png",
  "sample_stain_003.png"
]
```

---

### `POST /api/detect/sample/{sample_name}`
Executes defect detection on a pre-loaded sample.

#### Response `200 OK`
```json
{
  "id": 2,
  "timestamp": "2026-10-29T01:12:19.860000",
  "source": "sample",
  "filename": "sample_chip_004.png",
  "original_url": "/data/uploads/sample_chip_004_20261029_011219_abc123_orig.jpg",
  "result_url": "/data/results/sample_chip_004_20261029_011219_abc123_result.jpg",
  "total_defects": 3,
  "critical_count": 0,
  "major_count": 0,
  "minor_count": 3,
  "max_severity": "minor",
  "status": "flagged",
  "inference_time_ms": 1681.8,
  "total_time_ms": 1725.7,
  "notes": "Processed from demo sample catalog",
  "defects": [
    {
      "bbox": [120.4, 250.1, 190.8, 310.2],
      "confidence": 0.529,
      "class_id": 4,
      "class_name": "chip",
      "severity": "minor",
      "area_ratio": 0.012,
      "description": "Minor chip detected with 53% confidence"
    }
  ]
}
```

---

### `POST /api/detect/upload`
Upload an industrial image file via multipart form.

#### Form Data
- `file`: Image binary (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`)
- `notes` *(optional)*: Text string describing the inspection batch.

#### Response `201 Created`
Returns the complete `InspectionResponse` schema.

---

### `GET /api/detect/history`
Returns paginated inspection history.

#### Query Parameters
- `page` *(int, default: 1)*
- `page_size` *(int, default: 20)*
- `status` *(string, optional)*: `passed`, `flagged`, `rejected`
- `severity` *(string, optional)*: `critical`, `major`, `minor`

#### Response `200 OK`
```json
{
  "items": [...],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "pages": 3
}
```

---

## 3. Executive Quality Analytics

### `GET /api/analytics/overview`
Aggregated quality metrics across all inspections.

#### Response `200 OK`
```json
{
  "total_inspections": 42,
  "total_defects": 15,
  "defect_free_inspections": 30,
  "defect_rate_percentage": 28.6,
  "critical_defects_count": 2,
  "major_defects_count": 5,
  "minor_defects_count": 8,
  "unacknowledged_alerts": 1,
  "avg_inference_time_ms": 1650.4,
  "defect_types_distribution": {
    "crack": 4,
    "scratch": 5,
    "chip": 3,
    "dent": 2,
    "stain": 1
  },
  "severity_distribution": {
    "critical": 2,
    "major": 5,
    "minor": 8
  },
  "status_distribution": {
    "passed": 30,
    "flagged": 10,
    "rejected": 2
  }
}
```

---

### `GET /api/analytics/trends`
Historical volume and defect timeline for the past N days.

#### Query Parameters
- `days` *(int, default: 7)*

---

## 4. Quality Alerts & Routing

### `GET /api/alerts`
Returns quality alerts with optional filtering.

#### Query Parameters
- `acknowledged` *(bool, optional)*: `true` or `false`
- `severity` *(string, optional)*: `critical` or `major`

---

### `POST /api/alerts/{alert_id}/acknowledge`
Acknowledge an active alert by operator name.

#### Request Body
```json
{
  "acknowledged_by": "Shift Supervisor",
  "notes": "Component quarantined for destruction"
}
```

---

## 5. WebSockets

### `WS /ws/stream`
Real-time conveyor video stream inference.
- **Client Sends**: `{"frame": "<base64_data>", "return_annotated": true}`
- **Server Sends**:
```json
{
  "timestamp": "2026-10-29T01:15:00.000Z",
  "inference_time_ms": 12.5,
  "fps": 28.4,
  "total_defects": 1,
  "max_severity": "critical",
  "has_critical": true,
  "defects": [...],
  "frame_annotated": "data:image/jpeg;base64,..."
}
```

### `WS /ws/alerts`
Persistent WebSocket connection for real-time alert push notifications.

