"""Alert routing and WebSocket notification service for VisionGuard QA.

Classifies defects into actionable manufacturing alerts and broadcasts
them in real time to connected operator dashboards.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from sqlalchemy.orm import Session

from backend.database import AlertRecord, Inspection
from backend.models import AlertResponse

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for live notifications and streaming."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. Total active clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket disconnected. Total active clients: %d", len(self.active_connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a JSON message to all active WebSocket clients."""
        if not self.active_connections:
            return

        payload = json.dumps(message)
        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception as exc:
                logger.warning("Failed to send message to client: %s", exc)
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)


# Global connection manager instance
ws_manager = ConnectionManager()


class AlertService:
    """Handles alert creation, routing logic, and alert state management."""

    @staticmethod
    def process_inspection_alerts(
        db: Session,
        inspection: Inspection,
        detections: List[Dict[str, Any]],
    ) -> List[AlertRecord]:
        """Evaluate detected defects and create database alert records.

        Routing Rules:
        - Critical: Halts or flags for quarantine.
        - Major: Routes to secondary QA inspection.
        - Minor: Logged for statistical quality control (no alert created).
        """
        created_alerts: List[AlertRecord] = []

        for defect in detections:
            severity = defect.get("severity", "minor").lower()
            defect_type = defect.get("class_name", "defect")
            confidence = defect.get("confidence", 0.0)
            source = defect.get("source", "yolo")
            is_low_trust = getattr(inspection, "domain_trust", "HIGH") == "LOW"

            # Routing rule: Detections with trust LOW and source anomaly_cv are at most MAJOR
            if is_low_trust and source == "anomaly_cv" and severity == "critical":
                severity = "major"

            if severity == "critical":
                alert = AlertRecord(
                    inspection_id=inspection.id,
                    timestamp=datetime.datetime.utcnow(),
                    severity="critical",
                    defect_type=defect_type,
                    message=(
                        f"CRITICAL DEFECT DETECTED: {defect_type.upper()} "
                        f"(confidence: {confidence * 100:.1f}%). "
                        "Immediate line halt recommended."
                    ),
                    action_required="Emergency halt: Quarantine item and notify shift supervisor.",
                    acknowledged=False,
                )
                db.add(alert)
                created_alerts.append(alert)

            elif severity == "major":
                action_text = (
                    "Flagged for human review: Visual domain differs from flat steel training baseline."
                    if is_low_trust
                    else "Route component to offline QA station for manual inspection."
                )
                alert = AlertRecord(
                    inspection_id=inspection.id,
                    timestamp=datetime.datetime.utcnow(),
                    severity="major",
                    defect_type=defect_type,
                    message=(
                        f"MAJOR DEFECT DETECTED: {defect_type.capitalize()} "
                        f"(confidence: {confidence * 100:.1f}%)."
                        + (" (Flagged for human review: Out-of-Domain)" if is_low_trust else "")
                    ),
                    action_required=action_text,
                    acknowledged=False,
                )
                db.add(alert)
                created_alerts.append(alert)

        if created_alerts:
            db.commit()
            for alert in created_alerts:
                db.refresh(alert)
            logger.info("Created %d alerts for inspection #%d", len(created_alerts), inspection.id)

        return created_alerts

    @staticmethod
    async def broadcast_alerts(alerts: List[AlertRecord]) -> None:
        """Broadcast newly created alerts over WebSockets."""
        for alert in alerts:
            payload = {
                "type": "new_alert",
                "alert": {
                    "id": alert.id,
                    "inspection_id": alert.inspection_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "defect_type": alert.defect_type,
                    "message": alert.message,
                    "action_required": alert.action_required,
                    "acknowledged": alert.acknowledged,
                },
            }
            await ws_manager.broadcast(payload)

    @staticmethod
    def acknowledge_alert(
        db: Session,
        alert_id: int,
        acknowledged_by: str = "Operator",
    ) -> Optional[AlertRecord]:
        """Mark an alert as acknowledged by an operator."""
        alert = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
        if not alert:
            return None

        alert.acknowledged = True
        alert.acknowledged_at = datetime.datetime.utcnow()
        alert.acknowledged_by = acknowledged_by
        db.commit()
        db.refresh(alert)
        logger.info("Alert #%d acknowledged by %s", alert_id, acknowledged_by)
        return alert

    @staticmethod
    def get_alerts(
        db: Session,
        acknowledged: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AlertRecord]:
        """Query alert history with optional filtering."""
        query = db.query(AlertRecord)
        if acknowledged is not None:
            query = query.filter(AlertRecord.acknowledged == acknowledged)
        if severity is not None:
            query = query.filter(AlertRecord.severity == severity.lower())

        return (
            query.order_by(AlertRecord.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

