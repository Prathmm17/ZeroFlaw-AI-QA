"""Alerts API router for active factory notifications and operator acknowledgments."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import AlertRecord, get_db
from backend.models import AcknowledgeAlertRequest, AlertResponse
from backend.services.alert_service import AlertService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alerts", tags=["Quality Alerts"])


@router.get("", response_model=List[AlertResponse])
def get_alerts(
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment state"),
    severity: Optional[str] = Query(None, description="Filter by severity ('critical' or 'major')"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> List[AlertResponse]:
    """Retrieve quality alerts with optional filtering by status or severity."""
    alerts = AlertService.get_alerts(
        db=db,
        acknowledged=acknowledged,
        severity=severity,
        limit=limit,
        offset=offset,
    )
    return [AlertResponse.model_validate(a) for a in alerts]


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(
    alert_id: int,
    payload: AcknowledgeAlertRequest,
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Acknowledge an active alert by operator name."""
    alert = AlertService.acknowledge_alert(
        db=db,
        alert_id=alert_id,
        acknowledged_by=payload.acknowledged_by,
    )
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert #{alert_id} not found.",
        )
    return AlertResponse.model_validate(alert)

