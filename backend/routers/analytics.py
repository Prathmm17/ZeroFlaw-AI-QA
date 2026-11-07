"""Analytics API router providing defect distributions, trends, and quality yield metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import AnalyticsOverviewResponse, AnalyticsTrendsResponse
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/api/analytics", tags=["Quality Analytics"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def get_analytics_overview(db: Session = Depends(get_db)) -> AnalyticsOverviewResponse:
    """Retrieve executive overview metrics across all inspections and defects."""
    return AnalyticsService.get_overview(db)


@router.get("/trends", response_model=AnalyticsTrendsResponse)
def get_analytics_trends(
    days: int = Query(7, ge=1, le=90, description="Time window in days"),
    db: Session = Depends(get_db),
) -> AnalyticsTrendsResponse:
    """Retrieve historical quality and defect volume trend timeline."""
    return AnalyticsService.get_trends(db=db, days=days)

