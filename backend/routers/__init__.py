"""Backend routers package."""

from backend.routers.alerts import router as alerts_router
from backend.routers.analytics import router as analytics_router
from backend.routers.detection import router as detection_router
from backend.routers.stream import router as stream_router
from backend.routers.video import router as video_router

__all__ = [
    "alerts_router",
    "analytics_router",
    "detection_router",
    "stream_router",
    "video_router",
]

