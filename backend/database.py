"""Database connection and SQLAlchemy ORM models for VisionGuard QA.

Stores inspections, detected defects, and actionable quality alerts.
Uses SQLite with SQLAlchemy 2.0+ declarative mapping.
"""

from __future__ import annotations

import datetime
from typing import Generator, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

from backend.config import settings


# SQLAlchemy 2.0 Base
class Base(DeclarativeBase):
    pass


class Inspection(Base):
    """Represents a single inspection event (from upload, sample, or camera frame)."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, index=True
    )
    source: Mapped[str] = mapped_column(String(50), default="upload")  # upload, sample, camera
    filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str] = mapped_column(String(500))
    result_path: Mapped[str] = mapped_column(String(500))
    
    total_defects: Mapped[int] = mapped_column(Integer, default=0)
    critical_count: Mapped[int] = mapped_column(Integer, default=0)
    major_count: Mapped[int] = mapped_column(Integer, default=0)
    minor_count: Mapped[int] = mapped_column(Integer, default=0)
    
    max_severity: Mapped[str] = mapped_column(String(20), default="none")  # none, minor, major, critical
    status: Mapped[str] = mapped_column(String(20), default="passed")  # passed, flagged, rejected
    
    inference_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audit_report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    arbitration_tier: Mapped[str] = mapped_column(String(50), default="tier1_edge")
    arbitration_status: Mapped[str] = mapped_column(String(50), default="EDGE_CONFIRMED")
    domain_trust: Mapped[str] = mapped_column(String(20), default="HIGH")
    domain_warning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    defects: Mapped[List[DefectRecord]] = relationship(
        "DefectRecord", back_populates="inspection", cascade="all, delete-orphan"
    )
    alerts: Mapped[List[AlertRecord]] = relationship(
        "AlertRecord", back_populates="inspection", cascade="all, delete-orphan"
    )


class DefectRecord(Base):
    """Represents an individual defect detected during an inspection."""

    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20), index=True)  # minor, major, critical
    
    bbox_x1: Mapped[float] = mapped_column(Float)
    bbox_y1: Mapped[float] = mapped_column(Float)
    bbox_x2: Mapped[float] = mapped_column(Float)
    bbox_y2: Mapped[float] = mapped_column(Float)
    area_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    description: Mapped[str] = mapped_column(String(255))
    polygon_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrology_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tier: Mapped[str] = mapped_column(String(50), default="tier1_edge")
    vlm_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_mechanism: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="yolo")  # "yolo" | "anomaly_cv" | "vlm"

    inspection: Mapped[Inspection] = relationship("Inspection", back_populates="defects")


class AlertRecord(Base):
    """Represents an actionable alert generated from high-severity defects."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("inspections.id", ondelete="CASCADE"), index=True
    )
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, index=True
    )
    severity: Mapped[str] = mapped_column(String(20), index=True)  # major, critical
    defect_type: Mapped[str] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(String(500))
    action_required: Mapped[str] = mapped_column(String(255))
    
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    inspection: Mapped[Inspection] = relationship("Inspection", back_populates="alerts")


# Database engine and session
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables and perform automatic column migrations."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            # Inspection table migration
            insp_res = conn.execute(text("PRAGMA table_info(inspections)")).fetchall()
            insp_cols = {row[1] for row in insp_res}
            if "audit_report_json" not in insp_cols:
                conn.execute(text("ALTER TABLE inspections ADD COLUMN audit_report_json TEXT"))
            if "arbitration_tier" not in insp_cols:
                conn.execute(text("ALTER TABLE inspections ADD COLUMN arbitration_tier VARCHAR(50) DEFAULT 'tier1_edge'"))
            if "arbitration_status" not in insp_cols:
                conn.execute(text("ALTER TABLE inspections ADD COLUMN arbitration_status VARCHAR(50) DEFAULT 'EDGE_CONFIRMED'"))
            if "domain_trust" not in insp_cols:
                conn.execute(text("ALTER TABLE inspections ADD COLUMN domain_trust VARCHAR(20) DEFAULT 'HIGH'"))
            if "domain_warning" not in insp_cols:
                conn.execute(text("ALTER TABLE inspections ADD COLUMN domain_warning TEXT"))

            # Defects table migration
            res = conn.execute(text("PRAGMA table_info(defects)")).fetchall()
            existing_cols = {row[1] for row in res}
            if "polygon_json" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN polygon_json TEXT"))
            if "metrology_json" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN metrology_json TEXT"))
            if "tier" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN tier VARCHAR(50) DEFAULT 'tier1_edge'"))
            if "vlm_verified" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN vlm_verified BOOLEAN DEFAULT 0"))
            if "failure_mechanism" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN failure_mechanism VARCHAR(255)"))
            if "source" not in existing_cols:
                conn.execute(text("ALTER TABLE defects ADD COLUMN source VARCHAR(50) DEFAULT 'yolo'"))
            conn.commit()
        except Exception:
            pass

    # Automatically seed demonstration baseline if database is empty (e.g. after container restart)
    try:
        with SessionLocal() as session:
            if session.query(Inspection).count() == 0:
                sample_files = sorted(list(settings.SAMPLES_DIR.glob("*.png")) + list(settings.SAMPLES_DIR.glob("*.jpg")))
                if sample_files:
                    f1 = sample_files[0]
                    f2 = sample_files[min(1, len(sample_files) - 1)]
                    insp1 = Inspection(
                        source="sample",
                        filename=f1.name,
                        original_path=str(f1),
                        result_path=str(f1),
                        total_defects=1,
                        critical_count=0,
                        major_count=1,
                        minor_count=0,
                        max_severity="major",
                        status="flagged",
                        inference_time_ms=85.0,
                        total_time_ms=110.0,
                        notes="Automated baseline specimen inspection",
                        domain_trust="HIGH",
                    )
                    session.add(insp1)
                    session.flush()

                    session.add(
                        DefectRecord(
                            inspection_id=insp1.id,
                            class_id=1,
                            class_name="crazing",
                            confidence=0.88,
                            severity="major",
                            bbox_x1=100.0,
                            bbox_y1=120.0,
                            bbox_x2=320.0,
                            bbox_y2=380.0,
                            area_ratio=0.035,
                            description="Major crazing defect detected on specimen surface",
                            source="yolo",
                        )
                    )

                    insp2 = Inspection(
                        source="sample",
                        filename=f2.name,
                        original_path=str(f2),
                        result_path=str(f2),
                        total_defects=2,
                        critical_count=0,
                        major_count=0,
                        minor_count=2,
                        max_severity="minor",
                        status="passed",
                        inference_time_ms=92.0,
                        total_time_ms=120.0,
                        notes="Factory baseline specimen inspection",
                        domain_trust="HIGH",
                    )
                    session.add(insp2)
                    session.flush()

                    session.add(
                        DefectRecord(
                            inspection_id=insp2.id,
                            class_id=0,
                            class_name="crazing",
                            confidence=0.45,
                            severity="minor",
                            bbox_x1=210.0,
                            bbox_y1=150.0,
                            bbox_x2=350.0,
                            bbox_y2=240.0,
                            area_ratio=0.012,
                            description="Minor surface crazing fissure",
                            source="yolo",
                        )
                    )
                    session.commit()
    except Exception:
        pass


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides an active database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

