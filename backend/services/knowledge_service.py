"""Defect knowledge base service for VisionGuard QA.

Provides access to engineering defect definitions, typical causes, visual looks,
failure risks, recommended quality actions, and calibrated threshold benchmarks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_KNOWLEDGE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def get_all_defect_knowledge() -> Dict[str, Dict[str, Any]]:
    """Load and return the complete defect knowledge dictionary."""
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is None:
        knowledge_file = settings.DEFECT_KNOWLEDGE_PATH
        if not knowledge_file.is_file():
            # Fallback search
            knowledge_file = Path(__file__).resolve().parent.parent / "defect_knowledge.json"

        if knowledge_file.is_file():
            try:
                with open(knowledge_file, "r", encoding="utf-8") as f:
                    _KNOWLEDGE_CACHE = json.load(f)
            except Exception as e:
                logger.error("Failed to load defect knowledge from %s: %s", knowledge_file, e)
                _KNOWLEDGE_CACHE = {}
        else:
            logger.warning("Defect knowledge file not found at %s", knowledge_file)
            _KNOWLEDGE_CACHE = {}
    return _KNOWLEDGE_CACHE


def get_defect_knowledge(class_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve defect knowledge item by class name."""
    knowledge = get_all_defect_knowledge()
    key = class_name.lower().strip()
    return knowledge.get(key)


def enrich_defect_dict(det: Dict[str, Any]) -> Dict[str, Any]:
    """Attach definition, cause, action, visual_look, and risk to a defect dictionary."""
    class_name = det.get("class_name", "")
    info = get_defect_knowledge(class_name)
    if info:
        det["definition"] = info.get("definition", "")
        det["cause"] = info.get("cause", "")
        det["action"] = info.get("action", "")
        det["visual_look"] = info.get("visual_look", "")
        det["risk"] = info.get("risk", "")
    else:
        det.setdefault("definition", None)
        det.setdefault("cause", None)
        det.setdefault("action", None)
        det.setdefault("visual_look", None)
        det.setdefault("risk", None)
    return det

