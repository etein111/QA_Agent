# 从 MinerU *_content_list.json 加载为 MinerUElement 列表（复用 ai-narrator-agent-yc 逻辑）
from __future__ import annotations

import json
from typing import Any

from .data_types import MinerUElement

ALLOWED_TYPES = {"text", "header", "footer", "image", "table", "equation", "list", "page_number"}


def _assert_bbox_0_1000(bbox: list[Any]) -> tuple[float, float, float, float]:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox format: {bbox}")
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if not (0.0 <= x0 <= 1000.0 and 0.0 <= y0 <= 1000.0 and 0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0):
        raise ValueError(f"MinerU bbox out of [0,1000]: {bbox}")
    if x1 < x0 or y1 < y0:
        raise ValueError(f"Invalid bbox corners (x1<x0 or y1<y0): {bbox}")
    return x0, y0, x1, y1


def load_mineru_pre_content_list(json_path: str) -> list[MinerUElement]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected pre_content_list.json to be a JSON array.")

    elements: list[MinerUElement] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t not in ALLOWED_TYPES:
            continue
        page_idx = int(item.get("page_idx", 0))
        bbox = _assert_bbox_0_1000(item.get("bbox"))
        text = item.get("text")
        if isinstance(text, str):
            text = text.strip()
        else:
            text = None

        elements.append(
            MinerUElement(
                type=t,
                text=text,
                bbox=bbox,
                page_idx=page_idx,
                raw=item,
            )
        )

    elements.sort(key=lambda e: (e.page_idx, e.bbox[1], e.bbox[0]))
    return elements
