# 与 ai-narrator-agent-yc 中 MinerU Segment 逻辑对齐的数据类型
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ElementType = Literal["text", "header", "footer", "image", "table", "equation", "list", "page_number"]


@dataclass(frozen=True)
class MinerUElement:
    type: ElementType
    text: str | None
    bbox: tuple[float, float, float, float]  # x0,y0,x1,y1 in [0,1000]
    page_idx: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0


@dataclass
class Segment:
    """MinerU 单页内一个讲解块，与参考项目 Segment 一致。"""
    segment_id: str
    page_idx: int
    title: str
    elements: list[MinerUElement]
    bbox_union: BBox
    anchor_points: list[tuple[float, float]]
    tags: set[str] = field(default_factory=set)


@dataclass
class SegmentMetadata:
    """
    供检索使用的 segment 元数据：可序列化到 JSON，用于准确定位到
    哪个 ppt/pdf、哪一页、哪个章节(section/title)。
    """
    segment_id: str
    page_idx: int
    title: str
    doc_id: str
    filename: str
    text_content: str
    tags: list[str] = field(default_factory=list)
    # 预处理阶段计算好的 embedding 及其模型名（可选）
    embedding: list[float] | None = None
    embedding_model: str | None = None

    @property
    def page_display(self) -> int:
        """展示用页码（1-based）。"""
        return self.page_idx + 1
