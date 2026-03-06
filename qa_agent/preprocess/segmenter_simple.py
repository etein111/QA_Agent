# 按页将 MinerU 元素聚合成 Segment（不依赖 VLM），与参考项目 fallback 逻辑一致
from __future__ import annotations

from collections import defaultdict

from .data_types import BBox, MinerUElement, Segment, SegmentMetadata


def _segment_title_from_page_elements(elements: list[MinerUElement], page_idx: int) -> str:
    """优先取第一个 header 文本，否则取首段非空文本，再否则「第 N 页」。"""
    for e in elements:
        if e.type == "header" and e.text and e.text.strip():
            return e.text.strip()[:120]
    for e in elements:
        if e.text and e.text.strip():
            return e.text.strip()[:120]
    if any(e.type == "image" for e in elements):
        return "[Image]"
    if any(e.type == "table" for e in elements):
        return "[Table]"
    return f"[第 {page_idx + 1} 页]"


def _text_content_from_elements(elements: list[MinerUElement]) -> str:
    """拼接 segment 内所有文本，供检索用。"""
    parts = []
    for e in elements:
        if e.text and e.text.strip():
            parts.append(e.text.strip())
    return "\n".join(parts)


def build_slide_context_simple(elements: list[MinerUElement]) -> list[Segment]:
    """
    按页将元素聚合成 Segment：每页一个 Segment（与参考项目 VLM 失败时的 fallback 一致）。
    不依赖页图与 VLM，仅用 content_list 即可。
    """
    pages: dict[int, list[MinerUElement]] = defaultdict(list)
    for e in elements:
        if e.type in ("footer", "page_number"):
            continue
        pages[e.page_idx].append(e)

    if not pages:
        return []

    max_page = max(pages.keys())
    segments: list[Segment] = []
    seg_id_counter = 0

    for page_idx in range(max_page + 1):
        page_items = pages.get(page_idx, [])
        if not page_items:
            continue
        seg_id_counter += 1
        segment_id = f"p{page_idx:02d}_s{seg_id_counter:04d}"

        x0, y0, x1, y1 = page_items[0].bbox
        ub = BBox(x0, y0, x1, y1)
        for e in page_items[1:]:
            a0, b0, a1, b1 = e.bbox
            ub = ub.union(BBox(a0, b0, a1, b1))

        title = _segment_title_from_page_elements(page_items, page_idx)
        tags = set()
        if any(e.type == "equation" for e in page_items):
            tags.add("equation")
        if any(e.type == "table" for e in page_items):
            tags.add("table")
        if any(e.type == "image" for e in page_items):
            tags.add("image")
        if any(e.type == "header" for e in page_items):
            tags.add("header")

        segments.append(
            Segment(
                segment_id=segment_id,
                page_idx=page_idx,
                title=title,
                elements=page_items,
                bbox_union=ub,
                anchor_points=[ub.center],
                tags=tags,
            )
        )

    return segments


def segments_to_metadata(
    segments: list[Segment],
    doc_id: str,
    filename: str,
) -> list[SegmentMetadata]:
    """将 Segment 转为供检索与引用的 SegmentMetadata。"""
    out = []
    for seg in segments:
        text_content = _text_content_from_elements(seg.elements)
        out.append(
            SegmentMetadata(
                segment_id=seg.segment_id,
                page_idx=seg.page_idx,
                title=seg.title,
                doc_id=doc_id,
                filename=filename,
                text_content=text_content,
                tags=sorted(seg.tags),
            )
        )
    return out
