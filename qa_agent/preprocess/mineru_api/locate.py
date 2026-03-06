from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class MinerUOutputNotFound(FileNotFoundError):
    pass


@dataclass(frozen=True)
class MinerUOutputs:
    extract_dir: Path
    images_dir: Optional[Path]
    content_list_json: Path
    full_md: Optional[Path]
    origin_pdf: Optional[Path]
    layout_json: Optional[Path]
    model_json: Optional[Path]


def _pick_one(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    paths.sort(key=lambda p: (-p.stat().st_size, len(p.name)))
    return paths[0]


def locate_outputs(extract_dir: str | Path) -> MinerUOutputs:
    d = Path(extract_dir).expanduser().resolve()
    if not d.exists():
        raise MinerUOutputNotFound(f"extract_dir not found: {d}")
    if not d.is_dir():
        raise MinerUOutputNotFound(f"extract_dir is not a directory: {d}")

    images_dir = d / "images"
    if not images_dir.exists() or not images_dir.is_dir():
        images_dir = None

    content_list_candidates = list(d.glob("*_content_list.json"))
    if not content_list_candidates:
        content_list_candidates = list(d.rglob("*content_list*.json"))
    content_list_json = _pick_one(content_list_candidates)
    if content_list_json is None:
        raise MinerUOutputNotFound(
            f"No content_list json found under: {d}\nExpected something like '*_content_list.json'."
        )

    full_md = d / "full.md"
    if not full_md.exists():
        full_md = _pick_one(list(d.glob("*.md")))
    if full_md is not None and not full_md.exists():
        full_md = None

    origin_pdf = _pick_one(list(d.glob("*_origin.pdf")))
    if origin_pdf is None:
        origin_pdf = _pick_one(list(d.glob("*.pdf")))

    layout_json = d / "layout.json"
    if not layout_json.exists():
        layout_json = _pick_one(list(d.glob("*layout*.json")))
    if layout_json is not None and not layout_json.exists():
        layout_json = None

    model_json = _pick_one(list(d.glob("*_model.json")))
    if model_json is None:
        model_json = _pick_one(
            [p for p in d.glob("*.json") if p.name != "layout.json" and p != content_list_json]
        )

    return MinerUOutputs(
        extract_dir=d,
        images_dir=images_dir,
        content_list_json=content_list_json,
        full_md=full_md if full_md and full_md.exists() else None,
        origin_pdf=origin_pdf if origin_pdf and origin_pdf.exists() else None,
        layout_json=layout_json if layout_json and layout_json.exists() else None,
        model_json=model_json if model_json and model_json.exists() else None,
    )
