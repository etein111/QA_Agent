# 数据预处理流水线：data 下 ppt/pdf -> MinerU 解析 -> Segment 提取 -> data/preprocessed/*.json
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import os

from openai import OpenAI

from .data_types import SegmentMetadata
from .mineru_loader import load_mineru_pre_content_list
from .mineru_api.client import parse_files, MinerUApiError
from .mineru_api.locate import locate_outputs, MinerUOutputNotFound
from .segmenter_simple import build_slide_context_simple, segments_to_metadata


def _debug_log(hypothesis_id: str, message: str, data: dict) -> None:
    """
    将调试信息以 NDJSON 形式写入项目根的 debug-6dec40.log，用于分析同名 ppt/pptx 复用 MinerU 输出的问题。
    """
    import time
    import json as _json

    root = Path(__file__).resolve().parents[2]
    log_path = root / "debug-6dec40.log"
    payload = {
        "sessionId": "6dec40",
        "runId": "preprocess-pipeline",
        "hypothesisId": hypothesis_id,
        "location": "qa_agent/preprocess/pipeline.py:run_for_file",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    # #region agent log
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试失败静默忽略
        pass
    # #endregion agent log


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-_]+", "", s)
    return s or "doc"


def segment_metadata_to_dict(m: SegmentMetadata) -> dict:
    return {
        "segment_id": m.segment_id,
        "page_idx": m.page_idx,
        "title": m.title,
        "doc_id": m.doc_id,
        "filename": m.filename,
        "text_content": m.text_content,
        "tags": m.tags,
        "embedding": m.embedding,
        "embedding_model": m.embedding_model,
    }



def load_segments_json(path: str | Path) -> tuple[str, str, str | None, list[SegmentMetadata]]:
    """加载 preprocessed 的 segments JSON，返回 (doc_id, filename, embedding_model, segments)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc_id = data.get("doc_id", "")
    filename = data.get("filename", "")
    embedding_model = data.get("embedding_model")
    segments = [
        SegmentMetadata(
            segment_id=s["segment_id"],
            page_idx=int(s["page_idx"]),
            title=s["title"],
            doc_id=s.get("doc_id", doc_id),
            filename=s.get("filename", filename),
            text_content=s.get("text_content", ""),
            tags=s.get("tags") or [],
            embedding=s.get("embedding"),
            embedding_model=s.get("embedding_model", embedding_model),
        )
        for s in data.get("segments", [])
    ]
    return doc_id, filename, embedding_model, segments


def run_for_file(
    file_path: str | Path,
    data_root: str | Path,
    mineru_out_root: Optional[str | Path] = None,
    preprocessed_root: Optional[str | Path] = None,
    use_existing_mineru: bool = True,
    project_root: Optional[str | Path] = None,
) -> Path:
    """
    对单个 pptx/pdf 执行：MinerU 解析（或使用已有输出）-> content_list -> Segment -> 写出 JSON。

    :param file_path: data 下的 pptx 或 pdf 路径
    :param data_root: data 目录（用于解析 mineru_out 与 preprocessed 子目录）
    :param mineru_out_root: MinerU 解压目录根，默认 data_root / "mineru_output"
    :param preprocessed_root: 预处理 JSON 输出目录，默认 data_root / "preprocessed"
    :param use_existing_mineru: 若已存在对应 stem__mineru 目录则直接用，不再调 API
    :param project_root: 项目根目录（用于 parse_files 解析相对路径）
    :return: 写出的 segments JSON 路径
    """
    path = Path(file_path).resolve()
    data_root = Path(data_root).resolve()
    mineru_out_root = Path(mineru_out_root or data_root / "mineru_output").resolve()
    preprocessed_root = Path(preprocessed_root or data_root / "preprocessed").resolve()
    project_root = Path(project_root) if project_root else data_root.parent

    if not path.exists():
        raise FileNotFoundError(path)
    stem = path.stem
    suffix = path.suffix.lower()
    if suffix not in (".pptx", ".ppt", ".pdf"):
        raise ValueError(f"Only .pptx/.ppt/.pdf supported, got {path.name}")

    # 同名 ppt/pdf 需要不同 doc_id，这里把扩展名也编码进去
    # 例如：partial_derivative-pptx、partial_derivative-pdf
    doc_id = slugify(f"{stem}-{suffix.lstrip('.')}")
    filename = path.name

    # 为不同扩展名的同名文件使用不同的 MinerU 输出子目录，避免共享同一个 extract_dir
    file_out_root = mineru_out_root / f"{stem}-{suffix.lstrip('.')}"

    extract_dir: Optional[Path] = None
    if use_existing_mineru:
        candidate = file_out_root / f"{stem}__mineru"
        _debug_log(
            "H_same_name_reuse",
            "Checking existing MinerU output for file",
            {
                "file": str(path),
                "suffix": suffix,
                "file_out_root": str(file_out_root),
                "candidate": str(candidate),
                "candidate_exists": candidate.exists(),
                "candidate_is_dir": candidate.is_dir(),
            },
        )
        if candidate.exists() and candidate.is_dir():
            extract_dir = candidate

    if extract_dir is None:
        file_out_root.mkdir(parents=True, exist_ok=True)
        _debug_log(
            "H_same_name_reuse",
            "Calling parse_files for file (no existing MinerU output)",
            {
                "file": str(path),
                "suffix": suffix,
                "file_out_root": str(file_out_root),
            },
        )
        res = parse_files(
            input_files=[str(path)],
            out_root=str(file_out_root),
            project_root=str(project_root),
            model_version="vlm",
        )
        done = [it for it in res["items"] if it.get("state") == "done" and it.get("out_dir")]
        if not done:
            raise MinerUApiError(
                f"MinerU parse failed for {path.name}: {json.dumps(res, ensure_ascii=False, indent=2)}"
            )
        extract_dir = Path(done[0]["out_dir"])

    try:
        outs = locate_outputs(extract_dir)
    except MinerUOutputNotFound as e:
        raise FileNotFoundError(f"MinerU output incomplete: {e}") from e

    elements = load_mineru_pre_content_list(str(outs.content_list_json))
    segments = build_slide_context_simple(elements)
    meta_list = segments_to_metadata(segments, doc_id=doc_id, filename=filename)

    # --- 计算并填充 embedding（在预处理阶段完成） ---
    # 使用与在线检索相同的 embedding 模型，名称来自环境变量 QA_EMBEDDING_MODEL
    embed_model = os.getenv("QA_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        from ..config import get_settings as _get_settings, optional_http_client  # 局部导入以避免循环

        llm_settings = _get_settings().llm
        if not llm_settings.api_key:
            raise RuntimeError("缺少 AIHUB_API_KEY/OPENAI_API_KEY，无法为预处理计算 embedding。")

        client = OpenAI(
            base_url=llm_settings.base_url,
            api_key=llm_settings.api_key,
            timeout=llm_settings.timeout,
            http_client=optional_http_client(llm_settings.timeout),
        )

        texts = []
        for m in meta_list:
            content = (m.text_content or "")[:1024]
            if m.title:
                content = f"{m.title}\n{content}"
            texts.append(content or m.title or "")

        if texts:
            resp = client.embeddings.create(model=embed_model, input=texts)
            for m, emb in zip(meta_list, resp.data):
                m.embedding = emb.embedding
                m.embedding_model = embed_model
    except Exception:
        # embedding 失败时，不中断预处理，仅不写入 embedding 字段
        for m in meta_list:
            m.embedding = None
            m.embedding_model = None

    preprocessed_root.mkdir(parents=True, exist_ok=True)
    out_path = preprocessed_root / f"{doc_id}_segments.json"
    num_pages = max((s.page_idx for s in meta_list), default=-1) + 1
    payload = {
        "doc_id": doc_id,
        "filename": filename,
        "num_pages": num_pages,
        "num_segments": len(meta_list),
        "segments": [segment_metadata_to_dict(m) for m in meta_list],
    }
    # 若已成功计算 embedding，则在顶层记录 embedding_model，方便后续检测模型是否变化
    if any(m.embedding is not None and m.embedding_model for m in meta_list):
        payload["embedding_model"] = embed_model
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 预处理阶段即写入 kb.sqlite，供知识库直接加载
    if os.getenv("QA_KB_USE_SQLITE", "true").strip().lower() in ("true", "1", "yes"):
        try:
            from qa_agent.config import get_settings as _get_cfg
            from qa_agent.tools.knowledge_base import sync_segments_to_sqlite
            root = Path(_get_cfg().project_root)
            db_path = Path(os.getenv("QA_KB_DB_PATH", str(root / "data" / "kb.sqlite")))
            sync_segments_to_sqlite(meta_list, db_path, embed_model)
        except Exception:
            pass

    return out_path


def run_for_data_dir(
    data_root: str | Path,
    mineru_out_root: Optional[str | Path] = None,
    preprocessed_root: Optional[str | Path] = None,
    use_existing_mineru: bool = True,
    project_root: Optional[str | Path] = None,
    extensions: tuple[str, ...] = (".pptx", ".ppt", ".pdf"),
) -> List[Path]:
    """
    遍历 data 目录下所有 .pptx/.pdf，对每个执行 run_for_file，返回写出的 JSON 路径列表。
    """
    """
    规则：
    1）如果某个 pdf/pptx 对应的 JSON 已存在，且其中的 embedding_model 与当前环境变量 QA_EMBEDDING_MODEL 一致，
       则跳过该文件（既不重新跑 MinerU，也不重算 embedding）。
    2）否则（包括 JSON 不存在，或 embedding_model 变化），才对该文件调用 run_for_file，必要时复用已有 MinerU 输出。
    """
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        return []

    preprocessed_root = Path(preprocessed_root or data_root / "preprocessed").resolve()
    current_embed_model = os.getenv("QA_EMBEDDING_MODEL", "text-embedding-3-small")

    out_paths: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue

        stem = p.stem
        suffix = p.suffix.lower()
        doc_id = slugify(f"{stem}-{suffix.lstrip('.')}")
        json_path = preprocessed_root / f"{doc_id}_segments.json"

        # 如果已有 JSON，且其中 embedding_model 与当前 env 一致，则直接跳过该文件
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                existing_model = data.get("embedding_model")
            except Exception:
                existing_model = None

            if existing_model and existing_model == current_embed_model:
                out_paths.append(json_path)
                # 跳过时也把该文件对应的 segments 同步进 kb.sqlite，保证 DB 包含所有 JSON
                if os.getenv("QA_KB_USE_SQLITE", "true").strip().lower() in ("true", "1", "yes"):
                    try:
                        from qa_agent.config import get_settings as _get_cfg
                        from qa_agent.tools.knowledge_base import sync_segments_to_sqlite
                        root = Path(_get_cfg().project_root)
                        db_path = Path(os.getenv("QA_KB_DB_PATH", str(root / "data" / "kb.sqlite")))
                        _, _, _, segs = load_segments_json(json_path)
                        sync_segments_to_sqlite(segs, db_path, current_embed_model)
                    except Exception:
                        pass
                continue

        # 走正常流程：可能需要重新跑 MinerU 或仅重算 embedding
        try:
            out_paths.append(
                run_for_file(
                    p,
                    data_root=data_root,
                    mineru_out_root=mineru_out_root,
                    preprocessed_root=preprocessed_root,
                    use_existing_mineru=use_existing_mineru,
                    project_root=project_root,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Preprocess failed for {p.name}: {e}") from e

    # 强制将 preprocessed 下所有 *_segments.json 同步到 kb.sqlite，保证 DB 与 JSON 一致
    if os.getenv("QA_KB_USE_SQLITE", "true").strip().lower() in ("true", "1", "yes") and preprocessed_root.exists():
        try:
            import sqlite3
            from qa_agent.config import get_settings as _get_cfg
            from qa_agent.tools.knowledge_base import sync_segments_to_sqlite
            root = Path(_get_cfg().project_root)
            db_path = Path(os.getenv("QA_KB_DB_PATH", str(root / "data" / "kb.sqlite")))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            try:
                from qa_agent.tools.knowledge_base import _SEGMENTS_TABLE
                conn.executescript(_SEGMENTS_TABLE)
                conn.execute("DELETE FROM segments")
                conn.commit()
            finally:
                conn.close()
            for jp in sorted(preprocessed_root.glob("*_segments.json")):
                try:
                    _, _, emb_model, segs = load_segments_json(jp)
                    sync_segments_to_sqlite(segs, db_path, emb_model or current_embed_model)
                except Exception:
                    pass
        except Exception:
            pass

    return out_paths
