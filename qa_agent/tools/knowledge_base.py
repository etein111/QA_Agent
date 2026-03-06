# 内部知识库：基于预处理得到的 SegmentMetadata 做检索，精确定位到 ppt/pdf、页码、章节
# 支持 SQLite 持久化（元数据 + embedding），避免每次启动重算向量
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI

from ..config import get_settings, optional_http_client
from ..preprocess import load_segments_json, SegmentMetadata

# 环境变量：设为 false 禁用 SQLite，仅从 data/preprocessed/*.json 加载
_QA_KB_USE_SQLITE = os.getenv("QA_KB_USE_SQLITE", "true").strip().lower() in ("true", "1", "yes")
# 环境变量：知识库 DB 路径，默认 project_root/data/kb.sqlite
def _default_kb_db_path() -> Path:
    root = Path(get_settings().project_root)
    return Path(os.getenv("QA_KB_DB_PATH", str(root / "data" / "kb.sqlite")))


_SEGMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    page_idx INTEGER NOT NULL,
    text_content TEXT,
    tags TEXT,
    embedding_model TEXT,
    embedding_blob BLOB
);
"""


def sync_segments_to_sqlite(
    segments: List[SegmentMetadata],
    db_path: str | Path,
    embed_model: str,
) -> None:
    """
    将一批 SegmentMetadata 写入/更新到 kb.sqlite，供预处理流水线在写出 JSON 后调用。
    若 segment 已带 embedding 且 embedding_model 与 embed_model 一致，则写入 embedding_blob。
    """
    if not segments:
        return
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SEGMENTS_TABLE)
        for seg in segments:
            # 不同文档的 segment_id 可能重复（如 p00_s0001），用 doc_id 前缀保证主键唯一
            stored_id = f"{seg.doc_id}__{seg.segment_id}"
            tags_json = json.dumps(seg.tags, ensure_ascii=False)
            blob: bytes | None = None
            emb_model: str | None = None
            if getattr(seg, "embedding", None) and getattr(seg, "embedding_model", None) == embed_model:
                blob = np.array(seg.embedding, dtype=np.float32).tobytes()
                emb_model = embed_model
            conn.execute(
                """INSERT OR REPLACE INTO segments
                   (segment_id, doc_id, filename, title, page_idx, text_content, tags, embedding_model, embedding_blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored_id,
                    seg.doc_id,
                    seg.filename,
                    seg.title,
                    seg.page_idx,
                    seg.text_content or "",
                    tags_json,
                    emb_model,
                    blob,
                ),
            )
        conn.commit()
    finally:
        conn.close()


@dataclass
class RetrievedChunk:
    """检索结果，用于回答中引用：可精确定位到文件、页码、章节(section)。"""

    doc_id: str
    filename: str
    title: str
    page: int | None
    section: str | None
    snippet: str

    def format_citation(self) -> str:
        parts: List[str] = [f"文件名：{self.filename}"]
        if self.page is not None:
            parts.append(f"第 {self.page} 页")
        if self.title:
            parts.append(f"标题：{self.title}")
        if self.section:
            parts.append(self.section)
        return "【" + "，".join(parts) + "】"


def _load_all_segments(preprocessed_dir: str | Path) -> List[SegmentMetadata]:
    """从 data/preprocessed 下所有 *_segments.json 加载 SegmentMetadata 列表。"""
    d = Path(preprocessed_dir)
    if not d.exists() or not d.is_dir():
        return []
    out: List[SegmentMetadata] = []
    for p in d.glob("*_segments.json"):
        try:
            _, _, _, segments = load_segments_json(p)
            out.extend(segments)
        except Exception:
            continue
    return out


def _snippet(text: str, max_len: int = 280) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def _log_kb_retrieval(
    query: str,
    scored_segments: List[tuple[float, SegmentMetadata]],
    top_k: int,
) -> None:
    """
    将知识库检索的打分结果写入项目根目录的 retrieval-debug.log。
    """
    import json as _json
    import time

    settings = get_settings()
    root = Path(settings.project_root)
    log_path = root / "retrieval-debug.log"

    ts = int(time.time() * 1000)
    payload = {
        "timestamp": ts,
        "query": query,
        "top_k": top_k,
        "segments": [
            {
                "score": float(score),
                "doc_id": seg.doc_id,
                "filename": seg.filename,
                "page_idx": seg.page_idx,
                "page": seg.page_display,
                "title": seg.title,
                "snippet": _snippet(seg.text_content, max_len=160),
            }
            for score, seg in scored_segments[: top_k * 3]
        ],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败时静默忽略，避免影响正常请求
        pass


class KnowledgeBase:
    """
    基于 SegmentMetadata 的内部知识库：检索时精确定位到哪个 ppt/pdf、哪一页、哪个章节(section)。
    数据来源：若启用 QA_KB_USE_SQLITE，则从 SQLite（默认 data/kb.sqlite）加载，缺数据时从
    data/preprocessed/*_segments.json 同步；否则仅从 JSON 加载。
    """

    def __init__(
        self,
        preprocessed_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        use_sqlite: bool | None = None,
    ) -> None:
        settings = get_settings()
        root = Path(settings.project_root)
        self._preprocessed_dir = Path(preprocessed_dir or root / "data" / "preprocessed")
        self._segments: List[SegmentMetadata] = []
        self._segment_vectors: List[Tuple[SegmentMetadata, np.ndarray]] = []
        self._embed_client: OpenAI | None = None
        self._embed_model: str = os.getenv("QA_EMBEDDING_MODEL", "text-embedding-3-small")

        use_db = use_sqlite if use_sqlite is not None else _QA_KB_USE_SQLITE
        self._db_path: Path | None = Path(db_path) if db_path else (_default_kb_db_path() if use_db else None)

        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.executescript(_SEGMENTS_TABLE)
                cur = conn.execute("SELECT COUNT(*) FROM segments")
                if cur.fetchone()[0] == 0:
                    self._sync_from_preprocessed(conn)
                self._load_from_db(conn)
            finally:
                conn.close()
        else:
            self._reload()

    def _sync_from_preprocessed(self, conn: sqlite3.Connection) -> None:
        """从 data/preprocessed/*_segments.json 同步到 SQLite；已有 embedding 的按模型写入 blob。"""
        segments = _load_all_segments(self._preprocessed_dir)
        for seg in segments:
            stored_id = f"{seg.doc_id}__{seg.segment_id}"
            tags_json = json.dumps(seg.tags, ensure_ascii=False)
            blob: bytes | None = None
            emb_model: str | None = None
            if getattr(seg, "embedding", None) and getattr(seg, "embedding_model", None) == self._embed_model:
                blob = np.array(seg.embedding, dtype=np.float32).tobytes()
                emb_model = self._embed_model
            conn.execute(
                """INSERT OR REPLACE INTO segments
                   (segment_id, doc_id, filename, title, page_idx, text_content, tags, embedding_model, embedding_blob)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stored_id,
                    seg.doc_id,
                    seg.filename,
                    seg.title,
                    seg.page_idx,
                    seg.text_content or "",
                    tags_json,
                    emb_model,
                    blob,
                ),
            )
        conn.commit()

    def _load_from_db(self, conn: sqlite3.Connection) -> None:
        """从 SQLite 加载 segments 与向量到内存；缺失或模型不一致的 embedding 现场计算并回写。"""
        cur = conn.execute(
            "SELECT segment_id, doc_id, filename, title, page_idx, text_content, tags, embedding_model, embedding_blob FROM segments"
        )
        rows = cur.fetchall()
        self._segments = []
        self._segment_vectors = []

        for row in rows:
            seg_id, doc_id, filename, title, page_idx, text_content, tags_str, emb_model, emb_blob = row
            tags = json.loads(tags_str) if tags_str else []
            if emb_blob is not None and emb_model == self._embed_model:
                vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
                emb_list: List[float] | None = vec.tolist()
            else:
                vec = None
                emb_list = None
            seg = SegmentMetadata(
                segment_id=seg_id,
                doc_id=doc_id,
                filename=filename or "",
                title=title or "",
                page_idx=int(page_idx),
                text_content=text_content or "",
                tags=tags,
                embedding=emb_list,
                embedding_model=self._embed_model if emb_list else None,
            )
            self._segments.append(seg)
            if vec is not None:
                self._segment_vectors.append((seg, vec))
            else:
                content = (seg.text_content or "")[:1024]
                if seg.title:
                    content = f"{seg.title}\n{content}"
                try:
                    v = self._encode_text(content)
                    self._segment_vectors.append((seg, v))
                    conn.execute(
                        "UPDATE segments SET embedding_model = ?, embedding_blob = ? WHERE segment_id = ?",
                        (self._embed_model, v.tobytes(), seg_id),
                    )
                except Exception:
                    self._segment_vectors.append((seg, np.zeros(1, dtype=np.float32)))
        conn.commit()

    def _reload(self) -> None:
        """仅从 JSON 加载（未使用 SQLite 时）。"""
        self._segments = _load_all_segments(self._preprocessed_dir)
        self._segment_vectors = []

    def _get_embed_client(self) -> OpenAI:
        if self._embed_client is None:
            settings = get_settings().llm
            if not settings.api_key:
                raise RuntimeError(
                    "未配置 AIHUB_API_KEY 或 OPENAI_API_KEY 环境变量，无法调用 embedding 接口。"
                )
            # 这里直接使用 OpenAI 客户端，走同一个 aihub 中转站
            self._embed_client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key,
                timeout=settings.timeout,
                http_client=optional_http_client(settings.timeout),
            )
        return self._embed_client

    def _encode_text(self, text: str) -> np.ndarray:
        """
        使用 OpenAI embedding 将文本编码为向量。
        """
        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype=np.float32)
        client = self._get_embed_client()
        resp = client.embeddings.create(
            model=self._embed_model,
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def _ensure_segment_vectors(self) -> None:
        """
        为所有 SegmentMetadata 构建向量表示。
        """
        if self._segment_vectors:
            return
        vectors: List[Tuple[SegmentMetadata, np.ndarray]] = []
        for seg in self._segments:
            # 若预处理阶段已经计算并存储了与当前 embedding 模型一致的向量，则直接复用
            if getattr(seg, "embedding", None) and getattr(seg, "embedding_model", None) == self._embed_model:
                vec = np.array(seg.embedding, dtype=np.float32)
            else:
                # 否则在线计算一次，并仅缓存在内存中
                content = (seg.text_content or "")[:1024]
                if seg.title:
                    content = f"{seg.title}\n{content}"
                try:
                    vec = self._encode_text(content)
                except Exception:
                    # embedding 失败时用零向量占位，避免中断整个检索
                    vec = np.zeros(1, dtype=np.float32)
            vectors.append((seg, vec))
        self._segment_vectors = vectors

    def _search_scored(self, query: str) -> List[Tuple[float, SegmentMetadata]]:
        """
        对单个 query 为所有 segment 打分并按分数降序排列返回。
        供 search / search_multi 复用。
        """
        if not self._segments:
            return []
        try:
            self._ensure_segment_vectors()
            query_vec = self._encode_text(query)
            scored: List[Tuple[float, SegmentMetadata]] = []
            for seg, vec in self._segment_vectors:
                score = _cosine_similarity(query_vec, vec)
                scored.append((score, seg))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        except Exception:
            def _fallback_score(seg: SegmentMetadata) -> float:
                text = (seg.text_content or "") + " " + (seg.title or "")
                return float(text.count(query))
            scored = [(_fallback_score(seg), seg) for seg in self._segments]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        基于 SegmentMetadata 的检索：按 query 与 text_content/title 相关性排序，
        返回可精确定位到文件、页码、章节(section) 的 RetrievedChunk。
        """
        scored = self._search_scored(query)
        _log_kb_retrieval(query, scored, top_k=top_k)
        # 使用与 web_search 相同的阈值：若所有相似度都小于 WEB_SEARCH_MIN_SIM，则认为知识库无相关内容
        min_sim = float(os.getenv("WEB_SEARCH_MIN_SIM", "0.40"))
        filtered = [(score, seg) for score, seg in scored if score >= min_sim]
        if not filtered:
            return []
        top = [seg for score, seg in filtered[:top_k]]
        return [
            RetrievedChunk(
                doc_id=seg.doc_id,
                filename=seg.filename,
                title=seg.title or f"第 {seg.page_display} 页",
                page=seg.page_display,
                section=seg.title or None,
                snippet=_snippet(seg.text_content),
            )
            for seg in top
        ]

    def search_multi(self, queries: List[str], top_k: int = 5) -> List[RetrievedChunk]:
        """
        多子查询检索：对每个 query 分别打分，按 segment 合并取各 query 下的最高分。
        保证每个子查询至少召回其得分最高的 1 条（若 score>0），再按合并分数填满 top_k，
        避免「高阶偏微分 自然谬误」只召回偏微分、自然谬误零召回。
        """
        if not queries or not self._segments:
            return self.search(queries[0], top_k=top_k) if queries else []
        if len(queries) == 1:
            return self.search(queries[0], top_k=top_k)

        min_sim = float(os.getenv("WEB_SEARCH_MIN_SIM", "0.40"))

        best_score: dict[str, Tuple[float, SegmentMetadata]] = {}
        per_query_top: List[Tuple[float, SegmentMetadata]] = []
        for q in queries:
            q = (q or "").strip()
            if not q:
                continue
            scored = self._search_scored(q)
            if scored and scored[0][0] >= min_sim:
                per_query_top.append(scored[0])
            for score, seg in scored:
                if seg.segment_id not in best_score or score > best_score[seg.segment_id][0]:
                    best_score[seg.segment_id] = (score, seg)
        merged = sorted(best_score.values(), key=lambda x: x[0], reverse=True)
        _log_kb_retrieval("|".join(queries), merged, top_k=top_k)

        seen_ids: set[str] = set()
        top: List[SegmentMetadata] = []
        for score, seg in per_query_top:
            if seg.segment_id not in seen_ids:
                seen_ids.add(seg.segment_id)
                top.append(seg)
        for score, seg in merged:
            if len(top) >= top_k:
                break
            if seg.segment_id not in seen_ids and score >= min_sim:
                seen_ids.add(seg.segment_id)
                top.append(seg)
        # 若所有子查询下的最高分都低于阈值，则认为知识库对该问题没有可靠命中
        if not top:
            return []
        return [
            RetrievedChunk(
                doc_id=seg.doc_id,
                filename=seg.filename,
                title=seg.title or f"第 {seg.page_display} 页",
                page=seg.page_display,
                section=seg.title or None,
                snippet=_snippet(seg.text_content),
            )
            for seg in top
        ]


__all__ = ["KnowledgeBase", "RetrievedChunk", "sync_segments_to_sqlite"]
