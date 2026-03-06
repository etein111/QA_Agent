from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple
import os

import numpy as np
from ddgs import DDGS
from openai import OpenAI
import requests

from ..config import get_settings, optional_http_client
from dataclasses import dataclass
from typing import List


@dataclass
class WebSearchResult:
    """
    联网搜索结果的数据结构占位。
    当前项目已不再在后端主动调用第三方搜索引擎，
    仅保留该类型以兼容现有导入与类型标注。
    """
    title: str
    url: str
    snippet: str

    def format_citation(self) -> str:
        return f"【来源：{self.title}，{self.url}】"


def search_web(query: str, max_results: int = 3) -> List[WebSearchResult]:
    """
    联网搜索占位函数。
    实际联网搜索已交由上游 LLM（例如 gpt-5-mini）的自带联网能力完成，
    后端不再调用 DDG / Google 等第三方搜索 API。
    这里始终返回空列表，表示“本地未执行外部搜索”。
    """
    return []


__all__ = ["WebSearchResult", "search_web"]


def _keyword_fallback(query: str) -> str:
    """
    规则式关键词提取兜底：当 LLM 改写始终返回 null 时，去掉口语化 filler，保留核心词/短语，用空格连接。
    不依赖模型，保证搜索 query 可缩短、可读。
    """
    if not query or not query.strip():
        return query
    s = query.strip()
    # 常见口语/ filler（中英），用空格替代便于后续按空格分
    fillers_cn = [
        "具体是什么", "又有什么", "是什么意思", "是什么", "有什么", "有什么性质",
        "请", "请问", "告诉我", "想了解一下", "能不能", "可以", "怎样", "如何",
        "呢", "吗", "啊", "的", "了", "在", "时候", "时候要注意什么",
    ]
    fillers_en = [
        r"\bwhat\s+is\b", r"\btell\s+me\s+about\b", r"\bplease\b", r"\bcan\s+you\b",
        r"\bhow\s+to\b", r"\bwhat\s+are\b", r"\bexplain\b", r"\bdescribe\b",
    ]
    for f in fillers_cn:
        s = s.replace(f, " ")
    for pat in fillers_en:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    # 按标点拆成片段，保留长度>=2 的（避免单字噪音）
    parts = re.split(r"[，。？、,?!\s]+", s)
    kept = [p.strip() for p in parts if len(p.strip()) >= 2]
    result = " ".join(kept).strip()
    return result if result else query.strip()


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def format_citation(self) -> str:
        return f"【来源：{self.title}，{self.url}】"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def _get_embed_client() -> Tuple[OpenAI, str]:
    """
    返回用于重排搜索结果的 embedding 客户端及模型名。
    如无法创建客户端（如无 API key），由调用方决定是否回退。
    """
    settings = get_settings().llm
    api_key = settings.api_key
    if not api_key:
        raise RuntimeError("缺少 AIHUB_API_KEY/OPENAI_API_KEY，无法为 web_search 使用向量重排。")
    client = OpenAI(
        base_url=settings.base_url,
        api_key=api_key,
        timeout=settings.timeout,
        http_client=optional_http_client(settings.timeout),
    )
    model = os.getenv("QA_EMBEDDING_MODEL", "text-embedding-3-small")
    return client, model


def _rewrite_query_for_search(original_query: str) -> str:
    """
    使用 LLM 将用户问题改写成一条简短、关键词明确的搜索 query，便于多领域下首轮检索与相似度更准。
    若调用失败或未配置，返回原始 query。
    """
    if not original_query or not original_query.strip():
        return original_query
    settings = get_settings().llm
    if not settings.api_key:
        return original_query
    try:
        timeout = min(15.0, settings.timeout)
        client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=timeout,
            http_client=optional_http_client(timeout),
        )
        sys = (
                        "You are a search query generator. Your task is to extract core keywords from the user's question "
            "to form a highly effective search query for a search engine.\n"
            "Rules:\n"
            "1. Keep the same language as the user's input.\n"
            "2. Remove conversational filler words (e.g., 'what is', 'please').\n"
            "3. Separate distinct concepts with spaces.\n"
            "4. Output ONLY the keywords text. Do not output JSON or Markdown.\n\n"
            "Examples:\n"
            "User: What is the capital of France?\n"
            "Assistant: France capital\n"
            "User: python list vs tuple performance\n"
            "Assistant: python list tuple performance comparison\n"
            "User: 介绍一下量子纠缠\n"
            "Assistant: 量子纠缠 原理\n"
            "User: 自然谬误和全微分都是什么\n"
            "Assistant: 自然谬误 全微分 定义"
        )
        resp = client.chat.completions.create(
            model=settings.keyword_model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": original_query.strip()},
            ],
            max_tokens=150,
            temperature=0,
        )
        # 兼容 content 为 null、choices 为空或 message 缺失（部分模型/拦截策略会返回空内容）
        if not getattr(resp, "choices", None) or len(resp.choices) == 0:
            return _keyword_fallback(original_query)
        msg = getattr(resp.choices[0], "message", None)
        content = getattr(msg, "content", None) if msg else None
        rewritten = (content or "").strip()
        if rewritten:
            return rewritten
        # LLM 每次返回 null 时用规则式关键词兜底，仍能缩短 query、提升搜索效果
        return _keyword_fallback(original_query)
    except Exception:
        pass
    return _keyword_fallback(original_query)


def _debug_log(message: str, data: dict) -> None:
    """
    向 debug-6dec40.log 追加一行 NDJSON，用于调试 web_search 的向量重排行为。
    """
    import json
    import time

    settings = get_settings()
    root = Path(settings.project_root)
    log_path = root / "debug-6dec40.log"
    payload = {
        "sessionId": "6dec40",
        "runId": "web_search",
        "hypothesisId": "H_web_search_relevance",
        "location": "qa_agent/tools/web_search.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    # #region agent log
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion agent log


def _log_web_search(query: str, items: List[Tuple[float, WebSearchResult]]) -> None:
    """
    将联网搜索的结果及相似度写入 retrieval-debug.log，便于查看与 query 的相关性。
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
        "type": "web_search",
        "results": [
            {
                "score": float(score),
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            }
            for score, r in items
        ],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试用日志失败不影响主流程
        pass


def _log_web_search_raw(
    provider: str,
    original_query: str,
    search_query: str,
    raw_results: list[dict],
) -> None:
    """
    将底层搜索源（DDG / Google）返回的原始结果写入 retrieval-debug.log，便于对比过滤前后的差异。
    """
    import json as _json
    import time

    settings = get_settings()
    root = Path(settings.project_root)
    log_path = root / "retrieval-debug.log"

    ts = int(time.time() * 1000)
    payload = {
        "timestamp": ts,
        "query": original_query,
        "search_query": search_query,
        "provider": provider,
        "type": "web_search_raw",
        "results": [
            {
                "title": r.get("title") or r.get("name") or "",
                "url": r.get("href") or r.get("url") or r.get("link") or "",
                "snippet": r.get("body") or r.get("snippet") or "",
            }
            for r in raw_results
        ],
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _ddg_fetch(search_query: str, raw_limit: int) -> list[dict]:
    parts = [p.strip() for p in search_query.split() if p.strip()]
    if not parts:
        parts = [search_query]
    seen_urls = set()
    raw: list[dict] = []
    ssl_verify = os.getenv("QA_SSL_VERIFY", "true").strip().lower() not in ("false", "0", "no")
    with DDGS(proxy="http://127.0.0.1:7890", verify=ssl_verify) as ddgs:
        per_limit = max(raw_limit // len(parts), 5)
        for q in parts:
            for r in ddgs.text(q, region="cn-zh", max_results=per_limit):
                url = r.get("href") or r.get("url") or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw.append(r)
    return raw[:raw_limit]


def _google_fetch(search_query: str, raw_limit: int) -> list[dict]:
    """
    使用 Google Custom Search API 进行搜索。
    需配置环境变量 GOOGLE_CSE_API_KEY 与 GOOGLE_CSE_CX。
    返回结构与 _ddg_fetch 略兼容（包含 title/href/body）。
    """
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cx = os.getenv("GOOGLE_CSE_CX", "").strip()
    if not api_key or not cx:
        return []
    params = {
        "key": api_key,
        "cx": cx,
        "q": search_query,
        "num": min(raw_limit, 10),
        "hl": "zh-CN",
        "lr": "lang_zh-CN",
        "safe": "active",
    }
    try:
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        raw: list[dict] = []
        for it in items:
            raw.append(
                {
                    "title": it.get("title") or "",
                    "href": it.get("link") or "",
                    "body": it.get("snippet") or "",
                }
            )
        return raw
    except Exception:
        return []


def search_web(query: str, max_results: int = 3) -> List[WebSearchResult]:
    """
    使用 DuckDuckGo 联网搜索，并对结果做向量相似度重排。
    空结果时自动重试一次并可选回退到原 query，提高稳定性。
    """
    import time

    # 空 query 直接短路，避免对 DuckDuckGo 发送无效请求（会触发 keywords is mandatory）
    if not (query or "").strip():
        _debug_log(
            "empty_query_short_circuit",
            {"query": query, "max_results": max_results},
        )
        _log_web_search(query, [])
        return []

    use_rewrite = os.getenv("QA_WEB_SEARCH_REWRITE_QUERY", "").strip().lower() in ("1", "true", "yes")
    search_query = _rewrite_query_for_search(query) if use_rewrite else query
    raw_limit = max(max_results * 3, 20)

    provider = os.getenv("QA_WEB_SEARCH_PROVIDER", "ddg").strip().lower()

    def _fetch_once(q: str) -> List[dict]:
        if provider == "google":
            google_results = _google_fetch(q, raw_limit)
            if google_results:
                return google_results
            # 若 Google 无结果或配置缺失，退回 DDG
        return _ddg_fetch(q, raw_limit)

    raw_results: List[dict] = _fetch_once(search_query)
    if not raw_results and search_query != query:
        time.sleep(1.5)
        raw_results = _fetch_once(query)
    if not raw_results:
        time.sleep(1.5)
        raw_results = _fetch_once(search_query)

    # 记录底层搜索源原始返回结果，便于对比过滤前后行为
    _log_web_search_raw(provider=provider, original_query=query, search_query=search_query, raw_results=raw_results)

    results: List[WebSearchResult] = [
        WebSearchResult(
            title=r.get("title") or "未命名页面",
            url=r.get("href") or "",
            snippet=r.get("body") or "",
        )
        for r in raw_results
    ]

    # 额外稳健性过滤：要求核心关键词至少出现在标题或摘要中，避免「不确定性」被搜成各种「不……」用法
    core_kw = None
    try:
        kw_line = _keyword_fallback(query)
        if kw_line:
            core_kw = kw_line.split()[0]
    except Exception:
        core_kw = None
    if core_kw:
        filtered: List[WebSearchResult] = []
        for r in results:
            text = f"{r.title}\n{r.snippet}"
            if core_kw in text:
                filtered.append(r)
        if filtered:
            results = filtered

    if not results:
        _log_web_search(query, [])
        return []

    scored: List[Tuple[float, WebSearchResult]] = []
    try:
        client, model = _get_embed_client()
        from numpy import array

        resp_q = client.embeddings.create(model=model, input=search_query)
        q_vec = array(resp_q.data[0].embedding, dtype=np.float32)

        # 将 URL 纳入嵌入文本，利用域名/路径中的主题信息提升多领域下的相似度区分度
        texts = [(r, f"{r.title}\n{r.url}\n{r.snippet}") for r in results]
        resp = client.embeddings.create(model=model, input=[t for _, t in texts])
        for (r, _), emb in zip(texts, resp.data):
            v = array(emb.embedding, dtype=np.float32)
            score = _cosine_similarity(q_vec, v)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        strict_threshold = float(os.getenv("WEB_SEARCH_MIN_SIM", "0.50"))
        relaxed_threshold = float(os.getenv("WEB_SEARCH_RELAXED_MIN_SIM", "0.30"))

        tier = "strict"
        # 第一层：严格阈值
        strict_scored = [(s, r) for s, r in scored if s >= strict_threshold]

        if strict_scored:
            chosen = strict_scored
        else:
            # 第二层：放宽阈值，仅在本地过滤，不再额外请求 DDG
            tier = "relaxed"
            relaxed_scored = [(s, r) for s, r in scored if s >= relaxed_threshold]
            if relaxed_scored:
                chosen = relaxed_scored
            else:
                # 第三层：仍无明显高分结果，直接判定为“无可靠结果”，不再退回到简单 topN
                tier = "none"
                chosen = []

        top_items = chosen[:max_results]
        _log_web_search(query, top_items)
        _debug_log(
            "rerank_success",
            {
                "query_preview": query[:50],
                "tier": tier,
                "strict_threshold": strict_threshold,
                "relaxed_threshold": relaxed_threshold,
                "top": [
                    {"score": float(s), "title": it.title, "url": it.url}
                    for s, it in top_items[:3]
                ],
            },
        )
        return [r for _, r in top_items]

    except Exception as e:
        _debug_log(
            "rerank_failed",
            {"error": repr(e), "fallback_count": min(len(results), max_results)},
        )
        fallback_items = [(0.0, r) for r in results[:max_results]]
        _log_web_search(query, fallback_items)
        return results[:max_results]


__all__ = ["WebSearchResult", "search_web"]


