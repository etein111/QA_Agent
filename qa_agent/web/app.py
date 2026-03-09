from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from ..agent import QAAgent
from ..config import get_settings


app = FastAPI(title="QA Agent Web")

# 允许本地前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = get_settings()
_project_root = Path(_settings.project_root)
_data_dir = _project_root / "data"
_frontend_dir = _project_root / "frontend"

# 挂载静态文件：/files -> data，用于直接打开 pdf / pptx
if _data_dir.exists():
    app.mount("/files", StaticFiles(directory=str(_data_dir)), name="files")

# 前端静态资源（简单单页应用）
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


_agent = QAAgent()


def _debug_log(message: str, data: Dict[str, Any]) -> None:
    """
    向 debug-6dec40.log 追加一行 NDJSON，用于排查 Web 服务启动与请求路由问题。
    """
    import json
    import time

    payload = {
        "sessionId": "6dec40",
        "runId": "web",
        "hypothesisId": "H_web_unreachable",
        "location": "qa_agent/web/app.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = _project_root / "debug-6dec40.log"
    # #region agent log
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion agent log


def _parse_chat_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """从请求体解析 chat 可选参数，返回可传给 agent 的 kwargs。"""
    def clamp_float(v, lo, hi, default):
        try:
            x = float(v)
            return max(lo, min(hi, x))
        except (TypeError, ValueError):
            return default
    def clamp_int(v, lo, hi, default):
        try:
            x = int(v)
            return max(lo, min(hi, x))
        except (TypeError, ValueError):
            return default
    return {
        "use_web_search": bool(payload.get("use_web_search", True)),
        "temperature": clamp_float(payload.get("temperature"), 0.0, 1.0, 0.3),
        "kb_top_k": clamp_int(payload.get("kb_top_k"), 1, 20, 3),
        "web_max_results": clamp_int(payload.get("web_max_results"), 1, 15, 5),
    }


def _extract_citations(answer_text: str) -> List[Dict[str, Any]]:
    """
    从 answer.text 中抽取所有 [引用: ...] 内联引用，返回结构化 citations 列表。
    每个元素含 display_text、span(start,end)、source_type(kb|web|other)、以及可选的 kb/web 元数据。
    """
    if not answer_text:
        return []
    citations: List[Dict[str, Any]] = []
    # 匹配 [引用: ...]，内层不包含 ]，避免跨引用
    pattern = re.compile(r"\[引用:\s*([^\]]+)\]")
    for m in pattern.finditer(answer_text):
        start, end = m.start(), m.end()
        display_text = m.group(0)
        inner = (m.group(1) or "").strip()
        # 联网搜索格式：来源：标题, https://...
        if "来源：" in inner:
            source_type = "web"
            web_meta: Dict[str, Any] = {}
            rest = inner.replace("来源：", "", 1).strip()
            # 最后一个形如 https?:// 的片段视为 url，其余为 title
            url_match = re.search(r"(https?://[^\s,]+(?:\s*[^\s,]*)?)", rest)
            if url_match:
                web_meta["url"] = url_match.group(1).strip().rstrip("，, ")
                web_meta["title"] = rest[: url_match.start()].strip().rstrip("，,").strip() or rest
            else:
                web_meta["title"] = rest
            citation: Dict[str, Any] = {
                "display_text": display_text,
                "span": {"start": start, "end": end},
                "source_type": source_type,
                "web": web_meta,
            }
        else:
            source_type = "kb"
            kb_meta: Dict[str, Any] = {}
            # 文件名：xxx
            fn = re.search(r"文件名[：:]\s*([^，,]+)", inner)
            if fn:
                kb_meta["filename"] = fn.group(1).strip()
            # 第 N 页
            page = re.search(r"第\s*(\d+)\s*页", inner)
            if page:
                try:
                    kb_meta["page"] = int(page.group(1))
                except ValueError:
                    pass
            # 资料：/标题：xxx
            title = re.search(r"(?:资料|标题)[：:]\s*([^，,]+)", inner)
            if title:
                kb_meta["title"] = title.group(1).strip()
            # 剩余片段可作为 section
            section = inner
            for prefix in ["文件名", "第", "资料", "标题"]:
                section = re.sub(rf"{prefix}[^，,]*[，,]?\s*", "", section)
            if section.strip():
                kb_meta["section"] = section.strip()
            citation = {
                "display_text": display_text,
                "span": {"start": start, "end": end},
                "source_type": source_type,
                "kb": kb_meta,
            }
        citations.append(citation)
    return citations


def _map_qa_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    从规范化的 /api/qa 请求体中提取 Agent 所需的可选参数。
    """
    def clamp_float(v, lo, hi, default):
        try:
            x = float(v)
            return max(lo, min(hi, x))
        except (TypeError, ValueError):
            return default

    def clamp_int(v, lo, hi, default):
        try:
            x = int(v)
            return max(lo, min(hi, x))
        except (TypeError, ValueError):
            return default

    options = payload.get("options") or {}
    return {
        "use_web_search": bool(options.get("use_web_search", True)),
        "temperature": clamp_float(options.get("temperature"), 0.0, 1.0, 0.3),
        "kb_top_k": clamp_int(options.get("kb_top_k"), 1, 20, 3),
        "web_max_results": clamp_int(options.get("web_max_results"), 1, 15, 5),
    }


@app.post("/api/chat")
async def chat_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    简单的 HTTP 接口：接收 {history, user_query, temperature?, use_web_search?, kb_top_k?, web_max_results?}，返回 {answer}。
    """
    user_query = str(payload.get("user_query", ""))
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    params = _parse_chat_params(payload)
    _debug_log(
        "chat_request",
        {
            "user_query_preview": user_query[:50],
            "history_len": len(history),
            **params,
        },
    )
    answer = _agent.answer(history=history, user_query=user_query, **params)
    return {"answer": answer}


def _stream_qa_answer(history: list, user_query: str, use_internal_kb: bool = True, **kwargs: Any):
    """同步生成器：yield SSE 行（data: {...}\n\n），包含 session_id 和 question_id。"""
    for event in _agent.answer_stream(history=history, user_query=user_query, use_internal_kb=use_internal_kb, **kwargs):
        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _parse_qa_stream_payload(payload: Dict[str, Any]) -> tuple:
    """
    解析 /api/qa 流式请求，返回 (session_id, model, question_id, user_query, history, use_internal_kb, opts)
    """
    session_id = str(payload.get("session_id") or "")
    model = str(payload.get("model") or "")

    question = payload.get("question") or {}
    question_id = str(question.get("id") or "")
    user_query = str(question.get("text") or "")

    context = payload.get("context") or {}
    history = context.get("history") or []
    if not isinstance(history, list):
        history = []

    options = payload.get("options") or {}
    use_internal_kb = bool(options.get("use_internal_kb", True))
    opts = _map_qa_options(payload)

    return session_id, model, question_id, user_query, history, use_internal_kb, opts


@app.post("/api/qa")
async def qa_endpoint(payload: Dict[str, Any]):
    """
    规范化 JSON 接口：面向前端和其他后端服务。

    请求结构：
    {
      "session_id": "0000-0000-0000-0004",
      "model": "gpt-5-mini",
      "prompt": { "key": "xxxxx", "overrides": { ... } },
      "question": {
        "id": "q0000-0000-0000-0003",
        "text": "为什么 xxxxxx",
        "attachments": [ ... ]
      },
      "options": {
        "use_internal_kb": true,
        "use_web_search": true,
        "temperature": 0.3,
        "kb_top_k": 3,
        "web_max_results": 5,
        "stream": false
      },
      "context": { "history": [ ... ] }
    }

    当 options.stream = true 时，返回 SSE 流式响应。
    """
    import asyncio
    import queue
    from concurrent.futures import ThreadPoolExecutor

    options = payload.get("options") or {}
    is_stream = bool(options.get("stream", False))

    session_id, model, question_id, user_query, history, use_internal_kb, opts = _parse_qa_stream_payload(payload)

    _debug_log(
        "qa_request",
        {
            "session_id": session_id,
            "question_id": question_id,
            "user_query_preview": user_query[:50],
            "history_len": len(history),
            "use_internal_kb": use_internal_kb,
            "stream": is_stream,
            **opts,
        },
    )

    # 流式响应
    if is_stream:
        loop = asyncio.get_event_loop()
        q: queue.Queue = queue.Queue()
        executor = ThreadPoolExecutor(max_workers=2)

        def produce():
            for event in _agent.answer_stream(history=history, user_query=user_query, use_internal_kb=use_internal_kb, **opts):
                # 在事件中添加 session_id 和 question_id
                event["session_id"] = session_id
                event["question_id"] = question_id
                q.put("data: " + json.dumps(event, ensure_ascii=False) + "\n\n")
            q.put(None)

        async def event_generator() -> AsyncIterator[str]:
            loop.run_in_executor(executor, produce)
            while True:
                chunk = await loop.run_in_executor(executor, q.get)
                if chunk is None:
                    break
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 非流式响应
    answer_text = _agent.answer(
        history=history,
        user_query=user_query,
        use_internal_kb=use_internal_kb,
        **opts,
    )

    response: Dict[str, Any] = {
        "session_id": session_id,
        "question_id": question_id,
        "answer": {
            "id": "",
            "model": model or _settings.llm.model,
            "text": answer_text,
            "citations": _extract_citations(answer_text),
        },
        "usage": None,
        "trace": None,
    }
    return response


@app.post("/api/chat/image")
async def chat_image_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    图片问答接口：接收 {history, user_query, image_b64, temperature?, use_web_search?, kb_top_k?, web_max_results?}，返回 {answer}。
    image_b64 为去掉 data:image/...;base64, 前缀后的 base64 字符串。
    """
    user_query = str(payload.get("user_query", ""))
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    image_b64 = str(payload.get("image_b64", "") or "")
    params = _parse_chat_params(payload)
    _debug_log(
        "chat_image_request",
        {
            "user_query_preview": user_query[:50],
            "history_len": len(history),
            "has_image": bool(image_b64),
            **params,
        },
    )
    if not image_b64:
        # 没有图片时退回到普通文本接口的行为
        answer = _agent.answer(history=history, user_query=user_query, **params)
    else:
        answer = _agent.answer_with_image(
            history=history,
            user_query=user_query,
            image_b64=image_b64,
            **params,
        )
    return {"answer": answer}


def _stream_answer(history: list, user_query: str, **kwargs: Any):
    """同步生成器：yield SSE 行（data: {...}\\n\\n）。"""
    for event in _agent.answer_stream(history=history, user_query=user_query, **kwargs):
        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


@app.post("/api/chat/stream")
async def chat_stream_endpoint(payload: Dict[str, Any]) -> StreamingResponse:
    """
    流式接口：接收 {history, user_query, temperature?, use_web_search?, kb_top_k?, web_max_results?}，
    返回 text/event-stream。
    """
    import asyncio
    import queue
    from concurrent.futures import ThreadPoolExecutor

    user_query = str(payload.get("user_query", ""))
    history = payload.get("history") or []
    if not isinstance(history, list):
        history = []
    params = _parse_chat_params(payload)

    _debug_log(
        "chat_stream_request",
        {"user_query_preview": user_query[:50], "history_len": len(history), **params},
    )

    loop = asyncio.get_event_loop()
    q: queue.Queue = queue.Queue()
    executor = ThreadPoolExecutor(max_workers=2)

    def produce():
        for s in _stream_answer(history=history, user_query=user_query, **params):
            q.put(s)
        q.put(None)

    async def event_generator() -> AsyncIterator[str]:
        loop.run_in_executor(executor, produce)
        while True:
            chunk = await loop.run_in_executor(executor, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def index() -> FileResponse:
    """
    返回前端单页应用。
    """
    index_path = _frontend_dir / "index.html"
    _debug_log(
        "index_request",
        {
            "frontend_dir_exists": _frontend_dir.exists(),
            "index_path": str(index_path),
            "index_exists": index_path.exists(),
        },
    )
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    return FileResponse(str(index_path))


def main() -> None:
    import uvicorn

    _debug_log(
        "server_start",
        {
            "host": "0.0.0.0",
            "port": 8000,
            "project_root": str(_project_root),
        },
    )

    uvicorn.run(
        "qa_agent.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

