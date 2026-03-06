from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Iterator, Tuple

from .config import get_settings
from .llm_client import LLMClient
from .tools import KnowledgeBase, search_web, RetrievedChunk, WebSearchResult


def _split_query_for_multi_retrieval(query: str) -> List[str]:
    """
    将用户问题拆成多个子查询，便于多概念时分别命中不同文档。
    例如：「自然谬误和柠檬市场是什么」 -> ['自然谬误', '柠檬市场是什么']
    """
    if not (query or "").strip():
        return [query]
    # 按常见并列连接符拆分，保留每段前后空白 strip 后长度至少为 1
    parts = re.split(r"[和与、,，]", query)
    parts = [p.strip() for p in parts if len(p.strip()) >= 1]
    if len(parts) >= 2:
        return parts
    return [query]


SYSTEM_PROMPT_BASE = """
你是一个专门帮助学生回答问题的对话 Agent。

要求：
1. 尽量根据引用的内容组织回答，回答要尽量循序渐进、先直观解释再给出严格结论。
2. 你始终可以利用三类上下文：历史对话、内部知识库检索结果、联网搜索结果。
3. 当某一句话或某一段内容引用了“内部知识库”的信息时，请在这句/这段的末尾内联附上引用标记，
   形式为：`[引用: 文件名：xxx.pdf，第 N 页，章节名]`。
4. 当某一句话或某一段内容明显引用了“联网搜索”的结果时，请在这句/这段的末尾内联附上引用标记，
   形式为：`[引用: 来源：网页标题, https://example.com/...]`。
5. 引用应紧跟在对应句子或段落的结尾处，而不是统一集中到答案末尾。
6. 可以同时综合内部资料、联网搜索以及对话历史进行回答；如信息来源不清晰，可以不加引用。
7. 如果不确定，必须说明自己的不确定性，避免编造精确数字或定理编号。
8. 回答的字数控制在300字以内，格式采用markdown格式，条理清晰
"""


class QAAgent:
    """
    答疑对话 Agent。

    - 工具 1：内部知识库（ PPT / PDF + metadata）
    - 工具 2：联网搜索
    - 外挂知识库：由调用方直接传入的对话历史 messages
    """

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._kb = KnowledgeBase()
        self._retrieve_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qa_retrieve")

    def _retrieve_parallel(
        self,
        user_query: str,
        use_internal_kb: bool,
        use_web_search: bool,
        kb_top_k: int = 3,
        web_max_results: int = 5,
    ) -> Tuple[List[RetrievedChunk], List[WebSearchResult]]:
        """
        执行知识库检索，并预留联网搜索占位。
        具体的联网搜索由上游 LLM 自带的联网能力完成，此处不再主动调用外部搜索 API。
        """
        retrieved_chunks: List[RetrievedChunk] = []
        web_results: List[WebSearchResult] = []

        def do_kb() -> List[RetrievedChunk]:
            if not use_internal_kb:
                return []
            sub_queries = _split_query_for_multi_retrieval(user_query)
            if len(sub_queries) >= 2:
                return self._kb.search_multi(sub_queries, top_k=kb_top_k)
            return self._kb.search(user_query, top_k=kb_top_k)

        def do_web() -> List[WebSearchResult]:
            # 联网搜索交由上游 gpt-5-mini 自身的联网能力完成，这里不再调用本地 web_search
            return []

        if use_internal_kb and use_web_search:
            f_kb = self._retrieve_executor.submit(do_kb)
            f_web = self._retrieve_executor.submit(do_web)
            retrieved_chunks = f_kb.result()
            web_results = f_web.result()
        elif use_internal_kb:
            retrieved_chunks = do_kb()
        elif use_web_search:
            web_results = do_web()

        return retrieved_chunks, web_results

    def answer(
        self,
        history: List[Dict[str, str]],
        user_query: str,
        use_internal_kb: bool = True,
        use_web_search: bool = True,
        temperature: float = 0.3,
        kb_top_k: int = 3,
        web_max_results: int = 5,
    ) -> str:
        """
        :param history: 之前的对话记录（只需要 role + content），由外部“外挂知识库”直接传入。
        :param user_query: 当前用户问题。
        :param temperature: LLM 采样温度，0～1。
        :param kb_top_k: 知识库检索条数。
        :param web_max_results: 联网搜索条数。
        """
        retrieved_chunks, web_results = self._retrieve_parallel(
            user_query, use_internal_kb, use_web_search,
            kb_top_k=kb_top_k, web_max_results=web_max_results,
        )

        # 将检索结果整理成“工具上下文”，拼接到 system 提示之后
        tool_context_lines: List[str] = []

        if retrieved_chunks:
            tool_context_lines.append("【内部知识库检索结果（PPT / PDF / 数据】")
            for i, ch in enumerate(retrieved_chunks, start=1):
                tool_context_lines.append(
                    f"{i}. 文档：{ch.title}（{ch.filename}），"
                    f"位置：第 {ch.page} 页，{ch.section}。"
                    f"摘要：{ch.snippet}"
                )

        if web_results:
            tool_context_lines.append("【联网搜索结果摘要（DuckDuckGo）】")
            for i, r in enumerate(web_results, start=1):
                tool_context_lines.append(
                    f"{i}. 标题：{r.title}\n   URL：{r.url}\n   摘要：{r.snippet}"
                )

        tool_context = "\n".join(tool_context_lines) if tool_context_lines else ""

        # 构造 messages：先附加“外挂知识库”对话历史，再附加工具上下文，最后是当前提问
        messages: List[Dict[str, str]] = []
        messages.extend(history)

        if tool_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "下面是你可以使用的检索结果（内部知识库 + 联网搜索），"
                        "请在回答时尽量充分引用合适的信息源，并在回答中带上清晰的【引用】标记：\n"
                        + tool_context
                    ),
                }
            )

        messages.append({"role": "user", "content": user_query})

        settings = get_settings()
        trace_name = settings.langfuse.trace_name
        use_langfuse = settings.langfuse.enabled and bool(settings.langfuse.secret_key)

        # 根据前端是否开启联网搜索，动态拼接 system_prompt
        if use_web_search:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮你需要使用自身具备的联网搜索/浏览器能力，主动检索最新资料，并在引用处标明来源。"
            )
        else:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮禁止使用任何联网搜索或浏览器能力，请仅根据对话历史和内部知识库作答，不要假设或编造外部网页内容。"
            )

        # 根据前端是否开启联网搜索，动态拼接 system_prompt
        if use_web_search:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮你需要使用自身具备的联网搜索/浏览器能力，主动检索最新资料，并在引用处标明来源。"
            )
        else:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮禁止使用任何联网搜索或浏览器能力，请仅根据对话历史和内部知识库作答，不要假设或编造外部网页内容。"
            )

        # 根据前端是否开启联网搜索，动态拼接 system_prompt
        if auto_use_web:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮你需要使用自身具备的联网搜索/浏览器能力，主动检索最新资料，并在引用处标明来源。"
            )
        else:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮禁止使用任何联网搜索或浏览器能力，请仅根据对话历史、图片和内部知识库作答，不要假设或编造外部网页内容。"
            )

        if use_langfuse:
            from langfuse import get_client

            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="span",
                name=trace_name,
                input={"user_query": user_query},
            ):
                completion = self._llm.chat(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                )
        else:
            completion = self._llm.chat(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            )

        content = completion.choices[0].message.content or ""

        # 若知识库完全未命中，要求在回答末尾明确提示；根据是否允许联网动态调整文案
        if use_internal_kb and not retrieved_chunks:
            if use_web_search:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识与联网搜索。）"
            else:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识推理。）"
            content = (content or "").rstrip() + tail_notice

        # 在模型输出的基础上，为召回的每条 chunk 补一条引用，避免多文档时只显示 top1 的引用
        paragraphs = [p for p in content.split("\n\n")]

        def _find_para_without_citation(paras: List[str], start: int) -> int:
            if not paras:
                return 0
            for i in range(len(paras)):
                idx = (start + i) % len(paras)
                if "[引用:" not in paras[idx]:
                    return idx
            return min(start, len(paras) - 1)

        def _append_citation_to_paragraph(
            paras: List[str],
            citation_text: str,
            prefer_index: int,
        ) -> List[str]:
            if not paras:
                return paras
            idx = _find_para_without_citation(paras, prefer_index)
            if paras[idx].strip().endswith("]"):
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            else:
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            return paras

        # 内部知识库：为每条召回的 chunk 补一条引用（不按 doc_id 去重），保证 log 里看到的多个文档/多段都会在回答中兜底引用
        para_start = 1
        for ch in retrieved_chunks:
            raw = ch.format_citation().strip()
            inner = raw.strip("【】")
            kb_citation = f"[引用: {inner}]"
            paragraphs = _append_citation_to_paragraph(paragraphs, kb_citation, prefer_index=para_start)
            para_start += 1

        # 联网搜索：选 top1，从第 2 段起找空位，不在开头加引用
        if web_results:
            r0 = web_results[0]
            web_citation = f"[引用: 来源：{r0.title}, {r0.url}]"
            target_idx = _find_para_without_citation(paragraphs, 1)
            paragraphs = _append_citation_to_paragraph(paragraphs, web_citation, prefer_index=target_idx)

        return "\n\n".join(paragraphs)

    def answer_stream(
        self,
        history: List[Dict[str, str]],
        user_query: str,
        use_internal_kb: bool = True,
        use_web_search: bool = True,
        temperature: float = 0.3,
        kb_top_k: int = 3,
        web_max_results: int = 5,
    ) -> Iterator[Dict[str, Any]]:
        """
        与 answer() 相同的上下文与检索，但以流式方式 yield 回复片段。
        先 yield {"content": "..."} 的增量，最后 yield {"done": True, "full_content": "..."}（含引用补全）。
        """
        retrieved_chunks, web_results = self._retrieve_parallel(
            user_query, use_internal_kb, use_web_search,
            kb_top_k=kb_top_k, web_max_results=web_max_results,
        )

        tool_context_lines: List[str] = []
        if retrieved_chunks:
            tool_context_lines.append("【内部知识库检索结果（PPT / PDF / 数据】")
            for i, ch in enumerate(retrieved_chunks, start=1):
                tool_context_lines.append(
                    f"{i}. 文档：{ch.title}（{ch.filename}），"
                    f"位置：第 {ch.page} 页，{ch.section}。"
                    f"摘要：{ch.snippet}"
                )
        if web_results:
            tool_context_lines.append("【联网搜索结果摘要（DuckDuckGo）】")
            for i, r in enumerate(web_results, start=1):
                tool_context_lines.append(
                    f"{i}. 标题：{r.title}\n   URL：{r.url}\n   摘要：{r.snippet}"
                )

        tool_context = "\n".join(tool_context_lines) if tool_context_lines else ""
        messages: List[Dict[str, str]] = []
        messages.extend(history)
        if tool_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "下面是你可以使用的检索结果（内部知识库 + 联网搜索），"
                        "请在回答时尽量引用合适的信息源，并在回答中带上清晰的【引用】标记：\n"
                        + tool_context
                    ),
                }
            )
        messages.append({"role": "user", "content": user_query})

        content_parts: List[str] = []
        settings = get_settings()
        use_langfuse = settings.langfuse.enabled and bool(settings.langfuse.secret_key)

        # 根据前端是否开启联网搜索，动态拼接 system_prompt（与 answer 保持一致）
        if use_web_search:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮你需要使用自身具备的联网搜索/浏览器能力，主动检索最新资料，并在引用处标明来源。"
            )
        else:
            system_prompt = (
                SYSTEM_PROMPT_BASE
                + "\n9. 本轮禁止使用任何联网搜索或浏览器能力，请仅根据对话历史和内部知识库作答，不要假设或编造外部网页内容。"
            )

        if use_langfuse:
            from langfuse import get_client

            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="span",
                name=settings.langfuse.trace_name,
                input={"user_query": user_query},
            ):
                for chunk in self._llm.chat_stream(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                ):
                    content_parts.append(chunk)
                    yield {"content": chunk}
        else:
            for chunk in self._llm.chat_stream(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            ):
                content_parts.append(chunk)
                yield {"content": chunk}

        content = "".join(content_parts)

        # 若知识库完全未命中，要求在回答末尾明确提示；根据是否允许联网动态调整文案
        if use_internal_kb and not retrieved_chunks:
            if use_web_search:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识与联网搜索。）"
            else:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识推理。）"
            content = (content or "").rstrip() + tail_notice

        paragraphs = [p for p in content.split("\n\n")]

        def _find_para_without_citation(paras: List[str], start: int) -> int:
            if not paras:
                return 0
            for i in range(len(paras)):
                idx = (start + i) % len(paras)
                if "[引用:" not in paras[idx]:
                    return idx
            return min(start, len(paras) - 1)

        def _append_citation_to_paragraph(
            paras: List[str],
            citation_text: str,
            prefer_index: int,
        ) -> List[str]:
            if not paras:
                return paras
            idx = _find_para_without_citation(paras, prefer_index)
            if paras[idx].strip().endswith("]"):
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            else:
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            return paras

        para_start = 1
        for ch in retrieved_chunks:
            raw = ch.format_citation().strip()
            inner = raw.strip("【】")
            kb_citation = f"[引用: {inner}]"
            paragraphs = _append_citation_to_paragraph(paragraphs, kb_citation, prefer_index=para_start)
            para_start += 1
        if web_results:
            r0 = web_results[0]
            web_citation = f"[引用: 来源：{r0.title}, {r0.url}]"
            target_idx = _find_para_without_citation(paragraphs, 1)
            paragraphs = _append_citation_to_paragraph(paragraphs, web_citation, prefer_index=target_idx)

        full_content = "\n\n".join(paragraphs)
        yield {"done": True, "full_content": full_content}

    def answer_with_image(
        self,
        history: List[Dict[str, str]],
        user_query: str,
        image_b64: str,
        use_internal_kb: bool = True,
        use_web_search: bool = True,
        temperature: float = 0.3,
        kb_top_k: int = 3,
        web_max_results: int = 5,
    ) -> str:
        """
        支持图片输入的回答接口：在现有检索与引用逻辑基础上，将图片以多模态形式一并发送给 LLM。
        image_b64: 去掉 data:image/...;base64, 前缀后的 base64 字符串。
        """
        # 纯图片问题时自动关闭联网搜索，避免用空 query 访问 DuckDuckGo
        auto_use_web = use_web_search and bool((user_query or "").strip())
        retrieved_chunks, web_results = self._retrieve_parallel(
            user_query,
            use_internal_kb,
            auto_use_web,
            kb_top_k=kb_top_k,
            web_max_results=web_max_results,
        )

        tool_context_lines: List[str] = []
        if retrieved_chunks:
            tool_context_lines.append("【内部知识库检索结果（PPT / PDF / 假数据，仅作示例）】")
            for i, ch in enumerate(retrieved_chunks, start=1):
                tool_context_lines.append(
                    f"{i}. 文档：{ch.title}（{ch.filename}），"
                    f"位置：第 {ch.page} 页，{ch.section}。"
                    f"摘要：{ch.snippet}"
                )
        if web_results:
            tool_context_lines.append("【联网搜索结果摘要（DuckDuckGo）】")
            for i, r in enumerate(web_results, start=1):
                tool_context_lines.append(
                    f"{i}. 标题：{r.title}\n   URL：{r.url}\n   摘要：{r.snippet}"
                )

        tool_context = "\n".join(tool_context_lines) if tool_context_lines else ""

        messages: List[Dict[str, Any]] = []
        messages.extend(history)
        if tool_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "下面是你可以使用的检索结果（内部知识库 + 联网搜索），"
                        "请在回答时尽量充分引用合适的信息源，并在回答中带上清晰的【引用】标记：\n"
                        + tool_context
                    ),
                }
            )

        multimodal_content: List[Dict[str, Any]] = []
        if (user_query or "").strip():
            multimodal_content.append({"type": "text", "text": user_query})
        multimodal_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
        messages.append(
            {
                "role": "user",
                "content": multimodal_content,
            }
        )

        settings = get_settings()
        trace_name = settings.langfuse.trace_name
        use_langfuse = settings.langfuse.enabled and bool(settings.langfuse.secret_key)

        if use_langfuse:
            from langfuse import get_client

            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="span",
                name=trace_name,
                input={"user_query": user_query, "has_image": True},
            ):
                completion = self._llm.chat(
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                )
        else:
            completion = self._llm.chat(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
            )

        content = completion.choices[0].message.content or ""

        # 若知识库完全未命中，要求在回答末尾明确提示；根据是否允许联网动态调整文案
        if use_internal_kb and not retrieved_chunks:
            if use_web_search:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识、图片理解与联网搜索。）"
            else:
                tail_notice = "\n\n（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识与图片理解。）"
            content = (content or "").rstrip() + tail_notice

        # 复用与 answer 相同的引用补丁逻辑
        paragraphs = [p for p in content.split("\n\n")]

        def _find_para_without_citation(paras: List[str], start: int) -> int:
            if not paras:
                return 0
            for i in range(len(paras)):
                idx = (start + i) % len(paras)
                if "[引用:" not in paras[idx]:
                    return idx
            return min(start, len(paras) - 1)

        def _append_citation_to_paragraph(
            paras: List[str],
            citation_text: str,
            prefer_index: int,
        ) -> List[str]:
            if not paras:
                return paras
            idx = _find_para_without_citation(paras, prefer_index)
            if paras[idx].strip().endswith("]"):
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            else:
                paras[idx] = paras[idx].rstrip() + f" {citation_text}"
            return paras

        para_start = 1
        for ch in retrieved_chunks:
            raw = ch.format_citation().strip()
            inner = raw.strip("【】")
            kb_citation = f"[引用: {inner}]"
            paragraphs = _append_citation_to_paragraph(paragraphs, kb_citation, prefer_index=para_start)
            para_start += 1

        if web_results:
            r0 = web_results[0]
            web_citation = f"[引用: 来源：{r0.title}, {r0.url}]"
            target_idx = _find_para_without_citation(paragraphs, 1)
            paragraphs = _append_citation_to_paragraph(paragraphs, web_citation, prefer_index=target_idx)

        return "\n\n".join(paragraphs)


__all__ = ["QAAgent"]

