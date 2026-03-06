from __future__ import annotations

from typing import List, Dict, Any, Iterator

from .config import get_settings, optional_http_client

_settings = get_settings()


def _openai_client_class():
    """已配置 Langfuse 时使用其包装的 OpenAI，否则使用标准 OpenAI。"""
    if _settings.langfuse.enabled and _settings.langfuse.secret_key:
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI

            return LangfuseOpenAI
        except ImportError:
            pass
    from openai import OpenAI

    return OpenAI


class LLMClient:
    """
    通过 aihub（OpenAI 兼容接口）调用 LLM 的统一封装；
    若配置了 Langfuse，则使用 langfuse.openai 的 OpenAI 以自动上报监控。
    """

    def __init__(self) -> None:
        settings = _settings.llm
        if not settings.api_key:
            raise RuntimeError(
                "未配置 AIHUB_API_KEY 或 OPENAI_API_KEY 环境变量，无法调用 LLM。"
            )

        http_client = optional_http_client(settings.timeout)

        self._client = _openai_client_class()(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            http_client=http_client,
        )
        self._model = settings.model

    def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | Dict[str, Any] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Any:
        """
        统一的对话接口。messages 是标准的 OpenAI ChatCompletion 格式：
        [{"role": "user"|"system"|"assistant"|"tool", "content": "..."}]
        """

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return self._client.chat.completions.create(**payload)

    def chat_stream(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        流式对话接口，逐块 yield 助手回复的 content 片段。
        """
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        stream = self._client.chat.completions.create(**payload)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content


__all__ = ["LLMClient"]

