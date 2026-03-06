from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import httpx

# 优先从项目根目录加载 .env，便于 uv run 时生效
load_dotenv()


def optional_http_client(timeout: float = 60.0) -> "httpx.Client | None":
    """
    当环境变量 QA_SSL_VERIFY 为 false/0/no 时，返回不校验 SSL 的 httpx.Client（仅开发/内网用）。
    否则返回 None，由 OpenAI 使用默认客户端。
    """
    if os.getenv("QA_SSL_VERIFY", "true").strip().lower() in ("false", "0", "no"):
        import httpx as _httpx
        return _httpx.Client(verify=False, timeout=timeout)
    return None


class LangfuseSettings(BaseModel):
    """Langfuse 监控配置，trace 名称从 .env 的 LANGFUSE_TRACE_NAME 读取。"""

    secret_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    base_url: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    )
    trace_name: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_TRACE_NAME", "qa-agent")
    )
    enabled: bool = Field(
        default_factory=lambda: os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )


class LLMSettings(BaseModel):
    """
    统一的 LLM / aihub 配置。

    默认假设 aihub 提供 OpenAI 兼容接口，可以根据需要调整环境变量名称。
    """

    base_url: str = Field(
        default_factory=lambda: os.getenv("AIHUB_BASE_URL", "https://api.openai.com/v1")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("AIHUB_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    model: str = Field(default_factory=lambda: os.getenv("QA_MODEL", "gpt-4.1-mini"))
    keyword_model: str = Field(default_factory=lambda: os.getenv("KEYWORD_MODEL", "gpt-4o-mini"))
    timeout: float = Field(default=60.0)


class AppSettings(BaseModel):
    """应用级配置（如后续要做 Web 服务等，可在此扩展）。"""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    project_root: str = Field(
        default_factory=lambda: os.getenv("PROJECT_ROOT", os.getcwd())
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()

