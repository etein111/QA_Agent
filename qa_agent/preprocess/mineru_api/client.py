from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .locate import MinerUOutputNotFound

# 让根目录下的 .env 生效，读取 MINERU_TOKEN / MINERU_BASE_URL 等
load_dotenv()


# 项目根：qa_agent/preprocess/mineru_api/client.py -> 上三级为 QA_Agent 根
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _abs_path(p: str | Path, root: Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


BASE_URL = os.getenv("MINERU_BASE_URL", "https://mineru.net").rstrip("/")
API_V4 = f"{BASE_URL}/api/v4"


class MinerUApiError(RuntimeError):
    pass


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "*/*",
    }


def _debug_log(hypothesis_id: str, message: str, data: Dict[str, Any]) -> None:
    """向 debug-6dec40.log 追加一行 NDJSON，用于本次调试会话。"""
    payload = {
        "sessionId": "6dec40",
        "runId": "preprocess",
        "hypothesisId": hypothesis_id,
        "location": "qa_agent/preprocess/mineru_api/client.py:_require_token",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = _project_root() / "debug-6dec40.log"
    # #region agent log
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败时静默忽略，避免影响正常流程
        pass
    # #endregion agent log


def _require_token(token: Optional[str] = None) -> str:
    if token is not None and token.strip():
        return token.strip()

    env_has_token = "MINERU_TOKEN" in os.environ
    t = os.getenv("MINERU_TOKEN", "").strip()

    # 记录一次调试日志：用于确认是否加载到了 .env，以及当前工作目录
    _debug_log(
        hypothesis_id="H1",
        message="Checking MINERU_TOKEN in _require_token",
        data={
            "env_has_token_key": env_has_token,
            "token_non_empty": bool(t),
            "cwd": os.getcwd(),
        },
    )

    if not t:
        raise MinerUApiError(
            "MINERU_TOKEN not found. Set in .env: MINERU_TOKEN=your_token\n"
            "Optionally MINERU_BASE_URL=https://mineru.net"
        )
    return t


def _post_apply_upload_urls(
    token: str,
    files: List[Path],
    model_version: str = "vlm",
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    url = f"{API_V4}/file-urls/batch"
    payload = {
        "files": [{"name": f.name, "data_id": f.stem} for f in files],
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }
    resp = requests.post(url, headers=_headers(token), json=payload, timeout=timeout_sec)
    if resp.status_code != 200:
        raise MinerUApiError(f"[MinerU] apply_upload_urls HTTP {resp.status_code}: {resp.text}")
    j = resp.json()
    if j.get("code") != 0:
        raise MinerUApiError(f"[MinerU] apply_upload_urls code={j.get('code')} msg={j.get('msg')}")
    return j["data"]


def _put_upload_files(files: List[Path], file_urls: List[str], timeout_sec: int = 600) -> None:
    if len(files) != len(file_urls):
        raise ValueError(f"files({len(files)}) != file_urls({len(file_urls)})")
    for fpath, put_url in zip(files, file_urls):
        with open(fpath, "rb") as f:
            r = requests.put(put_url, data=f, timeout=timeout_sec)
        if r.status_code not in (200, 201, 204):
            raise MinerUApiError(f"[MinerU] upload failed: {fpath.name} -> HTTP {r.status_code}: {r.text}")


def _get_batch_results(token: str, batch_id: str, timeout_sec: int = 60) -> Dict[str, Any]:
    url = f"{API_V4}/extract-results/batch/{batch_id}"
    resp = requests.get(url, headers=_headers(token), timeout=timeout_sec)
    if resp.status_code != 200:
        raise MinerUApiError(f"[MinerU] get_batch_results HTTP {resp.status_code}: {resp.text}")
    j = resp.json()
    if j.get("code") != 0:
        raise MinerUApiError(f"[MinerU] get_batch_results code={j.get('code')} msg={j.get('msg')}")
    return j["data"]


def _wait_batch_done(
    token: str,
    batch_id: str,
    poll_interval_sec: float = 3.0,
    timeout_sec: float = 1800.0,
) -> List[Dict[str, Any]]:
    t0 = time.time()
    while True:
        data = _get_batch_results(token, batch_id)
        results = data.get("extract_result", []) or []
        if results and all(r.get("state") in ("done", "failed") for r in results):
            return results
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"[MinerU] batch {batch_id} not finished in {timeout_sec}s")
        time.sleep(poll_interval_sec)


def _download_and_unzip(zip_url: str, out_dir: Path, timeout_sec: int = 600) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "mineru_result.zip"
    with requests.get(zip_url, stream=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def parse_files(
    input_files: List[str],
    out_root: str,
    project_root: Optional[str | Path] = None,
    model_version: str = "vlm",
    *,
    token: Optional[str] = None,
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
    poll_interval_sec: float = 3.0,
    timeout_sec: float = 1800.0,
) -> Dict[str, Any]:
    root = Path(project_root) if project_root else _project_root()
    token = _require_token(token)
    files = [_abs_path(p, root) for p in input_files]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"not found: {f}")
    out_root_p = _abs_path(out_root, root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    data = _post_apply_upload_urls(
        token=token,
        files=files,
        model_version=model_version,
        enable_formula=enable_formula,
        enable_table=enable_table,
        language=language,
    )
    batch_id = data["batch_id"]
    file_urls = data["file_urls"]
    _put_upload_files(files, file_urls)
    results = _wait_batch_done(
        token=token,
        batch_id=batch_id,
        poll_interval_sec=poll_interval_sec,
        timeout_sec=timeout_sec,
    )

    items: List[Dict[str, Any]] = []
    for r in results:
        file_name = r.get("file_name") or ""
        state = r.get("state") or "unknown"
        item: Dict[str, Any] = {
            "file": file_name,
            "state": state,
            "out_dir": None,
            "zip_url": None,
            "error": None,
            "raw": r,
        }
        if state == "done":
            zip_url = r.get("full_zip_url")
            item["zip_url"] = zip_url
            if not zip_url:
                item["error"] = "done but missing full_zip_url"
            else:
                stem = Path(file_name).stem if file_name else "mineru_file"
                out_dir = out_root_p / f"{stem}__mineru"
                _download_and_unzip(zip_url, out_dir)
                item["out_dir"] = str(out_dir)
        else:
            item["error"] = r.get("err_msg") or "unknown error"
        items.append(item)

    return {"batch_id": batch_id, "items": items}
