# 命令行入口：uv run python -m qa_agent.preprocess [--data-dir ...] [--no-use-existing]
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_for_data_dir


def main() -> None:
    p = argparse.ArgumentParser(description="预处理 data 下 ppt/pdf：MinerU 解析并提取 Segment，写出到 data/preprocessed/")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="data 目录路径，默认取项目根下的 data",
    )
    p.add_argument(
        "--no-use-existing",
        action="store_true",
        help="不使用已有 MinerU 输出，强制重新调 API",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_root = args.data_dir or root / "data"
    if not data_root.exists():
        print(f"[WARN] data 目录不存在: {data_root}，跳过")
        return

    paths = run_for_data_dir(
        data_root=data_root,
        use_existing_mineru=not args.no_use_existing,
        project_root=root,
    )
    print(f"[OK] 预处理完成，共 {len(paths)} 个文件")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
