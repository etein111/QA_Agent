from __future__ import annotations

from typing import List, Dict

from rich.console import Console
from rich.markdown import Markdown

from .agent import QAAgent


console = Console()


def chat_loop() -> None:
    agent = QAAgent()
    history: List[Dict[str, str]] = []

    console.print("[bold green]高等数学学习问答 Agent[/bold green] （输入 `exit` 退出）")

    while True:
        try:
            user_input = console.input("[bold cyan]你：[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见，祝学习顺利！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            try:
                from langfuse import get_client

                get_client().flush()
            except Exception:  # noqa: S110
                pass
            console.print("再见，祝学习顺利！")
            break

        history.append({"role": "user", "content": user_input})

        console.print("[bold yellow]Agent 正在思考，请稍候...[/bold yellow]")
        try:
            answer = agent.answer(history=history, user_query=user_input)
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]发生错误：[/bold red]{e}")
            continue

        history.append({"role": "assistant", "content": answer})
        console.print(Markdown(answer))

        # 确保 Langfuse 收到本轮 trace（短生命周期应用需主动 flush）
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:  # noqa: S110
            pass


def main() -> None:
    chat_loop()


if __name__ == "__main__":
    main()

