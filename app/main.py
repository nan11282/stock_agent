"""
main.py — 入口
"""

from agent import Agent
import re

try:
    import readline  # 启用方向键/历史/行编辑（仅 Linux/macOS 终端）
except ImportError:
    pass

from runtime import build_default_llm

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def main():
    # CLI 是最直接的本地调试入口：同样走完整 Agent/ReAct/记忆/工具链路，
    # 因此这里的行为应与 Telegram 聊天保持一致。
    agent = Agent(llm=build_default_llm(), max_steps=20)

    print("=== A股投资助理 ===")
    print("reset → 清空对话   quit → 退出\n")

    while True:
        try:
            raw = input("Coco: ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        user_input = _CTRL_CHARS.sub("", raw).strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            # reset 只清空当前 CLI 会话上下文，长期记忆和数据库继续保留。
            agent.reset()
            continue

        response = agent.chat(user_input)
        print(f"\n助理: {response}\n")


if __name__ == "__main__":
    main()
