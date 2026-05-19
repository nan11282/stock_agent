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
    print("reset → 沉淀摘要后清空对话   clear → 只清短期对话   quit → 退出\n")

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
            # reset 是会话结束信号：先沉淀摘要，再清空当前 CLI 会话上下文。
            agent.reset()
            continue
        if user_input.lower() == "clear":
            # clear 只丢弃短期上下文，不写入长期记忆。
            agent.clear()
            continue

        streamed = []

        def print_delta(delta: str):
            if not streamed:
                print("\n助理: ", end="", flush=True)
            streamed.append(delta)
            print(delta, end="", flush=True)

        response = agent.chat(user_input, on_text_delta=print_delta)
        if streamed:
            print("\n")
        else:
            print(f"\n助理: {response}\n")


if __name__ == "__main__":
    main()
