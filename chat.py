"""命令行聊天入口。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from rag_agent import build_agent


def message_text(message: AIMessage) -> str:
    """兼容不同模型返回的文本内容格式。"""
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in message.content
    )


def main() -> None:
    agent = build_agent()
    history: list[HumanMessage | AIMessage] = []
    print("Agentic RAG 已启动。输入 exit 或 quit 退出。")
    while True:
        question = input("\n你：").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.invoke({"messages": [*history, HumanMessage(content=question)]})
        answer = result["messages"][-1]
        print(f"\n助手：{message_text(answer)}")
        history.extend([HumanMessage(content=question), answer])


if __name__ == "__main__":
    main()
