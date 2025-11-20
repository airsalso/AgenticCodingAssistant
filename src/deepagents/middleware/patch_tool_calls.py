"""메시지 히스토리에서 끊어진 도구 호출을 보정하는 미들웨어."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


class PatchToolCallsMiddleware(AgentMiddleware):
    """대응하는 도구 응답이 없는 호출을 감지해 보정하는 미들웨어."""

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """에이전트 실행 전에 AI 메시지의 누락된 도구 호출을 정리한다."""
        messages = state["messages"]
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # 메시지를 순회하면서 끊어진 도구 호출에 대한 대체 메시지를 삽입
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # 대응 메시지가 없으면 취소 안내 ToolMessage를 생성
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *patched_messages]}
