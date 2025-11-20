"""DeepAgents 그래프 빌더 모듈.

기본 투두 관리, 파일 시스템 조작, 서브에이전트 스폰 기능을 포함한 LangGraph
에이전트를 생성하는 도우미들을 제공한다.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendProtocol, BackendFactory
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """DeepAgents에서 기본으로 사용하는 언어 모델을 생성한다.

    Returns:
        Claude Sonnet 4 모델이 설정된 `ChatAnthropic` 인스턴스.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """딥 에이전트 그래프를 생성한다.

    기본적으로 투두 작성 도구(`write_todos`), 파일 조작 도구 6종
    (`write_file`, `ls`, `read_file`, `edit_file`, `glob_search`, `grep_search`),
    그리고 서브에이전트 호출 도구(`task`)를 포함한 LangGraph 그래프를 빌드한다.

    Args:
        model: 사용할 언어 모델. 값이 없으면 `get_default_model()` 결과를 사용한다.
        tools: 메인 에이전트에 부여할 외부 도구 목록.
        system_prompt: 메인 에이전트 시스템 프롬프트에 덧붙일 추가 지시문.
        middleware: 기본 미들웨어 체인 뒤에 붙일 사용자 정의 미들웨어 목록.
        subagents: `SubAgent` 또는 `CompiledSubAgent` 스펙 리스트. 비어 있으면
            기본 일반 목적 서브에이전트만 활성화된다.
        response_format: LangChain 구조화 응답 포맷 객체.
        context_schema: LangGraph 상태 스키마로 사용될 타입.
        checkpointer: LangGraph 체크포인터 인스턴스. 상태 내구성을 확보할 때 사용한다.
        store: 장기 저장소 구현. `StoreBackend` 사용 시 필수.
        backend: 파일 입출력을 처리할 백엔드 또는 백엔드 팩토리.
        interrupt_on: 도구 이름별 인터럽트 설정 딕셔너리. 값이 있으면 HITL 미들웨어가 추가된다.
        debug: LangChain `create_agent` 호출 시 전달할 디버그 모드 플래그.
        name: 에이전트 식별용 이름.
        cache: LangGraph 실행 캐시 구현.

    Returns:
        LangGraph에서 실행 가능한 `CompiledStateGraph` 인스턴스.
    """
    if model is None:
        model = get_default_model()

    deepagent_middleware = [
        # 기본 계획 수립을 위한 TODO 미들웨어
        TodoListMiddleware(),
        # 파일 시스템 도구 묶음 제공
        FilesystemMiddleware(backend=backend),
        # 서브에이전트 스폰 및 관리 기능
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                # 서브에이전트에게도 동일한 기본 기능 부여
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        # 대화가 길어질 때 컨텍스트를 유지하기 위한 요약
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        # Anthropic 프롬프트 캐시로 비용/지연 최소화
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        # 끊어진 도구 호출을 자동으로 보정
        PatchToolCallsMiddleware(),
    ]
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    if middleware is not None:
        deepagent_middleware.extend(middleware)

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
