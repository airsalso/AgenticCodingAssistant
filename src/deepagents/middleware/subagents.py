"""`task` 도구를 통해 서브에이전트를 제공하는 미들웨어 구현."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypedDict, cast
from typing_extensions import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command


class SubAgent(TypedDict):
    """서브에이전트 구성을 표현하는 스펙 딕셔너리.

    커스텀 서브에이전트를 정의하면 `SubAgentMiddleware`의 `default_middleware`
    목록이 먼저 적용되고, 이어서 이 스펙에 지정한 `middleware`가 추가된다.
    기본 미들웨어 없이 전부 직접 지정하려면 `SubAgentMiddleware` 생성 시
    `default_middleware=[]`를 전달하면 된다.
    """

    name: str
    """서브에이전트의 이름."""

    description: str
    """메인 에이전트가 참고하는 서브에이전트 설명."""

    system_prompt: str
    """서브에이전트에게 적용할 시스템 프롬프트."""

    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    """서브에이전트가 사용할 도구 목록."""

    model: NotRequired[str | BaseChatModel]
    """사용할 모델 또는 모델 이름. 지정하지 않으면 `default_model`을 사용한다."""

    middleware: NotRequired[list[AgentMiddleware]]
    """기본 미들웨어 이후에 추가할 미들웨어 리스트."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """도구별 인터럽트 설정 딕셔너리."""


class CompiledSubAgent(TypedDict):
    """사전에 컴파일된 서브에이전트 스펙."""

    name: str
    """서브에이전트의 이름."""

    description: str
    """서브에이전트 설명."""

    runnable: Runnable
    """실제로 실행할 LangGraph `Runnable` 객체."""


DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# 서브에이전트로 상태를 전달할 때 제외해야 하는 키 목록
_EXCLUDED_STATE_KEYS = ("messages", "todos")

TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501


def _get_subagents(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> tuple[dict[str, Any], list[str]]:
    """서브에이전트 스펙을 바탕으로 실행 가능한 인스턴스를 생성한다.

    Args:
        default_model: 모델을 명시하지 않은 서브에이전트에 사용할 기본 모델.
        default_tools: 도구를 명시하지 않은 서브에이전트에 부여할 기본 도구.
        default_middleware: 모든 서브에이전트에 공통 적용할 미들웨어. `None`이면 생략.
        default_interrupt_on: 일반 목적 서브에이전트의 인터럽트 설정. 개별 서브에이전트가
            별도 설정을 제공하지 않으면 이 값을 사용한다.
        subagents: 서브에이전트 스펙 또는 이미 컴파일된 서브에이전트 목록.
        general_purpose_agent: 일반 목적 서브에이전트를 포함할지 여부.

    Returns:
        `(agent_dict, description_list)` 튜플. `agent_dict`는 이름을 runnable에 매핑하고,
        `description_list`는 사용자 안내용 설명 문자열을 담는다.
    """
    # None이면 빈 리스트로 대체하여 기본 미들웨어를 생략
    default_subagent_middleware = default_middleware or []

    agents: dict[str, Any] = {}
    subagent_descriptions = []

    # 일반 목적 서브에이전트가 활성화된 경우 생성
    if general_purpose_agent:
        general_purpose_middleware = [*default_subagent_middleware]
        if default_interrupt_on:
            general_purpose_middleware.append(HumanInTheLoopMiddleware(interrupt_on=default_interrupt_on))
        general_purpose_subagent = create_agent(
            default_model,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            tools=default_tools,
            middleware=general_purpose_middleware,
        )
        agents["general-purpose"] = general_purpose_subagent
        subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    # 사용자 정의 서브에이전트를 순회하며 생성
    for agent_ in subagents:
        subagent_descriptions.append(f"- {agent_['name']}: {agent_['description']}")
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = [*default_subagent_middleware, *agent_["middleware"]] if "middleware" in agent_ else [*default_subagent_middleware]

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        agents[agent_["name"]] = create_agent(
            subagent_model,
            system_prompt=agent_["system_prompt"],
            tools=_tools,
            middleware=_middleware,
        )
    return agents, subagent_descriptions


def _create_task_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    task_description: str | None = None,
) -> BaseTool:
    """서브에이전트를 호출하는 `task` 도구를 만든다.

    Args:
        default_model: 서브에이전트 기본 모델.
        default_tools: 서브에이전트 기본 도구 목록.
        default_middleware: 공통으로 적용할 미들웨어.
        default_interrupt_on: 일반 목적 서브에이전트의 기본 인터럽트 설정.
        subagents: 서브에이전트 스펙 목록.
        general_purpose_agent: 일반 목적 서브에이전트를 포함할지 여부.
        task_description: 도구 설명을 덮어쓸 문구. `None`이면 기본 템플릿을 사용하며
            `{available_agents}` 플레이스홀더를 지원한다.

    Returns:
        서브에이전트를 타입별로 호출할 수 있는 `StructuredTool`.
    """
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        """서브에이전트 결과를 LangGraph 상태 업데이트 명령으로 변환한다.

        Args:
            result: 서브에이전트 실행 결과 딕셔너리.
            tool_call_id: 메인 에이전트가 부여한 도구 호출 ID.

        Returns:
            LangGraph 상태를 갱신할 `Command` 객체.
        """
        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(result["messages"][-1].text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        """요청된 서브에이전트 타입을 검증하고 실행 상태를 구성한다."""
        if subagent_type not in subagent_graphs:
            msg = f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in subagent_graphs]}"
            raise ValueError(msg)
        subagent = subagent_graphs[subagent_type]
        # 원본 상태를 변형하지 않도록 별도의 상태 딕셔너리를 생성
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    # 사용자 정의 설명이 있으면 사용하고, 없으면 기본 템플릿을 사용
    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        # 커스텀 설명에 플레이스홀더가 있으면 서브에이전트 설명으로 치환
        task_description = task_description.format(available_agents=subagent_description_str)

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """동기 방식으로 서브에이전트를 실행한다."""
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        # LangGraph Runnable을 호출하여 서브에이전트 작업을 수행
        result = subagent.invoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        """비동기 방식으로 서브에이전트를 실행한다."""
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        # 비동기 실행 경로에서는 `ainvoke`를 사용해 결과를 기다린다
        result = await subagent.ainvoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


class SubAgentMiddleware(AgentMiddleware):
    """`task` 도구를 통해 서브에이전트를 호출할 수 있도록 해주는 미들웨어.

    이 미들웨어는 메인 에이전트에 `task` 도구를 추가하여 복잡하거나 컨텍스트가
    방대한 업무를 별도의 서브에이전트로 위임할 수 있게 한다. 서브에이전트는
    여러 단계의 작업을 자체적으로 수행한 뒤 간결한 결과만 메인 에이전트에
    반환하므로, 스레드 컨텍스트를 깔끔하게 유지할 수 있다.

    기본으로 제공되는 일반 목적 서브에이전트는 메인 에이전트와 동일한 도구
    세트를 갖추되, 독립된 컨텍스트에서 실행되어 토큰과 상태를 절약한다.

    Args:
        default_model: 서브에이전트가 사용할 기본 모델 또는 모델 설정.
        default_tools: 기본 일반 목적 서브에이전트에 부여할 도구 목록.
        default_middleware: 서브에이전트 전반에 공통으로 적용할 미들웨어. `None`이면 적용하지 않는다.
        default_interrupt_on: 기본 일반 목적 서브에이전트에 사용할 인터럽트 설정.
        subagents: 추가로 등록할 커스텀 서브에이전트 목록.
        system_prompt: 메인 에이전트 시스템 프롬프트를 완전히 대체할 문자열.
        general_purpose_agent: 일반 목적 서브에이전트를 포함할지 여부. 기본값은 `True`.
        task_description: `task` 도구 설명을 덮어쓸 문자열. 생략 시 기본 템플릿 사용.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        # 기본 옵션으로 사용 (추가 미들웨어 없음)
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    subagents=[],
                )
            ],
        )

        # 서브에이전트에 커스텀 미들웨어 추가
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    default_middleware=[TodoListMiddleware()],
                    subagents=[],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        """서브에이전트 미들웨어를 초기화한다."""
        super().__init__()
        self.system_prompt = system_prompt
        task_tool = _create_task_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """서브에이전트 사용법을 시스템 프롬프트에 추가한다."""
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """비동기 모델 호출에도 서브에이전트 안내문을 주입한다."""
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return await handler(request)
