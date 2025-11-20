# DeepAgent 0.2 아키텍처 가이드

LangChain v1.0과 LangGraph v1.0 위에 구축된 `deepagents` 0.2 버전은 **하나의 지능형 감독 에이전트**와 **필요 시 스폰되는 서브에이전트**, **파일 시스템/검색 도구**, **풍부한 프롬프트**를 결합해 깊이 있는 작업 분해를 수행합니다. 이 문서는 로컬 포크의 구성을 최신 릴리스와 정렬하기 위한 아키텍처 개요입니다.

## 핵심 구성요소

- **`deepagents.graph.create_deep_agent`**
  - LangChain `create_agent` 프리빌트를 활용해 감독 에이전트를 컴파일합니다.
  - 기본 모델은 `ChatAnthropic(claude-sonnet-4-5-20250929)`이며, LangChain `BaseChatModel` 호환 인스턴스나 문자열로 교체할 수 있습니다.
  - 표준 미들웨어 스택:
    1. `TodoListMiddleware`: `write_todos` 계획 도구 제공
    2. `FilesystemMiddleware`: 가상/외부 파일 도구(`ls`, `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`)
    3. `SubAgentMiddleware`: `task` 도구로 서브에이전트 오케스트레이션
    4. `SummarizationMiddleware`: 장기 스레드에서 메시지를 요약
    5. `AnthropicPromptCachingMiddleware`: 프롬프트 캐싱 최적화
    6. `PatchToolCallsMiddleware`: dangling tool call 보정
    7. 옵션: `HumanInTheLoopMiddleware` (특정 도구 인터럽트)
  - `with_config({"recursion_limit": 1000})`로 무한 콜 보호.

- **`deepagents.middleware`**
  - `FilesystemMiddleware`: LangGraph 상태, LangChain ToolRuntime, 백엔드를 결합해 파일 관련 도구를 제공. `backend` 매개변수로 스토리지를 선택합니다.
  - `SubAgentMiddleware`: 기본 도구/모델/미들웨어를 공유하는 서브에이전트를 등록하며, 필요 시 `CompiledSubAgent`로 사전 컴파일된 LangGraph 런너를 연결할 수 있습니다.
  - `PatchToolCallsMiddleware`: 새 메시지가 이전 tool call 을 끊었을 때 자동으로 취소 메시지를 삽입합니다.

- **`deepagents.backends`**
  - `FilesystemBackend`: 실제 파일시스템 접근.
  - `StateBackend`: LangGraph 상태를 스토리지로 사용(기본 가상 파일시스템).
  - `StoreBackend`: LangGraph Store API 기반 장기 저장.
  - `CompositeBackend`: 다중 백엔드 라우팅(예: 특정 prefix는 디스크, 그 외는 상태).
  - 공통 프로토콜은 `BackendProtocol`에서 정의하며 `WriteResult`, `EditResult`로 결과/오류를 전달합니다.

- **`deepagents.middleware.subagents`**
  - `SubAgent` 스펙(TypedDict)으로 이름/설명/프롬프트/도구/모델/미들웨어/인터럽트를 정의.
  - `CompiledSubAgent`는 미리 컴파일된 LangGraph `Runnable`을 래핑.
  - `task` 도구 호출 시 서브에이전트가 독립 컨텍스트에서 실행되고 결과만 감독 에이전트에게 반환됩니다.

## 도구 & 백엔드 동작

- **Todo 리스트**: `write_todos` 호출 시 LangGraph 상태(`AgentState`)의 `todos` 키가 업데이트되며 `Command`로 메시지/상태를 동시에 갱신합니다.
- **파일 도구**:
  - `ls_info`, `glob_info`: `FileInfo` 구조를 반환해 파일 경로, 크기, 수정시간을 제공.
  - `read`: 라인 번호 포함 형식(`cat -n`)으로 텍스트를 반환하며 빈 파일은 경고 메시지 출력.
  - `write`/`edit`: `WriteResult`/`EditResult`에 성공/오류/상태 업데이트 정보를 담아 LangGraph 상태와 동기화합니다.
  - `grep`: `GrepMatch` 리스트로 결과를 구조화하여 후처리에 용이.
- **백엔드 선택**:
  - 기본 `StateBackend`는 LangGraph 체크포인터가 있을 때 자동으로 파일 상태를 보존.
  - 외부 스토리지를 사용하려면 `FilesystemBackend` 또는 사용자 정의 `BackendProtocol` 구현을 주입합니다.

## 인터럽트 & 거버넌스

- `create_deep_agent(interrupt_on={...})`에 도구별 설정을 전달하면 `HumanInTheLoopMiddleware`가 활성화되어 승인/편집/응답 플로우를 강제합니다.
- `SubAgentMiddleware`에서도 서브에이전트별 `interrupt_on`을 오버라이드할 수 있습니다.
- `PatchToolCallsMiddleware`는 미완료 tool call 로 인한 상태 손상을 방지합니다.

## 비동기 지원

0.2 버전부터 `async_create_deep_agent`는 제거되었으며, `create_deep_agent`가 동기/비동기 사용을 모두 지원합니다. MCP 및 기타 async 환경에서는 `await agent.ainvoke(...)` 또는 `agent.astream(...)` 패턴을 그대로 사용할 수 있습니다.

## 모범 사례

1. **프롬프트**: 내장 시스템 프롬프트는 광범위한 운영 규칙을 포함하지만, 도메인별 `system_prompt`를 반드시 추가하세요.
2. **서브에이전트**: 독립된 하위 작업은 `task` 도구로 위임해 컨텍스트와 토큰을 절약합니다.
3. **파일 편집**: `read_file`을 호출한 후 `edit_file`을 사용하고, 중복 매치를 피하기 위해 `replace_all` 또는 충분한 컨텍스트를 제공하세요.
4. **스토리지**: 장기 실행/감사를 위해 LangGraph Checkpointer + StoreBackend 조합을 고려하세요.
5. **인터럽트**: 민감한 외부 호출이나 쓰기 작업에는 `interrupt_on`으로 휴먼 인더 루프 정책을 추가하세요.

## 예시

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import SubAgentMiddleware

agent = create_deep_agent(
    tools=[...],
    system_prompt="당신은 보안 우선 개발 보조 에이전트입니다.",
    backend=FilesystemBackend(root_path="/workspace"),
    interrupt_on={"edit_file": True, "write_file": True},
    subagents=[
        {
            "name": "security-reviewer",
            "description": "코드 변경사항의 보안 영향 점검",
            "system_prompt": "보안 리뷰 체크리스트를 수행하고 위험을 요약하세요.",
            "tools": [],
        },
    ],
)
```

## 참고 자료

- DeepAgents 문서: <https://docs.langchain.com/oss/python/deepagents/overview>
- LangGraph Prebuilt 에이전트: <https://docs.langchain.com/oss/python/langgraph/reference/prebuilt>
- LangChain Tool 가이드: <https://docs.langchain.com/oss/python/langgraph/how-tos/tools>
- LangChain Anthropic 통합: <https://python.langchain.com/docs/integrations/chat/anthropic>

