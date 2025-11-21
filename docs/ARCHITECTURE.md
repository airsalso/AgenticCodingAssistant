# DeepAgents Architecture Overview

이 프로젝트는 `create_deep_agent`로 컴파일되는 LangGraph 그래프를 중심으로 합니다. 감독 에이전트는 기본 미들웨어 스택(Todo, Filesystem, SubAgent, Summarization, Anthropic Prompt Cache, PatchToolCalls, 필요 시 Human-in-the-Loop)을 거쳐 계획/파일/서브에이전트 도구를 제공하며, `recursion_limit` 1000으로 보호됩니다. 파일 작업은 플러그형 `BackendProtocol` 구현(State/Filesystem/Store/Composite)에 위임됩니다.

## Diagram (Excalidraw)
- **파일**: [`docs/architecture.excalidraw`](./architecture.excalidraw)
- **열기**: <https://excalidraw.com> → "Open" → 파일 업로드 후 편집/내보내기
- **구성**:
  - 사용자 요청이 감독 에이전트로 들어오면 LangGraph 그래프가 미들웨어를 통해 계획 작성(`write_todos`), 파일 작업(`ls`/`read`/`write`/`edit`/`glob`/`grep`), 서브에이전트 호출(`task`), 사용자 정의 도구를 실행합니다.
  - FilesystemMiddleware는 선택된 백엔드를 호출하고, SubAgentMiddleware는 동일한 기본 미들웨어와 모델을 공유하는 서브그래프를 스폰합니다.
  - SummarizationMiddleware와 AnthropicPromptCachingMiddleware가 긴 대화와 비용을 관리하고, PatchToolCallsMiddleware와 HumanInTheLoopMiddleware가 거버넌스/안전성을 보완합니다.

## Key References
- `src/deepagents/graph.py`의 `create_deep_agent`가 기본 모델/미들웨어/도구를 조립하고 LangGraph 에이전트를 반환합니다.
- `src/deepagents/backends/protocol.py`는 파일 작업을 수행하는 `BackendProtocol`과 결과 타입(`WriteResult`, `EditResult`)을 정의합니다.
