"""DeepAgent CLI Interface.

LangGraph dev server 없이 DeepAgent 코딩 어시스턴트를 테스트하고 사용하기 위한
대화형 커맨드라인 인터페이스입니다.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError, GraphInterrupt

if TYPE_CHECKING:
    from langgraph.pregel import Pregel

# 환경 변수 로드
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 상대 import를 위한 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_agent import (
    model,
    CODING_ASSISTANT_PROMPT,
    analyze_impact,
    execute_python_code,
    run_pytest,
    change_project_directory,
    speed_analyzer_agent,
    precision_analyzer_agent,
    code_refactor_agent,
    test_generator_agent,
    doc_sync_agent,
    file_summarizer_agent,
)

# =============================================================================
# 설정 클래스
# =============================================================================


@dataclass
class CLIConfig:
    """CLI 설정을 관리하는 데이터 클래스."""

    exit_commands: Set[str] = field(default_factory=lambda: {"exit", "quit", "q"})
    model: str = field(default_factory=lambda: os.environ.get("MODEL", "moonshotai/kimi-k2-0905"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").upper())
    thread_id: str = "cli-session"
    separator_length: int = 70
    max_messages: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0


# =============================================================================
# 로깅 설정
# =============================================================================


def setup_logging(log_level: str) -> logging.Logger:
    """로깅을 설정하고 로거를 반환합니다.

    Args:
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        설정된 로거 인스턴스
    """
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


# =============================================================================
# 헬퍼 함수
# =============================================================================


def extract_final_response(messages: List[BaseMessage]) -> str:
    """메시지 히스토리에서 가장 최근 AI 응답을 추출합니다.

    역순으로 메시지를 순회하여 첫 번째 AIMessage를 찾고, content가 문자열이면
    그대로 반환하고, 리스트 형태면 텍스트 부분만 추출하여 결합합니다.

    Args:
        messages: LangChain 메시지 리스트

    Returns:
        최종 AI 응답 텍스트, 또는 찾을 수 없는 경우 기본 메시지
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # tool_calls가 있는 경우는 건너뛰고 다음 메시지를 찾음
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                continue

            # content가 문자열인 경우
            if isinstance(msg.content, str):
                # 빈 문자열이 아니면 반환
                if msg.content.strip():
                    return msg.content
                # 빈 문자열이면 다음 메시지 확인
                continue
            # content가 리스트인 경우
            elif isinstance(msg.content, list):
                text_parts = []
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                result = "\n".join(text_parts) if text_parts else ""
                if result.strip():
                    return result

    return "[응답을 찾을 수 없습니다.]"


def handle_exit(reason: str = "종료") -> None:
    """종료 메시지를 출력합니다.

    Args:
        reason: 종료 이유
    """
    print(f"\n\n👋 {reason}. 안녕히 가세요!")


def handle_agent_error(
    error: Exception, context: str = "agent execution", logger: Optional[logging.Logger] = None
) -> None:
    """에이전트 실행 오류를 처리하고 로깅합니다.

    Args:
        error: 발생한 예외
        context: 오류가 발생한 컨텍스트
        logger: 로거 인스턴스
    """
    error_type = type(error).__name__

    if isinstance(error, GraphRecursionError):
        if logger:
            logger.error(f"Graph recursion limit reached during {context}: {error}")
        print(f"\n❌ 작업이 너무 복잡합니다. 단계를 나누어 다시 시도해주세요.")
    elif isinstance(error, GraphInterrupt):
        if logger:
            logger.error(f"Graph interrupt during {context}: {error}")
        print(f"\n❌ 작업이 중단되었습니다. 승인이 필요한 작업입니다.")
    elif isinstance(error, TimeoutError):
        if logger:
            logger.error(f"Timeout during {context}: {error}")
        print(f"\n❌ 시간 초과: 작업이 너무 오래 걸렸습니다.")
    elif isinstance(error, ConnectionError):
        if logger:
            logger.error(f"Connection error during {context}: {error}")
        print(f"\n❌ 연결 오류: API 서버에 연결할 수 없습니다.")
    else:
        if logger:
            logger.error(f"Error during {context}: {error}", exc_info=True)
        print(f"\n❌ Error ({error_type}): {str(error)}")

    print("다시 시도하거나 질문을 다시 작성해주세요.")


def print_welcome_banner(config: CLIConfig) -> None:
    """환영 배너를 출력합니다.

    Args:
        config: CLI 설정
    """
    sep = "=" * config.separator_length
    print(sep)
    print("DeepAgent Coding Assistant - Interactive CLI")
    print(sep)
    print(f"Model: {config.model}")
    print(f"Log Level: {config.log_level}")
    print("\n이 에이전트는 요청을 전문 서브에이전트에게 지능적으로 라우팅합니다.")
    print("사용 가능한 서브에이전트:")
    print("  • speed-analyzer: 빠른 정적 분석 (AST 기반)")
    print("  • precision-analyzer: 정밀 LSP 분석 (Pyright)")
    print("  • code-refactor: 코드 리팩토링 및 자가 복구")
    print("  • test-generator: pytest 테스트 자동 생성")
    print("  • doc-sync: 문서 동기화 (Docstring, README)")
    print("  • file-summarizer: 대용량 파일 요약")
    print("  • general-purpose: 기타 일반 작업")
    exit_cmds = "', '".join(config.exit_commands)
    print(f"\n세션을 종료하려면 '{exit_cmds}'를 입력하세요.")
    print(sep)
    print()


def trim_message_history(messages: List[BaseMessage], max_messages: int) -> List[BaseMessage]:
    """메시지 히스토리를 제한하여 메모리를 관리합니다.

    Args:
        messages: 메시지 리스트
        max_messages: 보존할 최대 메시지 수

    Returns:
        트림된 메시지 리스트 (초기 시스템 프롬프트 + 최근 메시지)
    """
    if len(messages) <= max_messages:
        return messages

    # 초기 시스템 프롬프트(처음 5개) 보존 + 최근 메시지 유지
    return list(messages[:5]) + list(messages[-(max_messages - 5) :])


# =============================================================================
# REPL 클래스
# =============================================================================


class CLIRepl:
    """대화형 CLI REPL을 관리하는 클래스."""

    def __init__(self, config: Optional[CLIConfig] = None):
        """REPL 초기화.

        Args:
            config: CLI 설정. None이면 기본 설정 사용.
        """
        self.config = config or CLIConfig()
        self.logger = setup_logging(self.config.log_level)
        self.local_app: Any = None  # CompiledGraph (Pregel)
        self.running = False

    def setup_agent(self) -> None:
        """DeepAgent와 FilesystemBackend를 초기화합니다."""
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from deepagents.middleware.filesystem import _get_filesystem_tools
        import coding_agent

        coding_agent._filesystem_backend = FilesystemBackend(
            root_dir=os.getcwd(),
            virtual_mode=True,  # coding_agent.py와 일관성 유지
        )

        # 파일시스템 도구 생성 (ls, read_file, write_file, edit_file, glob, grep, delete_file)
        fs_tools = _get_filesystem_tools(coding_agent._filesystem_backend)

        checkpointer = MemorySaver()

        self.local_app = create_deep_agent(
            model=model,
            tools=[
                *fs_tools,  # 파일시스템 도구 추가
                analyze_impact,
                execute_python_code,
                run_pytest,
                change_project_directory,
            ],
            system_prompt=CODING_ASSISTANT_PROMPT,
            backend=coding_agent._filesystem_backend,
            subagents=[  # type: ignore[arg-type]
                speed_analyzer_agent,
                precision_analyzer_agent,
                code_refactor_agent,
                test_generator_agent,
                doc_sync_agent,
                file_summarizer_agent,
            ],
            checkpointer=checkpointer,
            interrupt_on={
                "analyze_impact": False,
                "read_file": False,
                "write_file": False,
                "edit_file": False,
                "delete_file": True,  # 파일 삭제만 사용자 승인 필요
                "change_project_directory": False,
            },
        )

        self.logger.info("Agent initialized successfully")

    def process_user_input(self, user_input: str, config: RunnableConfig) -> None:
        """사용자 입력을 처리하고 에이전트 응답을 출력합니다.

        Args:
            user_input: 사용자 입력 텍스트
            config: LangGraph RunnableConfig (thread_id 포함)
        """
        state = {"messages": [HumanMessage(content=user_input)]}

        print("\n🤖 Assistant: ", end="", flush=True)

        try:
            step_count = 0

            # stream_mode='updates'로 변경: 각 단계의 변경사항만 수신 (메타데이터 전송 최소화)
            for chunk in self.local_app.stream(state, config=config, stream_mode="updates"):
                step_count += 1

                # chunk는 상태 업데이트 (변경사항만 포함, 전체 상태 아님)
                if chunk and isinstance(chunk, dict):
                    # 디버그 로깅 개선: 업데이트 키만 표시
                    if self.logger.isEnabledFor(logging.DEBUG) and step_count > 1:
                        update_keys = list(chunk.keys())
                        self.logger.debug(f"Step {step_count}: Update keys={update_keys}")

                        # 메시지 업데이트가 있으면 간단한 미리보기 표시
                        if "messages" in chunk:
                            messages = chunk["messages"]
                            if isinstance(messages, list) and messages:
                                last_msg = messages[-1]
                                msg_type = type(last_msg).__name__
                                msg_preview = ""
                                if isinstance(last_msg, AIMessage) and isinstance(
                                    last_msg.content, str
                                ):
                                    msg_preview = (
                                        f": {last_msg.content[:50]}..."
                                        if len(last_msg.content) > 50
                                        else f": {last_msg.content}"
                                    )
                                self.logger.debug(f"  └─ {msg_type}{msg_preview}")

            self.logger.info(f"Agent execution completed in {step_count} steps")

            # stream_mode="updates" 사용 시 최종 상태는 체크포인터에서 가져옴
            final_state = self.local_app.get_state(config)

            if final_state and final_state.values:
                messages = final_state.values.get("messages", [])

                # 메시지 히스토리 트림 (메모리 관리)
                if len(messages) > self.config.max_messages:
                    self.logger.warning(
                        f"Message history ({len(messages)}) exceeds limit ({self.config.max_messages}), trimming..."
                    )
                    # 트림된 메시지로 상태 업데이트
                    trimmed_messages = trim_message_history(messages, self.config.max_messages)
                    # 다음 실행을 위해 트림된 상태 사용
                    messages = trimmed_messages

                response = extract_final_response(messages)  # type: ignore[arg-type]

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Final state has {len(messages)} messages")

                print(response)
            else:
                print("[에이전트 실행이 완료되지 않았습니다.]")

        except (GraphRecursionError, GraphInterrupt, TimeoutError, ConnectionError) as e:
            handle_agent_error(e, "agent execution", self.logger)
        except Exception as e:
            handle_agent_error(e, "agent execution", self.logger)

    def run(self) -> None:
        """REPL 메인 루프를 실행합니다."""
        print_welcome_banner(self.config)

        self.setup_agent()

        config: RunnableConfig = {"configurable": {"thread_id": self.config.thread_id}}

        self.running = True

        while self.running:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in self.config.exit_commands:
                    handle_exit()
                    break

                self.process_user_input(user_input, config)

            except KeyboardInterrupt:
                handle_exit("중단")
                break
            except EOFError:
                handle_exit("EOF 감지")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in REPL: {e}", exc_info=True)
                print(f"\n❌ Unexpected error: {str(e)}")

    def cleanup(self) -> None:
        """리소스를 정리합니다."""
        self.running = False
        self.logger.info("REPL cleanup completed")


# =============================================================================
# 메인 엔트리 포인트
# =============================================================================


def main() -> int:
    """CLI 애플리케이션의 메인 엔트리 포인트.

    Returns:
        종료 코드 (성공 시 0, 오류 시 1)
    """
    try:
        config = CLIConfig()
        repl = CLIRepl(config)
        repl.run()
        repl.cleanup()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    # 직접 실행 시 CLI 인터페이스를 시작합니다
    sys.exit(main())
