"""CLI Interface for DeepAgentic Code Assistant.

Interactive command-line interface for testing and using the coding assistant
without requiring langgraph dev server.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

# .env 파일 로드
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 상대 import를 위해 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coding_agent import agent

# =============================================================================
# 상수 정의
# =============================================================================

# 종료 명령어
EXIT_COMMANDS = {"exit", "quit", "q"}

# 환경 변수에서 모델 이름 가져오기
MODEL = os.environ.get("MODEL", "moonshotai/kimi-k2-0905")

# 로깅 레벨 설정 (환경 변수 기반)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# =============================================================================
# 로깅 설정
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 헬퍼 함수
# =============================================================================

def _extract_final_response(messages: List[BaseMessage]) -> str:
    """Extract the final AI response from the message history.

    Args:
        messages: List of LangChain messages from the agent execution.

    Returns:
        The content of the final AI message, or a default message if not found.
    """
    # 역순으로 순회하여 첫 번째 AIMessage 찾기
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            # content가 문자열이면 그대로 반환
            if isinstance(msg.content, str):
                return msg.content
            # content가 리스트면 텍스트 부분만 추출
            elif isinstance(msg.content, list):
                text_parts = []
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                return "\n".join(text_parts) if text_parts else ""

    return "[응답을 찾을 수 없습니다.]"


def _handle_exit(reason: str = "종료") -> None:
    """Handle graceful exit with a farewell message.

    Args:
        reason: Reason for exit (e.g., "중단", "EOF 감지").
    """
    print(f"\n\n👋 {reason}. 안녕히 가세요!")


def _handle_agent_error(error: Exception, context: str = "agent execution") -> None:
    """Handle and log agent execution errors.

    Args:
        error: The exception that occurred.
        context: Context where the error occurred.
    """
    error_type = type(error).__name__

    # 구체적인 에러 타입별 처리
    if isinstance(error, GraphRecursionError):
        logger.error(f"Graph recursion limit reached during {context}: {error}")
        print(f"\n❌ 작업이 너무 복잡합니다. 단계를 나누어 다시 시도해주세요.")
    elif isinstance(error, TimeoutError):
        logger.error(f"Timeout during {context}: {error}")
        print(f"\n❌ 시간 초과: 작업이 너무 오래 걸렸습니다.")
    elif isinstance(error, ConnectionError):
        logger.error(f"Connection error during {context}: {error}")
        print(f"\n❌ 연결 오류: API 서버에 연결할 수 없습니다.")
    else:
        logger.error(f"Error during {context}: {error}", exc_info=True)
        print(f"\n❌ Error ({error_type}): {str(error)}")

    print("다시 시도하거나 질문을 다시 작성해주세요.")


# =============================================================================
# REPL 구현
# =============================================================================

def _cli_repl() -> None:
    """Interactive command-line REPL for testing the DeepAgent.

    Provides a simple interface to interact with the agent locally without
    requiring langgraph dev. Type 'exit' or 'quit' to end the session.

    The agent will automatically route your requests to appropriate subagents
    based on the task complexity and type.
    """
    print("=" * 70)
    print("DeepAgent Coding Assistant - Interactive CLI")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Log Level: {LOG_LEVEL}")
    print("\n이 에이전트는 요청을 전문 서브에이전트에게 지능적으로 라우팅합니다.")
    print("사용 가능한 서브에이전트:")
    print("  • speed-analyzer: 빠른 정적 분석 (AST 기반)")
    print("  • precision-analyzer: 정밀 LSP 분석 (Pyright)")
    print("  • code-refactor: 코드 리팩토링 및 자가 복구")
    print("  • test-generator: pytest 테스트 자동 생성")
    print("  • doc-sync: 문서 동기화 (Docstring, README)")
    print("  • file-summarizer: 대용량 파일 요약")
    print("  • general-purpose: 기타 일반 작업")
    exit_cmds = "', '".join(EXIT_COMMANDS)
    print(f"\n세션을 종료하려면 '{exit_cmds}'를 입력하세요.")
    print("=" * 70)
    print()

    # Agent에 메모리 체크포인터 추가
    # 체크포인터는 컴파일 시점에 바인딩해야 하므로 coding_agent 모듈에서 직접 임포트
    from coding_agent import (
        model,
        CODING_ASSISTANT_PROMPT,
        analyze_impact,
        execute_python_code,
        run_pytest,
        delete_file,
        change_project_directory,
        speed_analyzer_agent,
        precision_analyzer_agent,
        code_refactor_agent,
        test_generator_agent,
        doc_sync_agent,
        file_summarizer_agent,
        workspace_root
    )
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    checkpointer = MemorySaver()

    # Backend 인스턴스 생성 및 전역 변수에 설정
    import coding_agent
    coding_agent._filesystem_backend = FilesystemBackend(
        root_dir=os.getcwd(),
        virtual_mode=False,
    )

    # 체크포인터를 포함하여 에이전트 재생성
    local_app = create_deep_agent(
        model=model,
        tools=[
            analyze_impact,
            execute_python_code,
            run_pytest,
            delete_file,
            change_project_directory,
        ],
        system_prompt=CODING_ASSISTANT_PROMPT,
        backend=coding_agent._filesystem_backend,
        subagents=[
            speed_analyzer_agent,
            precision_analyzer_agent,
            code_refactor_agent,
            test_generator_agent,
            doc_sync_agent,
            file_summarizer_agent,
        ],
        checkpointer=checkpointer,  # 메모리 체크포인터 추가
        interrupt_on={
            "analyze_impact": False,
            "read_file": False,
            "write_file": False,
            "edit_file": False,
            "delete_file": True,  # 파일 삭제는 사용자 승인 필요 (CLI에서는 불가)
            "change_project_directory": False,  # 디렉토리 변경은 자동 허용
        },
    )

    config = {"configurable": {"thread_id": "cli-session"}}

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in EXIT_COMMANDS:
                _handle_exit()
                break

            # Prepare state with user message
            state = {"messages": [HumanMessage(content=user_input)]}

            print("\n🤖 Assistant: ", end="", flush=True)

            # Stream the agent execution to see all steps
            try:
                final_state = None
                step_count = 0

                # Use streaming to process all steps with debug output
                for chunk in local_app.stream(state, config=config, stream_mode="values"):
                    step_count += 1
                    final_state = chunk

                    # Debug: Show progress
                    if step_count > 1:  # Don't show first step (just input)
                        messages = chunk.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            msg_type = type(last_msg).__name__
                            logger.debug(f"Step {step_count}: {msg_type}")

                logger.info(f"Agent execution completed in {step_count} steps")

                if final_state:
                    # Extract and display the final response
                    messages = final_state.get("messages", [])
                    response = _extract_final_response(messages)

                    # Debug: Show message count
                    logger.debug(f"Final state has {len(messages)} messages")

                    print(response)
                else:
                    print("[에이전트 실행이 완료되지 않았습니다.]")

            except (GraphRecursionError, TimeoutError, ConnectionError) as e:
                _handle_agent_error(e, "agent execution")
            except Exception as e:
                _handle_agent_error(e, "agent execution")

        except KeyboardInterrupt:
            _handle_exit("중단")
            break
        except EOFError:
            _handle_exit("EOF 감지")
            break
        except Exception as e:
            logger.error(f"Unexpected error in REPL: {e}", exc_info=True)
            print(f"\n❌ Unexpected error: {str(e)}")


def main() -> int:
    """Main entry point for the CLI application.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        _cli_repl()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    # Run the CLI interface when executed directly
    sys.exit(main())
