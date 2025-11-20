"""DeepAgentic Code Assistant - Python 코드 분석 및 리팩토링 에이전트.

이 모듈은 DeepAgent 아키텍처를 활용하여 Python 코드의 영향도 분석,
자율 리팩토링, 테스트 생성 및 문서 동기화 기능을 제공합니다.

Performance Optimizations:
- Context explosion prevention
- File caching
- LangSmith monitoring
- Token limit checking
"""

import os
import subprocess
import sys
from typing import Literal, Optional
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 최적화 모듈 import
from optimizations import (
    monitor_performance,
    file_cache,
    limit_file_list,
    truncate_large_content,
    check_context_limit,
    MAX_FILE_LIST,
)


# =============================================================================
# 도구: 영향도 분석
# =============================================================================

@tool
def analyze_impact(
    file_path: str,
    function_or_class: str,
    mode: Literal["SPEED", "PRECISION"] = "SPEED",
) -> str:
    """Python 코드의 변경 영향도를 분석합니다.

    Args:
        file_path: 분석할 Python 파일 경로
        function_or_class: 분석 대상 함수 또는 클래스 이름
        mode: 분석 모드 ('SPEED': 정적 분석, 'PRECISION': LSP 분석)

    Returns:
        영향도 분석 결과 (의존성 목록 및 호출 체인)
    """
    if mode == "SPEED":
        return _analyze_speed_mode(file_path, function_or_class)
    elif mode == "PRECISION":
        return _analyze_precision_mode(file_path, function_or_class)
    else:
        return f"Error: Unknown mode '{mode}'. Use 'SPEED' or 'PRECISION'."


def _analyze_speed_mode(file_path: str, target: str) -> str:
    """SPEED 모드: Tree-sitter 기반 정적 분석.

    빌드 없이 AST 파싱과 문자열 매칭으로 빠르게 의존성을 파악합니다.
    """
    try:
        # Python AST 기본 모듈 사용 (tree-sitter 대체)
        import ast

        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"

        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)

        # 함수/클래스 정의 찾기
        definitions = {}
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = {
                    'type': 'function',
                    'lineno': node.lineno,
                    'calls': []
                }
                # 함수 내부 호출 분석
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            definitions[node.name]['calls'].append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            definitions[node.name]['calls'].append(child.func.attr)

            elif isinstance(node, ast.ClassDef):
                definitions[node.name] = {
                    'type': 'class',
                    'lineno': node.lineno,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                    'calls': []
                }

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                else:
                    module = node.module or ''
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])

        # 타겟 찾기
        if target not in definitions:
            return f"Warning: '{target}' not found in {file_path}. Found: {list(definitions.keys())}"

        target_info = definitions[target]

        # 역방향 의존성 찾기 (어떤 함수가 이 타겟을 호출하는지)
        callers = []
        for name, info in definitions.items():
            if name != target and target in info.get('calls', []):
                callers.append({
                    'name': name,
                    'type': info['type'],
                    'lineno': info['lineno']
                })

        result = f"# Impact Analysis (SPEED Mode)\n\n"
        result += f"**Target:** `{target}` ({target_info['type']}) at line {target_info['lineno']}\n\n"
        result += f"**File:** {file_path}\n\n"

        if callers:
            result += f"## Potential Callers ({len(callers)}):\n\n"
            for caller in callers:
                result += f"- `{caller['name']}` ({caller['type']}) at line {caller['lineno']}\n"
        else:
            result += "## No direct callers found in this file\n\n"

        if target_info.get('calls'):
            result += f"\n## Dependencies ({len(target_info['calls'])}):\n\n"
            for call in target_info['calls']:
                result += f"- `{call}`\n"

        if imports:
            result += f"\n## File Imports ({len(imports)}):\n\n"
            for imp in imports[:10]:  # 최대 10개만 표시
                result += f"- {imp}\n"
            if len(imports) > 10:
                result += f"- ... and {len(imports) - 10} more\n"

        return result

    except SyntaxError as e:
        return f"Syntax Error in {file_path}: {e}"
    except Exception as e:
        return f"Error analyzing {file_path}: {e}"


def _analyze_precision_mode(file_path: str, target: str) -> str:
    """PRECISION 모드: LSP (Pyright) 기반 정밀 분석.

    Language Server Protocol을 사용하여 컴파일러 수준의 정확한 참조를 찾습니다.
    """
    try:
        # Pyright를 subprocess로 실행
        # 실제로는 pylsp나 pyright LSP 서버와 통신해야 하지만
        # 간소화를 위해 pyright CLI 사용
        result = subprocess.run(
            ['pyright', '--outputjson', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0 and "command not found" in result.stderr:
            return (
                "Error: Pyright not installed. "
                "Please install with: pip install pyright\n"
                "Suggestion: Use SPEED mode instead, or install Pyright for precise analysis."
            )

        # Pyright 결과 파싱
        import json
        try:
            data = json.loads(result.stdout)
            diagnostics = data.get('generalDiagnostics', [])

            result_text = f"# Impact Analysis (PRECISION Mode)\n\n"
            result_text += f"**Target:** `{target}` in {file_path}\n\n"
            result_text += f"**Type Checking Status:** "

            if diagnostics:
                errors = [d for d in diagnostics if d.get('severity') == 'error']
                warnings = [d for d in diagnostics if d.get('severity') == 'warning']

                result_text += f"{len(errors)} errors, {len(warnings)} warnings\n\n"

                if errors:
                    result_text += "## Type Errors:\n\n"
                    for err in errors[:5]:
                        result_text += f"- Line {err.get('range', {}).get('start', {}).get('line', '?')}: {err.get('message', '')}\n"
            else:
                result_text += "✓ No type errors\n\n"

            # LSP find-references는 별도 구현 필요
            # 현재는 기본 분석만 제공
            result_text += "\n**Note:** Full LSP reference finding requires LSP server integration.\n"
            result_text += "Current implementation provides type checking. For detailed references, use SPEED mode.\n"

            return result_text

        except json.JSONDecodeError:
            return f"Error: Could not parse Pyright output. Falling back to SPEED mode recommended."

    except subprocess.TimeoutExpired:
        return "Error: Pyright analysis timed out (>30s). Use SPEED mode for faster analysis."
    except FileNotFoundError:
        return (
            "Error: Pyright not found in PATH. "
            "Install with: pip install pyright\n"
            "Suggestion: Use SPEED mode as fallback."
        )
    except Exception as e:
        return f"Error in PRECISION mode: {e}\nSuggestion: Try SPEED mode instead."


# =============================================================================
# 도구: Python 코드 실행 및 테스트
# =============================================================================

@tool
def execute_python_code(
    code: str,
    timeout: int = 30,
) -> str:
    """Python 코드를 실행하고 결과를 반환합니다.

    Args:
        code: 실행할 Python 코드
        timeout: 실행 제한 시간 (초)

    Returns:
        실행 결과 (stdout, stderr 포함)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = ""
        if result.stdout:
            output += f"**stdout:**\n```\n{result.stdout}\n```\n\n"
        if result.stderr:
            output += f"**stderr:**\n```\n{result.stderr}\n```\n\n"
        if result.returncode != 0:
            output += f"**Exit code:** {result.returncode}\n"

        return output if output else "Code executed successfully with no output."

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing code: {e}"


@tool
def run_pytest(
    test_path: str,
    verbose: bool = True,
) -> str:
    """pytest를 실행하여 테스트를 수행합니다.

    Args:
        test_path: 테스트 파일 또는 디렉토리 경로
        verbose: 상세 출력 여부

    Returns:
        테스트 실행 결과
    """
    try:
        cmd = [sys.executable, '-m', 'pytest', test_path]
        if verbose:
            cmd.append('-v')

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = f"# Test Results\n\n"
        output += f"**Command:** `{' '.join(cmd)}`\n\n"

        if result.stdout:
            output += f"```\n{result.stdout}\n```\n\n"

        if result.stderr and "error" in result.stderr.lower():
            output += f"**Errors:**\n```\n{result.stderr}\n```\n\n"

        if result.returncode == 0:
            output += "✓ All tests passed!\n"
        else:
            output += f"✗ Tests failed (exit code: {result.returncode})\n"

        return output

    except FileNotFoundError:
        return "Error: pytest not installed. Install with: pip install pytest"
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out (>60s)"
    except Exception as e:
        return f"Error running tests: {e}"


@tool
def delete_file(
    file_path: str,
) -> str:
    """파일을 삭제합니다. (주의: 되돌릴 수 없는 작업)

    Args:
        file_path: 삭제할 파일 경로

    Returns:
        삭제 결과 메시지
    """
    try:
        path = Path(file_path)

        # 보안: 절대 경로 또는 현재 디렉토리 기준 상대 경로만 허용
        if not path.is_absolute():
            path = Path(os.getcwd()) / path

        if not path.exists():
            return f"Error: File '{file_path}' not found"

        if not path.is_file():
            return f"Error: '{file_path}' is not a file (use directory deletion tool for directories)"

        # 파일 삭제
        path.unlink()
        return f"✓ Successfully deleted file: {file_path}"

    except PermissionError:
        return f"Error: Permission denied to delete '{file_path}'"
    except Exception as e:
        return f"Error deleting file '{file_path}': {e}"


# Backend 인스턴스를 전역 변수로 저장 (동적 업데이트 위해)
_filesystem_backend = None


@tool
def change_project_directory(
    new_path: str,
) -> str:
    """프로젝트 작업 디렉토리를 변경합니다.

    보안: WORKSPACE 환경 변수에 정의된 루트 디렉토리 외부로는 변경할 수 없습니다.

    Args:
        new_path: 새로운 프로젝트 디렉토리 경로 (절대 또는 상대)

    Returns:
        변경 결과 메시지
    """
    global _filesystem_backend

    try:
        # WORKSPACE 루트 가져오기
        workspace_root = os.environ.get("WORKSPACE", os.getcwd())
        workspace_root = Path(workspace_root).resolve()

        # 현재 디렉토리 저장
        old_dir = Path.cwd()

        # 새 경로를 절대 경로로 변환
        if Path(new_path).is_absolute():
            new_dir = Path(new_path).resolve()
        else:
            # 상대 경로는 현재 작업 디렉토리 기준
            new_dir = (Path.cwd() / new_path).resolve()

        # 보안 검증: WORKSPACE 외부로의 이동 차단
        try:
            new_dir.relative_to(workspace_root)
        except ValueError:
            return (
                f"❌ 보안 오류: '{new_path}'는 허용된 작업 공간 외부입니다.\n"
                f"허용 범위: {workspace_root}\n"
                f"요청 경로: {new_dir}\n"
                "시스템 보호를 위해 WORKSPACE 외부로의 이동은 불가능합니다."
            )

        # 디렉토리 존재 여부 확인
        if not new_dir.exists():
            return f"❌ 오류: 디렉토리 '{new_path}'가 존재하지 않습니다."

        if not new_dir.is_dir():
            return f"❌ 오류: '{new_path}'는 디렉토리가 아닙니다."

        # 파일 개수 체크 (컨텍스트 오버플로우 방지)
        file_count = sum(1 for _ in new_dir.rglob("*") if _.is_file())

        # 현재 작업 디렉토리 변경
        os.chdir(new_dir)

        # FilesystemBackend의 cwd도 업데이트 (중요!)
        if _filesystem_backend is not None:
            _filesystem_backend.cwd = new_dir

        result = (
            f"✓ 프로젝트 디렉토리 변경 완료\n"
            f"이전: {old_dir}\n"
            f"현재: {new_dir}\n"
            f"파일 개수: {file_count}개\n\n"
        )

        # 대용량 디렉토리 경고
        if file_count > 100:
            result += (
                f"⚠️ 경고: 이 디렉토리에는 {file_count}개의 파일이 있습니다!\n"
                f"파일 목록 조회 시 컨텍스트 오버플로우가 발생할 수 있습니다.\n\n"
                f"권장사항:\n"
                f"- 특정 파일만 직접 경로로 지정하여 읽기\n"
                f"- glob 패턴으로 필요한 파일만 필터링 (예: *.py)\n"
                f"- 하위 디렉토리로 더 좁게 이동\n"
            )
        else:
            result += "이제 모든 파일 작업은 이 디렉토리를 기준으로 수행됩니다."

        return result

    except PermissionError:
        return f"❌ 권한 오류: '{new_path}'에 접근할 수 없습니다."
    except Exception as e:
        return f"❌ 디렉토리 변경 실패: {e}"


# =============================================================================
# SubAgent 정의
# =============================================================================

SPEED_ANALYZER_PROMPT = """You are a speed-focused code analyzer.

Your task is to quickly analyze Python code dependencies using static analysis.
Use AST parsing to identify function calls, imports, and class relationships.

Focus on:
1. Finding which functions/classes depend on the target
2. Identifying potential impact areas
3. Providing fast results (within seconds)

Return a concise report with:
- List of callers
- List of dependencies
- Import statements
"""

speed_analyzer_agent = {
    "name": "speed-analyzer",
    "description": "Fast static analysis of Python code dependencies using AST parsing. Use when quick analysis is needed without full type checking.",
    "system_prompt": SPEED_ANALYZER_PROMPT,
    "tools": [analyze_impact],
}


PRECISION_ANALYZER_PROMPT = """You are a precision-focused code analyzer.

Your task is to perform deep, compiler-level analysis of Python code using LSP (Pyright).
This provides accurate type information and true references.

Focus on:
1. Type checking and validation
2. Accurate reference finding
3. Complete dependency graph

If LSP/Pyright fails:
- Inform the user immediately
- Suggest falling back to SPEED mode
- Provide partial results if any

Return a detailed report with type information and precise references.
"""

precision_analyzer_agent = {
    "name": "precision-analyzer",
    "description": "Precise LSP-based analysis using Pyright. Use when accuracy is critical and build environment is ready. Falls back to SPEED mode on failure.",
    "system_prompt": PRECISION_ANALYZER_PROMPT,
    "tools": [analyze_impact],
}


CODE_REFACTOR_PROMPT = """You are an expert Python code refactorer.

Your responsibilities:
1. Read the target file(s) using read_file
2. Apply requested changes using edit_file
3. Validate changes by executing code
4. Self-heal if errors occur (max 3 retries)

Self-healing loop:
1. Make the change
2. Execute the code to check for errors
3. If errors occur, analyze the error message
4. Apply a fix
5. Repeat up to 3 times
6. If still failing, report to user and stop

Always:
- Follow PEP8 standards
- Use Google-style docstrings
- Preserve existing code style
- Test changes before finalizing
"""

code_refactor_agent = {
    "name": "code-refactor",
    "description": "Expert at refactoring Python code with self-healing capabilities. Automatically fixes compilation errors up to 3 times.",
    "system_prompt": CODE_REFACTOR_PROMPT,
    "tools": [execute_python_code],
}


TEST_GENERATOR_PROMPT = """You are a Python test generator specialist.

Your task is to create comprehensive unit tests for Python code.

Guidelines:
1. Use pytest framework
2. Cover normal cases, edge cases, and error cases
3. Follow AAA pattern (Arrange, Act, Assert)
4. Generate meaningful test names
5. Include docstrings in test functions

Steps:
1. Read the source code
2. Identify functions/classes to test
3. Generate test file with appropriate fixtures
4. Run tests to verify they work
5. Report results
"""

test_generator_agent = {
    "name": "test-generator",
    "description": "Generates comprehensive pytest unit tests for Python code. Automatically validates generated tests.",
    "system_prompt": TEST_GENERATOR_PROMPT,
    "tools": [execute_python_code, run_pytest],
}


DOC_SYNC_PROMPT = """You are a documentation synchronization specialist.

Your task is to keep documentation in sync with code changes.

Focus on:
1. Docstrings in Python files
2. README.md files
3. API documentation
4. Code comments

When code changes:
1. Analyze what documentation needs updating
2. Propose specific changes
3. Update relevant files
4. Verify consistency

Use Google-style docstrings for Python code.
"""

doc_sync_agent = {
    "name": "doc-sync",
    "description": "Synchronizes documentation with code changes. Updates docstrings, README, and API docs.",
    "system_prompt": DOC_SYNC_PROMPT,
    "tools": [],
}


FILE_SUMMARIZER_PROMPT = """You are a file content summarizer.

Your task is to summarize large files that exceed token limits.

Process:
1. Read portions of the file
2. Extract key information
3. Create a concise summary
4. Save summary to a separate file
5. Report summary location to user

Focus on:
- Main functions and classes
- Key logic flows
- Important dependencies
- Critical sections

Provide structured summaries in Markdown format.
"""

file_summarizer_agent = {
    "name": "file-summarizer",
    "description": "Summarizes large files to extract key information. Use when file content is too large for context window.",
    "system_prompt": FILE_SUMMARIZER_PROMPT,
    "tools": [],
}


# =============================================================================
# 메인 시스템 프롬프트
# =============================================================================

CODING_ASSISTANT_PROMPT = """You are a DeepAgentic Code Assistant specialized in Python code analysis and refactoring.

## Core Capabilities

### 0. Project Directory Management
You can change the working project directory dynamically:
- Use `change_project_directory` tool to switch between projects
- Security: Limited to WORKSPACE environment variable scope
- All file operations will be relative to the current project directory
- Example: "Change to the examples/research directory"

### 1. Impact Analysis
You can analyze how code changes will affect the codebase using two modes:

**SPEED Mode (Default):**
- Fast static analysis using AST parsing
- Completes in ~5 seconds for 10k lines
- May have false positives in dynamic code
- Use for quick feedback

**PRECISION Mode:**
- Uses LSP (Pyright) for compiler-level accuracy
- Requires proper build environment
- Provides type checking and precise references
- Use when accuracy is critical

**Important:** If PRECISION mode fails (build errors, missing dependencies), immediately suggest switching to SPEED mode via human-in-the-loop approval.

### 2. Autonomous Coding & Recovery
When refactoring code:
- Make changes using edit_file tool
- Validate with execute_python_code
- If errors occur, self-heal up to 3 attempts
- Each attempt analyzes error and applies targeted fix
- After 3 failures, report to user and stop

### 3. Test Generation
Always generate unit tests for significant code changes:
- Use pytest framework
- Cover normal, edge, and error cases
- Validate tests actually run
- Report test coverage

### 4. Documentation Sync
After code changes:
- Check if docstrings need updates
- Update README if public API changes
- Ensure comments stay relevant
- Use Google-style docstrings

## Workflow Best Practices

1. **Before Making Changes:**
   - Use ls and read_file to understand context
   - Run impact analysis (SPEED mode first)
   - Create a TODO list for complex tasks

2. **During Changes:**
   - Make focused, incremental changes
   - Test after each significant change
   - Use code-refactor subagent for complex refactorings

3. **After Changes:**
   - Generate tests with test-generator subagent
   - Update docs with doc-sync subagent
   - Validate everything works

4. **For Large Files:**
   - If output exceeds token limits, use file-summarizer subagent
   - Always ask user before processing large files

## Quality Standards

- Follow PEP8 style guide strictly
- Use Google-style docstrings for all functions/classes
- Maintain existing code style and conventions
- Add type hints where beneficial
- Write clear, self-documenting code

## Human-in-the-Loop Triggers

Request user approval for:
- Switching from PRECISION to SPEED mode
- Processing files >5000 lines
- Making changes to >10 files
- Destructive operations (deletions)

Remember: You are helping developers write better code faster. Be precise, be helpful, be proactive.
"""


# =============================================================================
# Agent 생성
# =============================================================================

# =============================================================================
# Agent 설정 및 생성 (Exports for CLI)
# =============================================================================

# WORKSPACE 루트 설정 (보안 경계)
workspace_root = os.environ.get("WORKSPACE", os.getcwd())

# 모델 설정 - OpenRouter 사용
model = ChatOpenAI(
    model=os.environ.get("MODEL", "moonshotai/kimi-k2-0905"),
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
    temperature=0,
)

# Backend 인스턴스 생성 (전역 변수로 저장하여 change_project_directory에서 업데이트 가능)
_filesystem_backend = FilesystemBackend(
    root_dir=os.getcwd(),  # 현재 작업 디렉토리
    virtual_mode=False,  # change_project_directory에서 WORKSPACE 검증
)

# Agent 생성 (체크포인터 없는 기본 버전 - LangGraph dev용)
agent = create_deep_agent(
    model=model,
    tools=[
        analyze_impact,
        execute_python_code,
        run_pytest,
        delete_file,
        change_project_directory,
    ],
    system_prompt=CODING_ASSISTANT_PROMPT,
    backend=_filesystem_backend,
    subagents=[
        speed_analyzer_agent,
        precision_analyzer_agent,
        code_refactor_agent,
        test_generator_agent,
        doc_sync_agent,
        file_summarizer_agent,
    ],
    interrupt_on={
        # PRECISION 모드 실패시 SPEED 모드로 전환 제안하도록 인터럽트
        "analyze_impact": False,  # 결과를 보고 판단
        # 대용량 파일 처리시 인터럽트
        "read_file": False,
        # 파일 쓰기/편집시는 자동 진행
        "write_file": False,
        "edit_file": False,
        # 파일 삭제는 파괴적 작업이므로 반드시 사용자 승인 필요
        "delete_file": True,  # Human-in-the-Loop 필수
        # 디렉토리 변경은 자동 허용 (WORKSPACE 범위 내에서만)
        "change_project_directory": False,
    },
)


# Graph export for LangGraph
graph = agent
