"""에이전트에게 파일 시스템 도구를 제공하는 미들웨어 구현."""
# ruff: noqa: E501

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated
from typing_extensions import NotRequired

import os
from typing import Literal, Optional

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends.protocol import BackendProtocol, BackendFactory, WriteResult, EditResult
from deepagents.backends import StateBackend
from deepagents.backends.utils import (
    create_file_data,
    update_file_data,
    format_content_with_line_numbers,
    format_grep_matches,
    truncate_if_too_long,
)

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 2000
BACKEND_TYPES = (
    BackendProtocol
    | BackendFactory
)


class FileData(TypedDict):
    """파일 내용과 메타데이터를 함께 보관하기 위한 구조."""

    content: list[str]
    """파일 각 줄을 순서대로 저장한 목록."""

    created_at: str
    """파일이 최초 생성된 시점의 ISO 8601 타임스탬프."""

    modified_at: str
    """파일이 마지막으로 수정된 시점의 ISO 8601 타임스탬프."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """파일 상태 딕셔너리를 병합하면서 삭제 요청까지 처리한다.

    `right` 딕셔너리에서 값이 `None`인 키는 삭제 대상으로 간주하여 결과에서 제거하고,
    그 외 값은 기존 항목을 덮어쓴다. LangGraph의 주석형 리듀서 패턴과 호환되도록 설계되었다.

    Args:
        left: 기존 파일 상태. 초기화 단계에서는 `None`일 수 있다.
        right: 새로 반영할 파일 상태. 값이 `None`인 키는 삭제 신호로 취급된다.

    Returns:
        삭제를 반영하고 최신 내용으로 갱신된 파일 상태 딕셔너리.
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    """파일 경로를 보안 기준에 맞게 정규화한다.

    디렉터리 트래버설(`..`, `~`)을 차단하고, 슬래시 표기법을 통일하며, 필요 시
    허용된 경로 접두어만 사용할 수 있도록 검증한다.

    Args:
        path: 검증 및 정규화를 수행할 경로.
        allowed_prefixes: 허용할 경로 접두어 목록. 지정되면 정규화된 경로가 반드시
            해당 접두어 중 하나로 시작해야 한다.

    Returns:
        선행 슬래시(`/`)와 정방향 슬래시를 사용하는 정규화된 경로 문자열.

    Raises:
        ValueError: 경로에 허용되지 않은 패턴이 있거나 허용 접두어 조건을 만족하지 못할 때.
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized

class FilesystemState(AgentState):
    """파일 시스템 미들웨어가 유지하는 LangGraph 상태 정의."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """가상 파일 시스템의 파일 스냅샷 딕셔너리."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, filtering by directory.

Usage:
- The path parameter must be an absolute path, not a relative path
- The list_files tool will return a list of all files in the specified directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- The write_file tool will create the a new file.
- Prefer to edit existing files over creating new ones when possible."""


GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Usage:
- The glob tool finds files by matching patterns with wildcards
- Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character)
- Patterns can be absolute (starting with `/`) or relative
- Returns a list of absolute file paths that match the pattern

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a pattern in files.

Usage:
- The grep tool searches for text patterns across files
- The pattern parameter is the text to search for (literal string, not regex)
- The path parameter filters which directory to search in (default is the current working directory)
- The glob parameter accepts a glob pattern to filter which files to search (e.g., `*.py`)
- The output_mode parameter controls the output format:
  - `files_with_matches`: List only file paths containing matches (default)
  - `content`: Show matching lines with file path and line numbers
  - `count`: Show count of matches per file

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`"""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """런타임과 설정된 백엔드 정의를 바탕으로 실제 백엔드를 반환한다.

    Args:
        backend: `BackendProtocol` 구현체 또는 `ToolRuntime`을 입력받는 팩토리.
        runtime: 현재 LangChain 도구 실행 런타임.

    Returns:
        파일 작업을 수행할 `BackendProtocol` 구현체.
    """
    if callable(backend):
        return backend(runtime)
    return backend


def _ls_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`ls`(파일 목록) 도구 생성기.

    Args:
        backend: 파일 시스템 작업을 담당할 백엔드 또는 런타임을 받아 백엔드를 만드는 팩토리.
        custom_description: 기본 설명 대신 사용할 사용자 정의 문구.

    Returns:
        백엔드를 통해 디렉터리 목록을 반환하는 `BaseTool`.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> list[str]:
        resolved_backend = _get_backend(backend, runtime)
        validated_path = _validate_path(path)
        infos = resolved_backend.ls_info(validated_path)
        return [fi.get("path", "") for fi in infos]

    return ls


def _read_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`read_file` 도구 생성기.

    Args:
        backend: 파일 읽기를 처리할 백엔드 또는 런타임을 입력받는 백엔드 팩토리.
        custom_description: 기본 설명을 대체할 사용자 정의 문구.

    Returns:
        백엔드를 통해 파일 내용을 읽어오는 `BaseTool`.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        return resolved_backend.read(file_path, offset=offset, limit=limit)

    return read_file


def _write_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`write_file` 도구 생성기.

    Args:
        backend: 파일 생성을 담당할 백엔드 또는 런타임 입력 기반 백엔드 팩토리.
        custom_description: 기본 설명을 덮어쓸 문구.

    Returns:
        백엔드를 사용해 새 파일을 만드는 `BaseTool`.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: WriteResult = resolved_backend.write(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
        if res.files_update is not None:
            return Command(update={
                "files": res.files_update,
                "messages": [
                    ToolMessage(
                        content=f"Updated file {res.path}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })
        return f"Updated file {res.path}"

    return write_file


def _edit_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`edit_file` 도구 생성기.

    Args:
        backend: 문자열 치환을 수행할 백엔드 또는 런타임 기반 백엔드 팩토리.
        custom_description: 기본 설명을 덮어쓸 사용자 정의 문구.

    Returns:
        파일 내용에서 문자열을 치환해 주는 `BaseTool`.
    """
    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        resolved_backend = _get_backend(backend, runtime)
        file_path = _validate_path(file_path)
        res: EditResult = resolved_backend.edit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(update={
                "files": res.files_update,
                "messages": [
                    ToolMessage(
                        content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            })
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    return edit_file


def _glob_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`glob` 도구 생성기.

    Args:
        backend: 패턴 검색을 수행할 백엔드 또는 런타임 입력 기반 백엔드 팩토리.
        custom_description: 기본 설명을 덮어쓸 문구.

    Returns:
        글롭 패턴으로 파일 경로를 찾는 `BaseTool`.
    """
    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> list[str]:
        resolved_backend = _get_backend(backend, runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        return [fi.get("path", "") for fi in infos]

    return glob


def _grep_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """`grep` 도구 생성기.

    Args:
        backend: 패턴 검색을 처리할 백엔드 또는 런타임 기반 팩토리.
        custom_description: 기본 설명을 덮어쓸 문구.

    Returns:
        파일에서 문자열을 검색하는 `BaseTool`.
    """
    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    @tool(description=tool_description)
    def grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: Optional[str] = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        resolved_backend = _get_backend(backend, runtime)
        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    return grep


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
    "glob": _glob_tool_generator,
    "grep": _grep_tool_generator,
}


def _get_filesystem_tools(
    backend: BackendProtocol,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> list[BaseTool]:
    """파일 시스템 관련 도구들을 생성해 모은다.

    Args:
        backend: 파일 작업을 담당할 백엔드 인스턴스 또는 백엔드 팩토리.
        custom_tool_descriptions: 도구별 설명을 덮어쓸 커스텀 문구 딕셔너리.

    Returns:
        `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` 도구 목록.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []
    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here are the first 10 lines of the result:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """파일 시스템 도구 여섯 가지를 에이전트에 주입하는 미들웨어.

    이 미들웨어는 `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
    도구를 설정하고, `BackendProtocol`을 구현한 임의의 저장소를 통해 파일을
    보존한다.

    Args:
        backend: 파일 저장소 백엔드 또는 백엔드 팩토리. 지정하지 않으면
            LangGraph 상태에 쓰는 `StateBackend`를 사용한다. 지속 저장이나
            하이브리드 구성이 필요하면 `CompositeBackend`에 라우트를 추가한다.
        system_prompt: 파일 시스템 사용법을 설명하는 시스템 프롬프트를 덮어쓰기 위한 옵션.
        custom_tool_descriptions: 각 도구 설명을 덮어쓰기 위한 옵션.
        tool_token_limit_before_evict: 도구 결과가 이 토큰 수를 넘으면 파일 시스템에
            저장하도록 유도하는 임계값.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.memory.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # 기본값: 에이전트 상태에만 저장
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # 하이브리드 저장 구성 (임시 + /memories/ 경로는 영구 저장)
        backend = CompositeBackend(
            default=StateBackend(),
            routes={"/memories/": StoreBackend()}
        )
        agent = create_agent(middleware=[FilesystemMiddleware(memory_backend=backend)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """파일 시스템 미들웨어 인스턴스를 초기화한다.

        Args:
            backend: 파일 저장을 담당할 백엔드 또는 백엔드 팩토리. 생략하면 `StateBackend`를 사용한다.
            system_prompt: 시스템 프롬프트 전체를 덮어쓸 때 사용하는 문자열.
            custom_tool_descriptions: 도구별 설명을 덮어쓸 딕셔너리.
            tool_token_limit_before_evict: 도구 결과를 파일 시스템으로 이동시키기 전 허용할 토큰 수.
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict

        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Set system prompt (allow full override)
        self.system_prompt = system_prompt if system_prompt is not None else FILESYSTEM_SYSTEM_PROMPT

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출 전 파일 시스템 안내문을 시스템 프롬프트에 삽입한다.

        Args:
            request: 현재 처리 중인 모델 요청.
            handler: 수정된 요청을 넘겨 실행할 핸들러.

        Returns:
            핸들러가 반환한 모델 응답.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """비동기 모델 호출에서도 시스템 프롬프트를 보강한다.

        Args:
            request: 현재 처리 중인 모델 요청.
            handler: 수정된 요청을 넘겨 실행할 비동기 핸들러.

        Returns:
            핸들러가 반환한 모델 응답.
        """
        if self.system_prompt is not None:
            request.system_prompt = request.system_prompt + "\n\n" + self.system_prompt if request.system_prompt else self.system_prompt
        return await handler(request)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command) -> ToolMessage | Command:
        """도구 호출 결과가 임계값을 넘으면 파일 시스템으로 이동하도록 조치한다.

        Args:
            tool_result: 도구 호출이 반환한 `ToolMessage` 또는 `Command`.

        Returns:
            필요 시 파일 경로와 안내 메시지를 포함하도록 수정한 결과 객체.
        """
        if isinstance(tool_result, ToolMessage) and isinstance(tool_result.content, str):
            content = tool_result.content
            if self.tool_token_limit_before_evict and len(content) > 4 * self.tool_token_limit_before_evict:
                # 결과가 너무 큰 경우 파일 시스템에 저장할 경로를 준비
                file_path = f"/large_tool_results/{tool_result.tool_call_id}"
                file_data = create_file_data(content)
                state_update = {
                    "messages": [
                        ToolMessage(
                            TOO_LARGE_TOOL_MSG.format(
                                tool_call_id=tool_result.tool_call_id,
                                file_path=file_path,
                                content_sample=format_content_with_line_numbers(file_data["content"][:10], start_line=1),
                            ),
                            tool_call_id=tool_result.tool_call_id,
                        )
                    ],
                    "files": {file_path: file_data},
                }
                return Command(update=state_update)
        elif isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            message_updates = update.get("messages", [])
            file_updates = update.get("files", {})

            edited_message_updates = []
            for message in message_updates:
                if self.tool_token_limit_before_evict and isinstance(message, ToolMessage) and isinstance(message.content, str):
                    content = message.content
                    if len(content) > 4 * self.tool_token_limit_before_evict:
                        # 메시지 내용을 파일로 내보내고 안내 메시지를 삽입
                        file_path = f"/large_tool_results/{message.tool_call_id}"
                        file_data = create_file_data(content)
                        edited_message_updates.append(
                            ToolMessage(
                                TOO_LARGE_TOOL_MSG.format(
                                    tool_call_id=message.tool_call_id,
                                    file_path=file_path,
                                    content_sample=format_content_with_line_numbers(file_data["content"][:10], start_line=1),
                                ),
                                tool_call_id=message.tool_call_id,
                            )
                        )
                        file_updates[file_path] = file_data
                        continue
                edited_message_updates.append(message)
            return Command(update={**update, "messages": edited_message_updates, "files": file_updates})
        return tool_result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 결과가 과도하게 클 경우 파일 시스템으로 이관한다.

        Args:
            request: 처리 중인 도구 호출 요청.
            handler: 실제 도구를 실행할 핸들러.

        Returns:
            원본 `ToolMessage` 또는 상태 업데이트를 포함한 `Command`.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """비동기 도구 호출에서도 결과 크기를 확인해 필요 시 파일에 저장한다.

        Args:
            request: 처리 중인 도구 호출 요청.
            handler: 도구를 실행할 비동기 핸들러.

        Returns:
            원본 `ToolMessage` 또는 상태 업데이트를 포함한 `Command`.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result)
