"""메모리 백엔드 구현에서 공용으로 사용하는 유틸리티 함수 모음.

백엔드와 컴포지트 라우터가 활용하는 문자열 포매터와 구조화된 도우미를 제공해
문자열 파싱 없이 안전하게 구성할 수 있도록 돕는다.
"""

import re
import wcmatch.glob as wcglob
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, List, Dict

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000  # 도구 결과를 파일로 대체하기 위한 토큰 임계값과 동일
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"


class FileInfo(TypedDict, total=False):
    """파일 목록 정보를 담는 구조체."""
    path: str
    is_dir: bool
    size: int  # 대략적인 바이트 크기
    modified_at: str  # 사용 가능한 경우 ISO 형식 수정 시각


class GrepMatch(TypedDict):
    """grep 결과를 구조화해 담는 엔트리."""
    path: str
    line: int
    text: str


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """파일 내용을 줄 번호와 함께 포맷한다.

    Args:
        content: 문자열 또는 줄 목록 형태의 파일 내용.
        start_line: 줄 번호 시작 값(기본값 1).

    Returns:
        `cat -n` 스타일의 줄 번호가 포함된 문자열.
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content
    
    return "\n".join(
        f"{i + start_line:{LINE_NUMBER_WIDTH}d}\t{line[:MAX_LINE_LENGTH]}" 
        for i, line in enumerate(lines)
    )


def check_empty_content(content: str) -> str | None:
    """내용이 비어 있는지 검사하고 경고 메시지를 반환한다.

    Args:
        content: 검사할 문자열.

    Returns:
        빈 내용일 경우 경고 메시지, 그렇지 않으면 `None`.
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def file_data_to_string(file_data: dict[str, Any]) -> str:
    """`FileData` 구조를 개행으로 연결된 문자열로 변환한다."""
    return "\n".join(file_data["content"])


def create_file_data(content: str, created_at: str | None = None) -> dict[str, Any]:
    """파일 내용을 바탕으로 `FileData` 사전을 생성한다.

    Args:
        content: 파일 내용 문자열.
        created_at: 생성 시각(ISO 형식). 생략 시 현재 시각 사용.

    Returns:
        내용과 생성/수정 시각이 포함된 딕셔너리.
    """
    lines = content.split("\n") if isinstance(content, str) else content
    lines = [line[i:i+MAX_LINE_LENGTH] for line in lines for i in range(0, len(line) or 1, MAX_LINE_LENGTH)]
    now = datetime.now(UTC).isoformat()
    
    return {
        "content": lines,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: dict[str, Any], content: str) -> dict[str, Any]:
    """기존 `FileData`에 새 내용을 반영한다.

    Args:
        file_data: 기존 파일 데이터 사전.
        content: 새로 기록할 문자열.

    Returns:
        수정된 내용과 최신 수정 시각을 포함한 `FileData`.
    """
    lines = content.split("\n") if isinstance(content, str) else content
    lines = [line[i:i+MAX_LINE_LENGTH] for line in lines for i in range(0, len(line) or 1, MAX_LINE_LENGTH)]
    now = datetime.now(UTC).isoformat()
    
    return {
        "content": lines,
        "created_at": file_data["created_at"],
        "modified_at": now,
    }


def format_read_response(
    file_data: dict[str, Any],
    offset: int,
    limit: int,
) -> str:
    """읽기 응답으로 사용할 문자열을 생성한다.

    Args:
        file_data: 줄 목록과 메타데이터가 담긴 `FileData` 사전.
        offset: 0부터 시작하는 줄 오프셋.
        limit: 출력할 최대 줄 수.

    Returns:
        줄 번호가 포함된 문자열 또는 오류 메시지.
    """
    content = file_data_to_string(file_data)
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg
    
    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))
    
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
    
    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> tuple[str, int] | str:
    """문자열 치환을 수행하고 발생 횟수를 검증한다.

    Args:
        content: 원본 문자열.
        old_string: 찾을 문자열.
        new_string: 교체할 문자열.
        replace_all: `True`이면 모든 일치 항목을 치환.

    Returns:
        성공 시 `(새_문자열, 치환 횟수)` 튜플, 실패 시 오류 메시지.
    """
    occurrences = content.count(old_string)
    
    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"
    
    if occurrences > 1 and not replace_all:
        return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
    
    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """결과 길이가 토큰 한도를 넘으면 잘라낸다.

    문자열은 4자 ≈ 1토큰으로 추산한다.
    """
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]
        return result
    else:  # 문자열
        if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
        return result


def _validate_path(path: str | None) -> str:
    """경로 문자열을 검증하고 정규화한다.

    Args:
        path: 검증할 경로.

    Returns:
        슬래시(`/`)로 시작하는 정규화된 경로.

    Raises:
        ValueError: 경로가 비어 있거나 잘못된 경우.
    """
    path = path or "/"
    if not path or path.strip() == "":
        raise ValueError("Path cannot be empty")
    
    normalized = path if path.startswith("/") else "/" + path
    
    if not normalized.endswith("/"):
        normalized += "/"
    
    return normalized


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    """글롭 패턴과 경로 조건에 맞는 파일 목록을 반환한다.

    Args:
        files: 파일 경로와 `FileData`를 매핑한 딕셔너리.
        pattern: 글롭 패턴(예: `"*.py"`, `"**/*.ts"`).
        path: 검색을 시작할 기준 경로.

    Returns:
        수정 시각 역순으로 정렬된 경로 문자열(개행 구분). 일치 항목이 없으면 `"No files found"`.
    """
    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No files found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    # 표준 글롭 규칙:
    # - 경로 구분자가 없는 패턴은 `path` 기준 현재 디렉터리만 검색.
    # - 재귀 검색이 필요하면 반드시 `"**"`를 사용.
    effective_pattern = pattern

    matches = []
    for file_path, file_data in filtered.items():
        relative = file_path[len(normalized_path) :].lstrip("/")
        if not relative:
            relative = file_path.split("/")[-1]

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """grep 결과를 출력 모드에 맞게 포맷한다.

    Args:
        results: 파일 경로별로 `(줄 번호, 내용)` 튜플을 모아 둔 딕셔너리.
        output_mode: 출력 형식. `files_with_matches`, `content`, `count` 중 하나.

    Returns:
        지정된 형식에 맞춰 생성한 문자열.
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    elif output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    else:
        lines = []
        for file_path in sorted(results.keys()):
            lines.append(f"{file_path}:")
            for line_num, line in results[file_path]:
                lines.append(f"  {line_num}: {line}")
        return "\n".join(lines)


def _grep_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """파일 내용에서 정규식 패턴을 검색한다.

    Args:
        files: 파일 경로와 `FileData` 딕셔너리.
        pattern: 찾을 정규식 패턴.
        path: 검색 기준 경로.
        glob: 글롭 패턴으로 검색 대상을 제한할 때 사용.
        output_mode: 결과 출력 형식.

    Returns:
        포맷된 검색 결과 문자열. 일치 항목이 없으면 "No matches found".
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return "No matches found"

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)


# -------- 구성에서 재사용되는 구조화 헬퍼 --------

def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str:
    """메모리에 저장된 파일 맵에서 구조화된 grep 결과를 반환한다.

    Args:
        files: 파일 경로와 `FileData` 사전.
        pattern: 검색할 정규식 패턴.
        path: 검색 기준 경로.
        glob: 글롭 패턴 필터.

    Returns:
        성공 시 `GrepMatch` 리스트, 잘못된 입력이면 오류 메시지 문자열.
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _validate_path(path)
    except ValueError:
        return []

    filtered = {fp: fd for fp, fd in files.items() if fp.startswith(normalized_path)}

    if glob:
        filtered = {
            fp: fd
            for fp, fd in filtered.items()
            if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)
        }

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        for line_num, line in enumerate(file_data["content"], 1):
            if regex.search(line):
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return matches


def build_grep_results_dict(matches: List[GrepMatch]) -> Dict[str, list[tuple[int, str]]]:
    """구조화된 매치를 포매터가 사용하는 딕셔너리 형태로 변환한다."""
    grouped: Dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: List[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """구조화된 grep 결과를 지정한 모드에 맞게 포맷한다."""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
