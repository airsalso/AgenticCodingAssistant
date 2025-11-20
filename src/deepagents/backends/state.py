"""StateBackend: LangGraph 에이전트 상태에 파일을 보관하는 백엔드."""

import re
from typing import Any, Literal, Optional, TYPE_CHECKING

from langchain.tools import ToolRuntime

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    _glob_search_files,
    grep_matches_from_files,
)
from deepagents.backends.utils import FileInfo, GrepMatch
from deepagents.backends.protocol import WriteResult, EditResult


class StateBackend:
    """LangGraph 상태를 저장소로 활용하는 휘발성 파일 백엔드.

    대화 스레드 동안에는 파일 상태가 유지되지만 스레드가 바뀌면 초기화된다.
    LangGraph의 체크포인터와 연동되며, 상태 갱신은 항상 `Command` 객체를 통해
    수행해야 한다.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """런타임 핸들을 받아 상태 백엔드를 초기화한다."""
        self.runtime = runtime
    
    def ls_info(self, path: str) -> list[FileInfo]:
        """상태에 저장된 파일 목록을 반환한다."""
        files = self.runtime.state.get("files", {})
        infos: list[FileInfo] = []
        for k, fd in files.items():
            if not k.startswith(path):
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": k,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", ""),
            })
        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    # 간결한 API 유지를 위해 구형 ls() 헬퍼는 제거
    
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일을 읽고 줄 번호를 포함한 문자열을 반환한다."""
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return f"Error: File '{file_path}' not found"
        
        return format_read_response(file_data, offset, limit)
    
    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성하고 상태 업데이트 정보를 반환한다."""
        files = self.runtime.state.get("files", {})
        
        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")
        
        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: new_file_data})
    
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """파일 내용에서 문자열을 치환하고 상태를 갱신한다."""
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)
        
        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return EditResult(error=result)
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        return EditResult(path=file_path, files_update={file_path: new_file_data}, occurrences=int(occurrences))
    
    # 간결한 API 유지를 위해 구형 grep() 헬퍼는 제거

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """상태에 저장된 파일에서 패턴 검색을 수행한다."""
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """글롭 패턴에 매칭되는 파일 정보를 반환한다."""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append({
                "path": p,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", "") if fd else "",
            })
        return infos

# 공급자 클래스를 제거하고 `lambda rt: StateBackend(rt)` 형태의 팩토리를 사용
