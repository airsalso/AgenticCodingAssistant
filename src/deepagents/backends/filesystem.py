"""FilesystemBackend: 실제 파일 시스템과 직접 상호작용하는 백엔드 구현.

보안 및 검색 개선 사항:
- `virtual_mode` 활성화 시 루트 경계 안으로 경로를 정규화하여 디렉터리 탈출 방지
- 가능하면 `O_NOFOLLOW`를 사용해 심볼릭 링크를 따라가지 않음
- JSON 파싱을 활용한 ripgrep 기반 검색과 정규식을 활용한 파이썬 백업 검색 제공,
  동시에 가상 경로 처리 규칙을 유지
"""

import os
import re
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from .utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
    truncate_if_too_long,
)
import wcmatch.glob as wcglob
from deepagents.backends.utils import FileInfo, GrepMatch
from deepagents.backends.protocol import WriteResult, EditResult



class FilesystemBackend:
    """로컬 파일 시스템과 직접 입출력을 수행하는 백엔드.

    실제 파일 경로를 사용하며 상대 경로는 현재 작업 디렉터리를 기준으로 해석한다.
    파일 내용은 텍스트로 읽고 쓰고, 생성/수정 시간 등 메타데이터는 파일 시스템의
    속성값을 이용해 계산한다.
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        """파일 시스템 백엔드를 초기화한다.

        Args:
            root_dir: 파일 작업의 기준이 될 루트 디렉터리. 지정하지 않으면 현재 작업 디렉터리를 사용한다.
            virtual_mode: 가상 모드 여부. `True`이면 루트 밖으로 벗어나는 경로를 차단한다.
            max_file_size_mb: 검색 시 읽을 수 있는 최대 파일 크기(MB 단위).
        """
        self.cwd = Path(root_dir) if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """보안 검사를 거쳐 파일 경로를 정규화한다.

        `virtual_mode=True`이면 모든 입력 경로를 루트 하위 가상 경로로 취급하여
        `..`·`~`와 같은 트래버설을 금지하고 루트 밖으로 벗어나지 않도록 검사한다.
        비가상 모드에서는 절대 경로를 그대로 허용하고, 상대 경로는 현재 작업
        디렉터리를 기준으로 해석한다.

        Args:
            key: 절대/상대/가상 경로 문자열.

        Returns:
            정규화된 절대 `Path` 객체.
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(f"Path outside root directory: {key}") from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def ls_info(self, path: str) -> list[FileInfo]:
        """파일 시스템에서 디렉터리 목록을 조회한다.

        Args:
            path: 나열할 절대 디렉터리 경로.

        Returns:
            `FileInfo` 구조와 호환되는 딕셔너리 목록.
        """
        dir_path = self._resolve_path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[FileInfo] = []

        # 현재 작업 디렉터리를 문자열로 변환하여 비교
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # 디렉터리 트리를 순회
        try:
            for path in dir_path.rglob("*"):
                # 파일만 추려내어 메타데이터를 수집한다
                try:
                    is_file = path.is_file()
                except OSError:
                    continue
                if is_file:
                    abs_path = str(path)
                    if not self.virtual_mode:
                        try:
                            st = path.stat()
                            results.append({
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                            })
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                        continue
                    # 현재 작업 디렉터리 접두어가 있으면 제거
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str):]
                    elif abs_path.startswith(str(self.cwd)):
                        # 작업 디렉터리 문자열이 슬래시로 끝나지 않는 경우 처리
                        relative_path = abs_path[len(str(self.cwd)):].lstrip("/")
                    else:
                        # 범위를 벗어난 경로는 그대로 유지
                        relative_path = abs_path

                    virt_path = "/" + relative_path
                    try:
                        st = path.stat()
                        results.append({
                            "path": virt_path,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": virt_path, "is_dir": False})
        except (OSError, PermissionError):
            pass

        # 경로 기준으로 정렬해 순서를 안정적으로 유지
        results.sort(key=lambda x: x.get("path", ""))
        return results

    # 간결한 API를 위해 구형 ls() 헬퍼는 제거

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일을 읽고 줄 번호를 추가해 반환한다.

        Args:
            file_path: 절대 또는 상대 파일 경로.
            offset: 읽기 시작할 줄 번호(0부터 시작).
            limit: 읽을 최대 줄 수.

        Returns:
            줄 번호가 포함된 문자열 또는 오류 메시지.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"

        try:
            # 심볼릭 링크를 따라가지 않도록 가능하면 O_NOFOLLOW 플래그 사용
            try:
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                # 플래그가 지원되지 않거나 실패하면 일반 open 사용
                with open(resolved_path, "r", encoding="utf-8") as f:
                    content = f.read()

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
        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"
    
    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성하고 내용을 기록한다.

        Args:
            file_path: 생성할 파일의 절대 또는 상대 경로.
            content: 파일에 기록할 문자열.

        Returns:
            성공 또는 오류 정보를 담은 `WriteResult`.
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        try:
            # 필요한 경우 상위 디렉터리를 생성
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # 심볼릭 링크를 따라가지 않도록 가능하면 O_NOFOLLOW 사용
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            
            return WriteResult(path=file_path, files_update=None)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")
    
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """파일에서 특정 문자열을 치환한다.

        Args:
            file_path: 수정할 파일의 경로.
            old_string: 치환 대상 문자열.
            new_string: 새 문자열.
            replace_all: `True`이면 모든 일치 항목을 교체.

        Returns:
            치환 결과 및 발생 횟수를 담은 `EditResult`.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # 안전하게 읽기
            try:
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            
            if isinstance(result, str):
                return EditResult(error=result)
            
            new_content, occurrences = result
            
            # 안전하게 쓰기
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")
    
    # Removed legacy grep() convenience to keep lean surface

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """파일 시스템에서 패턴 검색을 수행하고 원시 결과를 반환한다.

        Args:
            pattern: 정규식 또는 문자열 패턴.
            path: 검색 기준 경로.
            glob: 글롭 패턴으로 검색 범위를 제한.

        Returns:
            파일별 `GrepMatch` 목록 또는 오류 문자열.
        """

        # 정규식을 먼저 검증
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # 기준 경로 정규화
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return []

        if not base_full.exists():
            return []

        # 우선 ripgrep을 사용하고 실패하면 파이썬 구현으로 대체
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            # 각 파일별로 (줄 번호, 내용) 쌍을 구조화된 결과에 누적
            for line_num, line_text in items:
                matches.append({"path": fpath, "line": int(line_num), "text": line_text})
        return matches

    def _ripgrep_search(
        self, pattern: str, base_full: Path, include_glob: Optional[str]
    ) -> Optional[dict[str, list[tuple[int, str]]]]:
        """ripgrep CLI를 호출해 매치 결과를 수집한다."""
        cmd = ["rg", "--json"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = "/" + str(p.resolve().relative_to(self.cwd))
                except Exception:
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(
        self, pattern: str, base_full: Path, include_glob: Optional[str]
    ) -> dict[str, list[tuple[int, str]]]:
        """파이썬 정규식으로 파일 내용을 검색하는 백업 루틴."""
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if include_glob and not wcglob.globmatch(fp.name, include_glob, flags=wcglob.BRACE):
                continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results
    
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """글롭 패턴으로 파일 정보를 수집한다."""
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return []

        results: list[FileInfo] = []
        try:
            # 하위 디렉터리까지 재귀적으로 탐색하여 패턴을 찾는다
            for matched_path in search_path.rglob(pattern):
                # 글롭 패턴과 일치하는 경로를 찾아 메타데이터를 수집
                try:
                    is_file = matched_path.is_file()
                except OSError:
                    continue
                if not is_file:
                    continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append({
                            "path": abs_path,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    cwd_str = str(self.cwd)
                    if not cwd_str.endswith("/"):
                        cwd_str += "/"
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str):]
                    elif abs_path.startswith(str(self.cwd)):
                        relative_path = abs_path[len(str(self.cwd)):].lstrip("/")
                    else:
                        relative_path = abs_path
                    virt = "/" + relative_path
                    try:
                        st = matched_path.stat()
                        results.append({
                            "path": virt,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results
