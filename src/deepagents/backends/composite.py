"""경로 접두어에 따라 서로 다른 백엔드로 라우팅하는 CompositeBackend 구현."""

from typing import Any, Literal, Optional, TYPE_CHECKING

from langchain.tools import ToolRuntime

from deepagents.backends.protocol import BackendProtocol, BackendFactory, WriteResult, EditResult
from deepagents.backends.state import StateBackend
from deepagents.backends.utils import FileInfo, GrepMatch
from deepagents.backends.protocol import BackendFactory


class CompositeBackend:

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """경로 접두어에 따라 다른 백엔드를 사용할 수 있도록 초기화한다.

        Args:
            default: 라우트에 매칭되지 않는 경로를 처리할 기본 백엔드.
            routes: 경로 접두어와 백엔드의 매핑 딕셔너리.
        """
        # 기본 백엔드
        self.default = default

        # 가상 경로별 백엔드 라우팅 테이블
        self.routes = routes

        # 접두어가 긴 순으로 정렬하여 가장 구체적인 경로부터 검사
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """주어진 경로를 처리할 백엔드와 정규화된 키를 반환한다.

        Args:
            key: 원본 파일 경로.

        Returns:
            `(backend, stripped_key)` 튜플. `stripped_key`는 라우트 접두어를 제거하되
            선행 슬래시는 유지한다.
        """
        # 가장 긴 접두어부터 검사하여 일치 여부 판단
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # 접두어를 제거하되 선행 슬래시를 강제로 유지
                # 예: "/memories/notes.txt" → "/notes.txt"
                suffix = key[len(prefix):]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key

    def ls_info(self, path: str) -> list[FileInfo]:
        """경로에 해당하는 백엔드에서 파일 목록을 조회한다.

        Args:
            path: 조회할 절대 경로.

        Returns:
            라우트 접두어가 적용된 `FileInfo` 딕셔너리 리스트.
        """
        # 특정 라우트 하위인지 우선 확인
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # 해당 라우트에 연결된 백엔드만 조회
                suffix = path[len(route_prefix):]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # 루트에서는 기본 백엔드와 모든 라우트 백엔드를 합산
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))
            for route_prefix, backend in self.sorted_routes:
                infos = backend.ls_info("/")
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    results.append(fi)
            results.sort(key=lambda x: x.get("path", ""))
            return results

        # 어떤 라우트에도 해당하지 않으면 기본 백엔드만 조회
        return self.default.ls_info(path)


    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """경로에 맞는 백엔드에서 파일 내용을 읽어온다.

        Args:
            file_path: 절대 파일 경로.
            offset: 읽기 시작할 줄(0부터 시작).
            limit: 최대 읽기 줄 수.

        Returns:
            줄 번호가 포함된 파일 내용 또는 오류 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)


    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """여러 백엔드에서 패턴 검색을 수행하고 결과를 병합한다.

        Args:
            pattern: 검색할 정규식 또는 패턴 문자열.
            path: 검색 범위를 제한할 절대 경로.
            glob: 글롭 패턴으로 검색 대상을 제한할 때 사용.

        Returns:
            `GrepMatch` 목록 또는 오류 메시지 문자열.
        """
        # 경로가 특정 라우트를 가리키면 해당 백엔드만 검색
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # 아니면 기본 백엔드와 모든 라우트 백엔드를 검색 후 병합
        all_matches: list[GrepMatch] = []
        raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
        if isinstance(raw_default, str):
            # 오류 발생 시 문자열 메시지를 그대로 반환
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = backend.grep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                # 오류 발생 시 문자열 메시지를 그대로 반환
                return raw
            all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

        return all_matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """글롭 패턴으로 파일 목록을 검색한다.

        Args:
            pattern: 글롭 패턴 문자열.
            path: 검색 기준 경로. 기본값은 루트(`/`).

        Returns:
            라우트 접두어가 적용된 `FileInfo` 리스트.
        """
        results: list[FileInfo] = []

        # 패턴이 아닌 경로 기준으로 라우트 결정
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                infos = backend.glob_info(pattern, search_path if search_path else "/")
                return [
                    {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
                    for fi in infos
                ]

        # 특정 라우트를 찾지 못하면 기본 백엔드와 모든 라우트 백엔드를 조회
        results.extend(self.default.glob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = backend.glob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # 정렬하여 결과 순서를 안정적으로 유지
        results.sort(key=lambda x: x.get("path", ""))
        return results


    def write(
            self,
            file_path: str,
            content: str,
    ) -> WriteResult:
        """경로에 맞는 백엔드에 새 파일을 작성한다.

        Args:
            file_path: 절대 파일 경로.
            content: 파일 내용 문자열.

        Returns:
            성공 시 `WriteResult`, 실패 시 오류 메시지를 포함한 결과.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        # 상태 기반 업데이트라면 기본 백엔드의 상태에도 반영하여 목록이 최신화되도록 한다
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> EditResult:
        """경로에 맞는 백엔드에서 문자열 치환을 수행한다.

        Args:
            file_path: 절대 파일 경로.
            old_string: 교체 대상 문자열.
            new_string: 교체할 문자열.
            replace_all: `True`이면 모든 일치 항목을 교체.

        Returns:
            성공 시 `EditResult`, 실패 시 오류 정보를 포함한 결과.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res


 
