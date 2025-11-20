"""StoreBackend: LangGraph BaseStore와 연동되는 영속 백엔드."""

import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

from langgraph.config import get_config
from langgraph.store.base import BaseStore, Item
from deepagents.backends.protocol import WriteResult, EditResult

from deepagents.backends.utils import (
    create_file_data,
    update_file_data,
    file_data_to_string,
    format_read_response,
    perform_string_replacement,
    _glob_search_files,
    grep_matches_from_files,
)
from deepagents.backends.utils import FileInfo, GrepMatch


class StoreBackend:
    """LangGraph BaseStore에 파일을 저장하는 영속 백엔드.

    대화를 넘어서도 파일이 유지되며, 네임스페이스를 통해 스레드와 어시스턴트를
    분리한다.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """런타임 핸들을 받아 스토어 백엔드를 초기화한다."""
        self.runtime = runtime


    def _get_store(self) -> BaseStore:
        """런타임에서 BaseStore 인스턴스를 꺼낸다.

        Raises:
            ValueError: 런타임에 스토어가 없을 때.
        """
        store = self.runtime.store
        if store is None:
            msg = "Store is required but not available in runtime"
            raise ValueError(msg)
        return store
    
    def _get_namespace(self) -> tuple[str, ...]:
        """스토어 작업에 사용할 네임스페이스를 결정한다.

        우선순위:
        1) `self.runtime.config`가 있으면 해당 값을 사용.
        2) 없으면 `langgraph.config.get_config()`을 조회.
        3) 모두 실패하면 `("filesystem",)`을 기본값으로 사용.

        설정 메타데이터에 `assistant_id`가 있으면 `(assistant_id, "filesystem")`를 반환해
        멀티 에이전트를 분리한다.
        """
        namespace = "filesystem"

        # 런타임이 제공하는 설정 정보를 우선 사용
        runtime_cfg = getattr(self.runtime, "config", None)
        if isinstance(runtime_cfg, dict):
            assistant_id = runtime_cfg.get("metadata", {}).get("assistant_id")
            if assistant_id:
                return (assistant_id, namespace)
            return (namespace,)

        # LangGraph 컨텍스트로 폴백하되 실행 컨텍스트 밖에서 호출될 때의 예외를 감안
        try:
            cfg = get_config()
        except Exception:
            return (namespace,)

        try:
            assistant_id = cfg.get("metadata", {}).get("assistant_id")  # type: ignore[assignment]
        except Exception:
            assistant_id = None

        if assistant_id:
            return (assistant_id, namespace)
        return (namespace,)
    
    def _convert_store_item_to_file_data(self, store_item: Item) -> dict[str, Any]:
        """스토어 항목을 `FileData` 구조로 변환한다.

        Args:
            store_item: 파일 데이터를 담고 있는 스토어 항목.

        Returns:
            `content`, `created_at`, `modified_at` 키를 가진 딕셔너리.

        Raises:
            ValueError: 필수 필드가 없거나 타입이 잘못된 경우.
        """
        if "content" not in store_item.value or not isinstance(store_item.value["content"], list):
            msg = f"Store item does not contain valid content field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "created_at" not in store_item.value or not isinstance(store_item.value["created_at"], str):
            msg = f"Store item does not contain valid created_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        if "modified_at" not in store_item.value or not isinstance(store_item.value["modified_at"], str):
            msg = f"Store item does not contain valid modified_at field. Got: {store_item.value.keys()}"
            raise ValueError(msg)
        return {
            "content": store_item.value["content"],
            "created_at": store_item.value["created_at"],
            "modified_at": store_item.value["modified_at"],
        }
    
    def _convert_file_data_to_store_value(self, file_data: dict[str, Any]) -> dict[str, Any]:
        """`FileData`를 스토어에 저장 가능한 형태로 변환한다."""
        return {
            "content": file_data["content"],
            "created_at": file_data["created_at"],
            "modified_at": file_data["modified_at"],
        }

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        page_size: int = 100,
    ) -> list[Item]:
        """페이지네이션을 적용해 조건에 맞는 모든 항목을 수집한다.

        Args:
            store: 조회할 BaseStore 인스턴스.
            namespace: 검색할 네임스페이스 경로.
            query: 자연어 검색 쿼리.
            filter: 결과를 제한할 키-값 필터.
            page_size: 페이지당 가져올 항목 수.

        Returns:
            검색 조건에 부합하는 `Item` 리스트.
        """
        all_items: list[Item] = []
        offset = 0
        while True:
            page_items = store.search(
                namespace,
                query=query,
                filter=filter,
                limit=page_size,
                offset=offset,
            )
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < page_size:
                break
            offset += page_size

        return all_items
    
    def ls_info(self, path: str) -> list[FileInfo]:
        """스토어에서 파일 목록을 조회한다."""
        store = self._get_store()
        namespace = self._get_namespace()
        
        # 스토어별 필터 규칙에 의존하지 않도록 로컬에서 경로 접두어로 필터링
        items = self._search_store_paginated(store, namespace)
        infos: list[FileInfo] = []
        for item in items:
            if not str(item.key).startswith(path):
                continue
            try:
                fd = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": item.key,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", ""),
            })
        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    # Removed legacy ls() convenience to keep lean surface
    
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일을 읽어 줄 번호가 포함된 문자열로 반환한다.

        Args:
            file_path: 조회할 파일의 절대 경로.
            offset: 읽기 시작 줄 번호(0부터 시작).
            limit: 읽을 최대 줄 수.

        Returns:
            줄 번호가 포함된 문자열 또는 오류 메시지.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        item: Optional[Item] = store.get(namespace, file_path)
        
        if item is None:
            return f"Error: File '{file_path}' not found"
        
        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return f"Error: {e}"
        
        return format_read_response(file_data, offset, limit)
    
    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일을 생성하고 `WriteResult`를 반환한다.

        Args:
            file_path: 생성할 파일의 절대 경로.
            content: 파일에 기록할 문자열.

        Returns:
            성공 여부와 오류 메시지를 포함한 `WriteResult`.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        
        # 동일 경로의 파일이 이미 존재하는지 확인
        existing = store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")
        
        # 새 파일을 생성
        file_data = create_file_data(content)
        store_value = self._convert_file_data_to_store_value(file_data)
        store.put(namespace, file_path, store_value)
        return WriteResult(path=file_path, files_update=None)
    
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """파일에서 문자열을 치환하고 `EditResult`를 반환한다.

        Args:
            file_path: 수정할 파일의 절대 경로.
            old_string: 교체 대상 문자열.
            new_string: 새로 대체할 문자열.
            replace_all: `True`이면 모든 일치 항목을 치환.

        Returns:
            치환 결과와 횟수를 포함한 `EditResult`.
        """
        store = self._get_store()
        namespace = self._get_namespace()
        
        # 기존 파일을 조회
        item = store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")
        
        try:
            file_data = self._convert_store_item_to_file_data(item)
        except ValueError as e:
            return EditResult(error=f"Error: {e}")
        
        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)
        
        if isinstance(result, str):
            return EditResult(error=result)
        
        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        
        # 스토어에 수정된 데이터를 저장
        store_value = self._convert_file_data_to_store_value(new_file_data)
        store.put(namespace, file_path, store_value)
        return EditResult(path=file_path, files_update=None, occurrences=int(occurrences))
    
    # Removed legacy grep() convenience to keep lean surface

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """스토어에 저장된 파일에서 패턴 검색을 수행한다."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            # 스토어 항목을 FileData 형식으로 변환하여 검색 대상 딕셔너리에 적재
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
        return grep_matches_from_files(files, pattern, path, glob)
    
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """글롭 패턴 조건을 만족하는 파일 정보를 반환한다."""
        store = self._get_store()
        namespace = self._get_namespace()
        items = self._search_store_paginated(store, namespace)
        files: dict[str, Any] = {}
        for item in items:
            # glob 검색을 위해 FileData 구조로 변환
            try:
                files[item.key] = self._convert_store_item_to_file_data(item)
            except ValueError:
                continue
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


# Provider classes removed: prefer callables like `lambda rt: StoreBackend(rt)`
