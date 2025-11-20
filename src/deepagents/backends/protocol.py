"""플러그형 메모리 백엔드를 위한 프로토콜 정의.

모든 백엔드 구현이 따라야 하는 `BackendProtocol`을 제공하며, 각 백엔드는
상태, 파일 시스템, 데이터베이스 등 다양한 저장소에 파일을 보관하되 동일한
인터페이스로 파일 작업을 수행해야 한다.
"""

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable, Callable, TypeAlias, Any
from langchain.tools import ToolRuntime
from deepagents.backends.utils import FileInfo, GrepMatch

from dataclasses import dataclass


@dataclass
class WriteResult:
    """백엔드 쓰기 작업 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 `None`.
        path: 성공 시 작성된 파일의 절대 경로, 실패 시 `None`.
        files_update: 체크포인트 기반 백엔드가 LangGraph 상태에 반영할 업데이트.
            외부 저장소를 사용하면 이미 영구 저장되므로 `None`이 된다.

    Examples:
        >>> # 체크포인트 저장소
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # 외부 저장소
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # 오류 사례
        >>> WriteResult(error="File exists")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None


@dataclass
class EditResult:
    """백엔드 편집 작업 결과.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 `None`.
        path: 편집된 파일의 절대 경로.
        files_update: 체크포인트 백엔드가 상태를 갱신할 때 사용하는 딕셔너리.
        occurrences: 실제로 치환된 횟수. 실패 시 `None`.

    Examples:
        >>> # 체크포인트 저장소
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # 외부 저장소
        >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
        >>> # 오류 사례
        >>> EditResult(error="File not found")
    """

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None

@runtime_checkable
class BackendProtocol(Protocol):
    """플러그형 메모리 백엔드를 위한 공통 인터페이스.

    백엔드는 상태, 파일 시스템, 데이터베이스 등 다양한 저장소에 파일을 보관할 수 있지만
    이 프로토콜이 정의한 메서드를 통해 일관된 방식으로 동작해야 한다.

    모든 파일 데이터는 다음과 같은 딕셔너리 형태를 따른다:
    {
        "content": list[str],      # 파일 본문을 줄 단위로 저장
        "created_at": str,         # ISO 형식의 생성 시각
        "modified_at": str,        # ISO 형식의 수정 시각
    }
    """

    def ls_info(self, path: str) -> list["FileInfo"]:
        """파일 메타데이터를 포함한 구조화된 목록을 반환한다."""
        ...

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """줄 번호가 포함된 파일 내용을 읽거나 오류 메시지를 반환한다."""
        ...

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list["GrepMatch"] | str:
        """검색 결과 목록 또는 잘못된 입력에 대한 오류 문자열을 반환한다."""
        ...

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """글롭 패턴에 매칭된 파일 정보를 반환한다."""
        ...

    def write(
            self,
            file_path: str,
            content: str,
    ) -> WriteResult:
        """새 파일을 생성하고 `WriteResult`를 반환한다."""
        ...

    def edit(
            self,
            file_path: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
    ) -> EditResult:
        """파일에서 문자열을 치환하고 `EditResult`를 반환한다."""
        ...


BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
