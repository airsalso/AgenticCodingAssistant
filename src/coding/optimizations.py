"""Performance Optimizations for DeepAgentic Code Assistant.

This module provides:
1. Configuration management
2. Context explosion prevention
3. LangSmith performance monitoring
4. File reading cache with improved key strategy
5. Parallel processing utilities
6. History management
"""

import functools
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """캐시 설정 (환경 변수 지원)."""
    max_size: int = int(os.getenv("CACHE_MAX_SIZE", "200"))
    max_memory_mb: int = int(os.getenv("CACHE_MAX_MEMORY_MB", "50"))
    max_file_size: int = int(os.getenv("CACHE_MAX_FILE_SIZE", "50000"))


@dataclass
class ProcessingConfig:
    """처리 설정."""
    max_file_list: int = int(os.getenv("MAX_FILE_LIST", "50"))
    max_workers: int = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 4)))


# 전역 설정 인스턴스
cache_config = CacheConfig()
processing_config = ProcessingConfig()

# 하위 호환성을 위한 상수
MAX_FILE_LIST = processing_config.max_file_list
MAX_FILE_SIZE_FOR_AUTO_READ = cache_config.max_file_size


# =============================================================================
# 1. Custom Exceptions
# =============================================================================

class FileReadError(Exception):
    """파일 읽기 실패 시 발생하는 예외."""
    pass


# =============================================================================
# 2. 컨텍스트 폭발 방지
# =============================================================================


def limit_file_list(files: List[str], max_count: int = MAX_FILE_LIST) -> Tuple[List[str], str]:
    """파일 목록을 제한하여 컨텍스트 폭발 방지.

    Args:
        files: 파일 경로 리스트
        max_count: 최대 표시 개수

    Returns:
        (제한된 파일 리스트, 초과 정보 메시지) 튜플
    """
    if len(files) <= max_count:
        return files, ""

    limited = files[:max_count]
    overflow_msg = (
        f"\n⚠️ 파일이 너무 많습니다 ({len(files)}개 중 {max_count}개만 표시)\n"
        f"추가 {len(files) - max_count}개 파일은 생략되었습니다.\n"
        f"glob 패턴을 사용하여 범위를 좁히거나 하위 디렉토리로 이동하세요."
    )

    return limited, overflow_msg


def truncate_large_content(content: str, max_lines: int = 1000) -> Tuple[str, bool]:
    """큰 파일 내용을 자동으로 자름.

    Args:
        content: 파일 내용
        max_lines: 최대 라인 수

    Returns:
        (잘린 내용, 잘렸는지 여부) 튜플
    """
    lines = content.splitlines()

    if len(lines) <= max_lines:
        return content, False

    truncated = "\n".join(lines[:max_lines])
    truncated += f"\n\n... (truncated {len(lines) - max_lines} lines)"

    return truncated, True


# =============================================================================
# 3. LangSmith 성능 모니터링
# =============================================================================

def monitor_performance(func_name: Optional[str] = None) -> Callable:
    """LangSmith와 함께 사용할 성능 모니터링 데코레이터.

    Args:
        func_name: 함수 이름 (없으면 자동 감지)

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = func_name or func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                logger.info(
                    f"✓ {name} completed in {elapsed:.2f}s",
                    extra={
                        "function": name,
                        "elapsed_time": elapsed,
                        "status": "success"
                    }
                )

                return result

            except Exception as e:
                elapsed = time.time() - start_time

                logger.error(
                    f"✗ {name} failed after {elapsed:.2f}s: {e}",
                    extra={
                        "function": name,
                        "elapsed_time": elapsed,
                        "status": "error",
                        "error": str(e)
                    }
                )

                raise

        return wrapper
    return decorator


# =============================================================================
# 4. 파일 읽기 캐싱
# =============================================================================

class FileCache:
    """파일 내용 캐싱 클래스 (개선된 캐시 키 전략 및 메모리 관리).

    LRU 정책을 사용하며 OrderedDict로 순서를 명시적으로 관리합니다.
    """

    def __init__(self, max_size: int = 100, max_memory_mb: int = 50):
        """파일 캐시 초기화.

        Args:
            max_size: 최대 캐시 항목 수
            max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self._cache: OrderedDict[str, Tuple[str, int]] = OrderedDict()
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self.hits = 0
        self.misses = 0

    def _get_cache_key(self, file_path: str) -> Optional[Tuple[str, float]]:
        """개선된 캐시 키 생성 (절대 경로 + mtime).

        Args:
            file_path: 파일 경로

        Returns:
            (캐시 키, mtime) 튜플 또는 None (오류 시)

        Note:
            절대 경로와 수정 시간을 조합하여 캐시 키를 생성합니다.
            심볼릭 링크를 해결하고 stat()은 한 번만 호출합니다.
        """
        try:
            path = Path(file_path).resolve()
            stat = path.stat()
            cache_key = f"{path}:{stat.st_mtime}"
            return (cache_key, stat.st_mtime)
        except OSError as e:
            logger.debug(f"Failed to get cache key for {file_path}: {e}")
            return None

    def get(self, file_path: str) -> Optional[str]:
        """캐시에서 파일 내용 가져오기 (LRU 업데이트).

        Args:
            file_path: 파일 경로

        Returns:
            캐시된 내용 또는 None
        """
        key_info = self._get_cache_key(file_path)
        if not key_info:
            self.misses += 1
            return None

        cache_key, _ = key_info

        if cache_key in self._cache:
            content, size = self._cache.pop(cache_key)
            self._cache[cache_key] = (content, size)
            self.hits += 1
            logger.debug(f"Cache hit: {file_path}")
            return content

        self.misses += 1
        return None

    def set(self, file_path: str, content: str) -> None:
        """파일 내용을 캐시에 저장 (메모리 제한 적용, LRU 정책).

        Args:
            file_path: 파일 경로
            content: 파일 내용
        """
        key_info = self._get_cache_key(file_path)
        if not key_info:
            return

        cache_key, _ = key_info
        content_size = len(content) * 2

        # 대용량 파일은 캐시하지 않음
        if content_size > cache_config.max_file_size:
            logger.debug(f"File too large for cache: {file_path} ({content_size} bytes)")
            return

        # LRU: 메모리 제한 초과 시 가장 오래된 항목 제거
        while (self.current_memory + content_size > self.max_memory_bytes or
               len(self._cache) >= self.max_size):
            if not self._cache:
                break
            oldest_key, (_, old_size) = self._cache.popitem(last=False)
            self.current_memory -= old_size
            logger.debug(f"Evicted from cache: {oldest_key}")

        self._cache[cache_key] = (content, content_size)
        self.current_memory += content_size
        logger.debug(f"Cache set: {file_path} ({content_size} bytes)")

    def clear(self) -> None:
        """캐시 전체 삭제."""
        self._cache.clear()
        self.current_memory = 0
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """캐시 통계 반환."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_mb": f"{self.current_memory / 1024 / 1024:.2f}",
            "max_memory_mb": f"{self.max_memory_bytes / 1024 / 1024:.0f}",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


# 전역 파일 캐시 인스턴스 (설정값 사용)
file_cache = FileCache(
    max_size=cache_config.max_size,
    max_memory_mb=cache_config.max_memory_mb
)


def read_file_with_cache(file_path: str, limit: int = 500, offset: int = 0) -> str:
    """캐시를 활용한 파일 읽기.

    Args:
        file_path: 파일 경로
        limit: 최대 라인 수
        offset: 시작 라인 번호

    Returns:
        파일 내용 (제한 적용) 또는 에러 메시지
    """
    path = Path(file_path)

    if not path.exists():
        error_msg = f"File '{file_path}' not found"
        logger.warning(error_msg)
        return f"Error: {error_msg}"

    # 캐시에서 먼저 확인
    cached = file_cache.get(str(path))

    if cached is not None:
        if offset > 0 or limit < float('inf'):
            lines = cached.splitlines()
            if offset >= len(lines):
                return ""
            selected = lines[offset:offset + limit]
            return "\n".join(selected)
        return cached

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError as e:
        error_msg = f"Encoding error reading '{file_path}': {e}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except IOError as e:
        error_msg = f"I/O error reading '{file_path}': {e}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error reading '{file_path}': {e}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

    if len(content) * 2 <= cache_config.max_file_size:
        file_cache.set(str(path), content)

    if offset > 0 or limit < float('inf'):
        lines = content.splitlines()
        if offset >= len(lines):
            return ""
        selected = lines[offset:offset + limit]
        return "\n".join(selected)

    return content


# =============================================================================
# 5. 병렬 처리
# =============================================================================

def process_files_parallel(
    file_paths: List[str],
    process_func: Callable[[str], Any],
    max_workers: Optional[int] = None
) -> List[Tuple[str, Union[Any, FileReadError]]]:
    """여러 파일을 병렬로 처리.

    Args:
        file_paths: 처리할 파일 경로 리스트
        process_func: 각 파일에 적용할 함수
        max_workers: 최대 워커 수 (None이면 CPU 코어 수 사용)

    Returns:
        (파일 경로, 처리 결과 또는 에러) 튜플 리스트
    """
    if max_workers is None:
        max_workers = processing_config.max_workers

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_func, fp): fp
            for fp in file_paths
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append((file_path, result))
            except FileReadError as e:
                logger.warning(f"File read error for {file_path}: {e}")
                results.append((file_path, e))
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
                results.append((file_path, FileReadError(f"Processing error: {e}")))

    return results


# =============================================================================
# 6. 히스토리 관리
# =============================================================================

def compress_message_history(
    messages: List[Dict[str, Any]],
    max_messages: int = 10
) -> List[Dict[str, Any]]:
    """대화 히스토리를 압축하여 컨텍스트 절약.

    Args:
        messages: 메시지 리스트
        max_messages: 유지할 최대 메시지 수

    Returns:
        압축된 메시지 리스트

    Note:
        실제 프로덕션에서는 DeepAgent의 SummarizationMiddleware가 처리합니다.
    """
    if len(messages) <= max_messages:
        return messages

    recent_messages = messages[-max_messages:]
    old_messages = messages[:-max_messages]

    if old_messages:
        summary_content = f"[이전 대화 요약: {len(old_messages)}개 메시지]\n"
        summary_content += "주요 논의 내용: "

        first_msg = old_messages[0].get("content", "")[:100]
        last_msg = old_messages[-1].get("content", "")[:100]
        summary_content += f"시작: {first_msg}... 종료: {last_msg}..."

        summary_msg = {
            "role": "system",
            "content": summary_content
        }
        return [summary_msg] + recent_messages

    return recent_messages


def estimate_token_count(text: str) -> int:
    """텍스트의 대략적인 토큰 수 추정 (평균 4자당 1토큰).

    Args:
        text: 텍스트

    Returns:
        추정 토큰 수
    """
    return len(text) // 4


def check_context_limit(text: str, limit: int = 200000) -> Tuple[bool, int]:
    """컨텍스트 한계 초과 여부 확인.

    Args:
        text: 확인할 텍스트
        limit: 토큰 한계 (기본 200K, 여유분 포함)

    Returns:
        (한계 초과 여부, 추정 토큰 수) 튜플
    """
    estimated = estimate_token_count(text)
    return estimated > limit, estimated
