"""Performance Optimizations for DeepAgentic Code Assistant.

This module provides:
1. Context explosion prevention
2. LangSmith performance monitoring
3. File reading cache
4. Parallel processing utilities
5. History management
"""

import functools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 0. 컨텍스트 폭발 방지
# =============================================================================

# 파일 목록 최대 개수 제한
MAX_FILE_LIST = 50
MAX_FILE_SIZE_FOR_AUTO_READ = 10000  # 10KB


def limit_file_list(files: List[str], max_count: int = MAX_FILE_LIST) -> tuple[List[str], str]:
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


def truncate_large_content(content: str, max_lines: int = 1000) -> tuple[str, bool]:
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
# 1. LangSmith 성능 모니터링
# =============================================================================

def monitor_performance(func_name: str = None):
    """LangSmith와 함께 사용할 성능 모니터링 데코레이터.

    Args:
        func_name: 함수 이름 (없으면 자동 감지)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
# 2. 파일 읽기 캐싱
# =============================================================================

class FileCache:
    """파일 내용 캐싱 클래스."""

    def __init__(self, max_size: int = 100):
        """Initialize file cache.

        Args:
            max_size: 최대 캐시 항목 수
        """
        self._cache: Dict[str, tuple[str, float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, file_path: str) -> str | None:
        """캐시에서 파일 내용 가져오기.

        Args:
            file_path: 파일 경로

        Returns:
            캐시된 내용 또는 None
        """
        path = Path(file_path).resolve()
        key = str(path)

        if key in self._cache:
            content, cached_mtime = self._cache[key]

            # 파일이 수정되지 않았는지 확인
            try:
                current_mtime = path.stat().st_mtime
                if current_mtime == cached_mtime:
                    self.hits += 1
                    logger.debug(f"Cache hit: {file_path}")
                    return content
            except OSError:
                pass

        self.misses += 1
        return None

    def set(self, file_path: str, content: str) -> None:
        """파일 내용을 캐시에 저장.

        Args:
            file_path: 파일 경로
            content: 파일 내용
        """
        path = Path(file_path).resolve()
        key = str(path)

        try:
            mtime = path.stat().st_mtime

            # 캐시 크기 제한
            if len(self._cache) >= self.max_size:
                # 가장 오래된 항목 제거 (간단한 FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = (content, mtime)
            logger.debug(f"Cache set: {file_path}")

        except OSError:
            pass

    def clear(self) -> None:
        """캐시 전체 삭제."""
        self._cache.clear()
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
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


# 전역 파일 캐시 인스턴스
file_cache = FileCache(max_size=100)


# =============================================================================
# 3. 병렬 처리
# =============================================================================

def process_files_parallel(
    file_paths: List[str],
    process_func: Callable[[str], Any],
    max_workers: int = 4
) -> List[tuple[str, Any]]:
    """여러 파일을 병렬로 처리.

    Args:
        file_paths: 처리할 파일 경로 리스트
        process_func: 각 파일에 적용할 함수
        max_workers: 최대 워커 수

    Returns:
        (파일 경로, 처리 결과) 튜플 리스트
    """
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
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append((file_path, f"Error: {e}"))

    return results


# =============================================================================
# 4. 히스토리 관리
# =============================================================================

def compress_message_history(
    messages: List[Dict[str, Any]],
    max_messages: int = 10,
    summarize_threshold: int = 20
) -> List[Dict[str, Any]]:
    """대화 히스토리를 압축하여 컨텍스트 절약.

    Args:
        messages: 메시지 리스트
        max_messages: 유지할 최대 메시지 수
        summarize_threshold: 요약을 시작할 메시지 수

    Returns:
        압축된 메시지 리스트
    """
    if len(messages) <= summarize_threshold:
        return messages

    # 최근 메시지는 유지
    recent_messages = messages[-max_messages:]

    # 오래된 메시지는 요약 (실제로는 DeepAgent의 SummarizationMiddleware가 처리)
    old_messages = messages[:-max_messages]

    if old_messages:
        summary_msg = {
            "role": "system",
            "content": f"[이전 대화 요약: {len(old_messages)}개 메시지가 요약되었습니다.]"
        }
        return [summary_msg] + recent_messages

    return recent_messages


def estimate_token_count(text: str) -> int:
    """텍스트의 대략적인 토큰 수 추정.

    간단한 휴리스틱: 평균 4자당 1토큰

    Args:
        text: 텍스트

    Returns:
        추정 토큰 수
    """
    return len(text) // 4


def check_context_limit(text: str, limit: int = 200000) -> tuple[bool, int]:
    """컨텍스트 한계 초과 여부 확인.

    Args:
        text: 확인할 텍스트
        limit: 토큰 한계 (기본 200K, 여유분 포함)

    Returns:
        (한계 초과 여부, 추정 토큰 수) 튜플
    """
    estimated = estimate_token_count(text)
    return estimated > limit, estimated
