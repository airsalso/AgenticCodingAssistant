"""DeepAgentic Code Assistant - Python 코드 분석 및 리팩토링 에이전트.

이 모듈은 DeepAgent 아키텍처를 활용하여 다음 기능을 제공합니다:
- Python 코드 영향도 분석 (AST 기반 SPEED 모드, LSP 기반 PRECISION 모드)
- 자율 리팩토링 및 자가 치유 (최대 3회 재시도)
- 자동 테스트 생성 (pytest 기반)
- 문서 동기화 (docstring, README 등)

주요 최적화:
- 파일 및 분석 결과 캐싱으로 반복 작업 성능 향상
- 병렬 처리로 다중 파일 분석 속도 개선 (3개 이상 파일 시)
- 컨텍스트 폭발 방지 (파일 크기/개수 제한)
- 토큰 제한 체크 및 대용량 파일 요약
"""

import ast
import json
import logging
import os
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

import networkx as nx

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import _get_filesystem_tools
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from optimizations import (
    check_context_limit,
    file_cache,
    limit_file_list,
    monitor_performance,
    process_files_parallel,
    read_file_with_cache,
    truncate_large_content,
    MAX_FILE_LIST,
    MAX_FILE_SIZE_FOR_AUTO_READ,
)

# =============================================================================
# 설정 및 전역 변수
# =============================================================================

logger = logging.getLogger(__name__)

# 성능 최적화 상수
MAX_ANALYSIS_CACHE_SIZE = 1000
PARALLEL_THRESHOLD = 3
MAX_PARALLEL_WORKERS = 8
TRUNCATE_RESULT_LENGTH = 500

# 분석 결과 캐시 (FIFO 순서 보장을 위해 OrderedDict 사용)
_analysis_result_cache: OrderedDict[str, str] = OrderedDict()

# 그래프 캐시 (SPEED 모드용, FIFO 순서 보장)
_graph_cache: OrderedDict[str, Dict] = OrderedDict()

# 파일시스템 백엔드 (change_project_directory에서 업데이트)
_filesystem_backend: Optional[FilesystemBackend] = None


# =============================================================================
# 데이터 구조: Symbol 클래스
# =============================================================================

@dataclass
class Symbol:
    """코드 심볼 정보."""
    name: str
    type: str  # 'function', 'class', 'method', 'variable'
    file_path: str
    line: int
    column: int
    parent: Optional[str] = None  # 클래스의 경우 메서드

    def __hash__(self):
        return hash((self.file_path, self.name, self.line))

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return (self.file_path, self.name, self.line) == (other.file_path, other.name, other.line)


# =============================================================================
# 오류 처리: AnalysisError 클래스
# =============================================================================

class AnalysisError(Exception):
    """분석 오류 기본 클래스."""
    def __init__(self, message: str, fallback_mode: str = "SPEED"):
        super().__init__(message)
        self.fallback_mode = fallback_mode


# =============================================================================
# 그래프 분석 헬퍼 함수
# =============================================================================

def _build_call_graph(tree: ast.AST, file_path: str) -> nx.DiGraph:
    """호출 관계 그래프 구축.

    Args:
        tree: AST 트리
        file_path: 파일 경로

    Returns:
        호출 관계를 나타내는 방향 그래프
    """
    G = nx.DiGraph()
    current_function = None
    current_class = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            G.add_node(node.name, type='class', lineno=node.lineno, file=file_path)
            current_class = node.name
            current_function = None

        elif isinstance(node, ast.FunctionDef):
            node_name = node.name
            if current_class:
                # 메서드인 경우
                node_name = f"{current_class}.{node.name}"
                G.add_node(node_name, type='method', lineno=node.lineno, file=file_path, parent=current_class)
            else:
                # 일반 함수
                G.add_node(node_name, type='function', lineno=node.lineno, file=file_path)

            current_function = node_name

        elif isinstance(node, ast.Call) and current_function:
            # 함수 호출 분석
            if isinstance(node.func, ast.Name):
                callee = node.func.id
                G.add_edge(current_function, callee)
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
                G.add_edge(current_function, callee)

    return G


def _resolve_symbol(tree: ast.AST, target_name: str, file_path: str) -> Optional[Symbol]:
    """정확한 심볼 찾기.

    Args:
        tree: AST 트리
        target_name: 찾을 심볼 이름
        file_path: 파일 경로

    Returns:
        찾은 Symbol 객체 또는 None
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == target_name:
            return Symbol(
                name=node.name,
                type='function',
                file_path=file_path,
                line=node.lineno,
                column=node.col_offset
            )
        elif isinstance(node, ast.ClassDef) and node.name == target_name:
            return Symbol(
                name=node.name,
                type='class',
                file_path=file_path,
                line=node.lineno,
                column=node.col_offset
            )
    return None


def _find_callers_with_depth(graph: nx.DiGraph, target: str, max_depth: int = 3) -> List[str]:
    """깊이 제한이 있는 그래프 순회로 호출자 찾기.

    Args:
        graph: 호출 그래프
        target: 대상 노드
        max_depth: 최대 탐색 깊이

    Returns:
        호출자 목록
    """
    if target not in graph:
        return []

    callers = []
    try:
        # BFS로 역방향 탐색 (predecessors)
        predecessors = nx.bfs_tree(graph.reverse(), target, depth_limit=max_depth)
        callers = list(predecessors.nodes())
        # 자기 자신 제외
        if target in callers:
            callers.remove(target)
    except nx.NetworkXError:
        pass

    return callers


def _get_or_build_call_graph(file_path: str) -> nx.DiGraph:
    """캐시에서 그래프를 가져오거나 새로 빌드.

    Args:
        file_path: 파일 경로

    Returns:
        호출 그래프
    """
    path = Path(file_path)
    if not path.exists():
        return nx.DiGraph()

    # 캐시 확인
    cache_key = str(path.resolve())
    mtime = path.stat().st_mtime

    if cache_key in _graph_cache:
        cached_data = _graph_cache[cache_key]
        if cached_data['mtime'] == mtime:
            logger.info(f"Graph cache hit: {cache_key}")
            return cached_data['graph']

    # 새로 빌드
    try:
        source = read_file_with_cache(file_path)
        if source.startswith("Error:"):
            return nx.DiGraph()

        tree = ast.parse(source, filename=file_path)
        graph = _build_call_graph(tree, file_path)

        # 캐시 저장
        _graph_cache[cache_key] = {
            'graph': graph,
            'mtime': mtime
        }

        # 캐시 크기 제한 (FIFO: 가장 오래된 항목 제거)
        if len(_graph_cache) > MAX_ANALYSIS_CACHE_SIZE:
            _graph_cache.popitem(last=False)

        logger.info(f"Graph built and cached: {cache_key}")
        return graph

    except Exception as e:
        logger.error(f"Error building graph for {file_path}: {e}")
        return nx.DiGraph()


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
    """SPEED 모드: 그래프 기반 정적 분석 (깊이 제한 포함).

    Args:
        file_path: 분석할 Python 파일 경로
        target: 분석 대상 함수 또는 클래스 이름

    Returns:
        영향도 분석 결과
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"

        # 캐시를 활용한 파일 읽기
        source = read_file_with_cache(file_path)
        if source.startswith("Error:"):
            return source

        tree = ast.parse(source, filename=file_path)

        # Symbol 해결
        symbol = _resolve_symbol(tree, target, file_path)
        if not symbol:
            # 기존 방식으로 폴백
            definitions = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions[node.name] = {'type': type(node).__name__, 'lineno': node.lineno}
            return f"Warning: '{target}' not found in {file_path}. Found: {list(definitions.keys())}"

        # 그래프 기반 분석
        graph = _get_or_build_call_graph(file_path)

        # 깊이 제한이 있는 역방향 의존성 찾기 (depth=3)
        callers = _find_callers_with_depth(graph, target, max_depth=3)

        # 정방향 의존성 (이 타겟이 무엇을 호출하는지)
        dependencies = []
        if target in graph:
            dependencies = list(graph.successors(target))

        # Import 정보 수집
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend([f"{module}.{alias.name}" for alias in node.names])

        # 결과 포맷팅
        result = f"# Impact Analysis (SPEED Mode - Graph-Based)\n\n"
        result += f"**Target:** `{symbol.name}` ({symbol.type}) at line {symbol.line}\n\n"
        result += f"**File:** {file_path}\n\n"

        if callers:
            result += f"## Potential Callers (depth ≤ 3): {len(callers)}\n\n"
            for caller in callers[:20]:  # 최대 20개만 표시
                if caller in graph:
                    node_data = graph.nodes[caller]
                    result += f"- `{caller}` ({node_data.get('type', 'unknown')}) at line {node_data.get('lineno', '?')}\n"
                else:
                    result += f"- `{caller}`\n"
            if len(callers) > 20:
                result += f"- ... and {len(callers) - 20} more\n"
        else:
            result += "## No callers found within depth 3\n\n"

        if dependencies:
            result += f"\n## Dependencies ({len(dependencies)}):\n\n"
            for dep in dependencies[:20]:
                result += f"- `{dep}`\n"
            if len(dependencies) > 20:
                result += f"- ... and {len(dependencies) - 20} more\n"

        if imports:
            result += f"\n## File Imports ({len(imports)}):\n\n"
            for imp in imports[:10]:
                result += f"- {imp}\n"
            if len(imports) > 10:
                result += f"- ... and {len(imports) - 10} more\n"

        return result

    except SyntaxError as e:
        return f"Syntax Error in {file_path}: {e}"
    except Exception as e:
        logger.error(f"Error in SPEED mode analysis: {e}")
        return f"Error analyzing {file_path}: {e}"


# =============================================================================
# LSP 분석기 클래스
# =============================================================================

class LSPAnalyzer:
    """LSP 기반 정밀 분석기 (Pyright 사용)."""

    def __init__(self):
        self._initialized = False
        self._pyright_available = None

    def is_ready(self) -> bool:
        """LSP 서버가 준비되었는지 확인.

        Returns:
            Pyright가 설치되어 있고 사용 가능하면 True
        """
        if self._pyright_available is not None:
            return self._pyright_available

        try:
            result = subprocess.run(
                ['pyright', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._pyright_available = result.returncode == 0
            if self._pyright_available:
                logger.info("Pyright LSP server is available")
                self._initialized = True
            return self._pyright_available
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._pyright_available = False
            return False

    def find_references(self, file_path: str, symbol: Symbol) -> List[str]:
        """심볼의 참조를 찾습니다 (현재는 타입 체크로 제한).

        Args:
            file_path: 파일 경로
            symbol: 찾을 심볼

        Returns:
            참조 위치 목록
        """
        if not self.is_ready():
            raise AnalysisError(
                "LSP server not ready. Pyright is not installed or not available.",
                fallback_mode="SPEED"
            )

        try:
            result = subprocess.run(
                ['pyright', '--outputjson', file_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            data = json.loads(result.stdout)
            diagnostics = data.get('generalDiagnostics', [])

            # 현재는 타입 에러를 반환 (실제 LSP 참조 찾기는 더 복잡한 구현 필요)
            references = []
            for diag in diagnostics:
                if symbol.name in diag.get('message', ''):
                    line = diag.get('range', {}).get('start', {}).get('line', 0)
                    references.append(f"Line {line}: {diag.get('message', '')}")

            return references

        except json.JSONDecodeError:
            raise AnalysisError("Failed to parse Pyright output")
        except subprocess.TimeoutExpired:
            raise AnalysisError("Pyright analysis timed out")


# 전역 LSP 분석기 인스턴스
_lsp_analyzer = LSPAnalyzer()


def _analyze_precision_mode(file_path: str, target: str) -> str:
    """PRECISION 모드: LSP 기반 정밀 분석 (자동 폴백 포함).

    Args:
        file_path: 분석할 Python 파일 경로
        target: 분석 대상 함수 또는 클래스 이름

    Returns:
        영향도 분석 결과
    """
    try:
        # LSP 준비 확인
        if not _lsp_analyzer.is_ready():
            raise AnalysisError(
                "LSP server not ready. Pyright is not installed or not available.\n"
                "Install with: pip install pyright",
                fallback_mode="SPEED"
            )

        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"

        # Symbol 해결
        source = read_file_with_cache(file_path)
        if source.startswith("Error:"):
            return source

        tree = ast.parse(source, filename=file_path)
        symbol = _resolve_symbol(tree, target, file_path)

        if not symbol:
            raise AnalysisError(f"Symbol '{target}' not found in {file_path}")

        # LSP 참조 찾기
        references = _lsp_analyzer.find_references(file_path, symbol)

        # Pyright 타입 체크 실행
        result = subprocess.run(
            ['pyright', '--outputjson', file_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        data = json.loads(result.stdout)
        diagnostics = data.get('generalDiagnostics', [])

        result_text = f"# Impact Analysis (PRECISION Mode - LSP-Based)\n\n"
        result_text += f"**Target:** `{symbol.name}` ({symbol.type}) at line {symbol.line}\n\n"
        result_text += f"**File:** {file_path}\n\n"

        # 타입 체크 상태
        if diagnostics:
            errors = [d for d in diagnostics if d.get('severity') == 'error']
            warnings = [d for d in diagnostics if d.get('severity') == 'warning']

            result_text += f"**Type Checking:** {len(errors)} errors, {len(warnings)} warnings\n\n"

            if errors:
                result_text += "## Type Errors:\n\n"
                for err in errors[:10]:
                    line_num = err.get('range', {}).get('start', {}).get('line', '?')
                    message = err.get('message', '')
                    result_text += f"- Line {line_num}: {message}\n"
                if len(errors) > 10:
                    result_text += f"- ... and {len(errors) - 10} more\n"
        else:
            result_text += "**Type Checking:** ✓ No type errors\n\n"

        # 참조 정보
        if references:
            result_text += f"\n## References ({len(references)}):\n\n"
            for ref in references[:10]:
                result_text += f"- {ref}\n"
            if len(references) > 10:
                result_text += f"- ... and {len(references) - 10} more\n"

        return result_text

    except AnalysisError as e:
        # 자동 폴백
        logger.warning(f"PRECISION mode failed: {e}. Falling back to {e.fallback_mode} mode.")
        return (
            f"⚠️ PRECISION mode failed: {e}\n\n"
            f"**Auto-fallback to {e.fallback_mode} mode:**\n\n"
            + _analyze_speed_mode(file_path, target)
        )
    except subprocess.TimeoutExpired:
        logger.warning("Pyright timed out. Falling back to SPEED mode.")
        return (
            "⚠️ Pyright analysis timed out (>30s)\n\n"
            "**Auto-fallback to SPEED mode:**\n\n"
            + _analyze_speed_mode(file_path, target)
        )
    except json.JSONDecodeError:
        logger.warning("Failed to parse Pyright output. Falling back to SPEED mode.")
        return (
            "⚠️ Failed to parse Pyright output\n\n"
            "**Auto-fallback to SPEED mode:**\n\n"
            + _analyze_speed_mode(file_path, target)
        )
    except Exception as e:
        logger.error(f"Unexpected error in PRECISION mode: {e}")
        return (
            f"⚠️ Error in PRECISION mode: {e}\n\n"
            "**Auto-fallback to SPEED mode:**\n\n"
            + _analyze_speed_mode(file_path, target)
        )


def _cached_analysis(
    file_path: str,
    function_or_class: str,
    mode: Literal["SPEED", "PRECISION"] = "SPEED",
) -> str:
    """내부: 캐시를 활용한 영향도 분석.

    Args:
        file_path: 분석할 Python 파일 경로
        function_or_class: 분석 대상 함수 또는 클래스 이름
        mode: 분석 모드

    Returns:
        영향도 분석 결과
    """
    # 캐시 키: 절대경로 + 타겟 + 모드 + 수정시간
    try:
        path = Path(file_path).resolve()
        cache_key = f"{path}:{function_or_class}:{mode}"

        if path.exists():
            mtime = path.stat().st_mtime
            cache_key += f":{mtime}"
    except OSError:
        # 경로 해결 실패 시 원본 경로 사용
        cache_key = f"{file_path}:{function_or_class}:{mode}"

    # 캐시 확인
    if cache_key in _analysis_result_cache:
        logger.info(f"Analysis cache hit: {cache_key}")
        return f"[CACHED RESULT]\n\n{_analysis_result_cache[cache_key]}"

    # 분석 실행
    result = (
        _analyze_speed_mode(file_path, function_or_class)
        if mode == "SPEED"
        else _analyze_precision_mode(file_path, function_or_class)
    )

    # 캐시 크기 제한 (FIFO: 가장 오래된 항목 제거)
    if len(_analysis_result_cache) >= MAX_ANALYSIS_CACHE_SIZE:
        _analysis_result_cache.popitem(last=False)

    _analysis_result_cache[cache_key] = result
    logger.info(f"Analysis result cached: {cache_key}")

    return result


@tool
def analyze_impact_cached(
    file_path: str,
    function_or_class: str,
    mode: Literal["SPEED", "PRECISION"] = "SPEED",
) -> str:
    """캐시를 활용한 영향도 분석. 동일한 파일/타겟/모드 조합은 캐시에서 반환.

    Args:
        file_path: 분석할 Python 파일 경로
        function_or_class: 분석 대상 함수 또는 클래스 이름
        mode: 분석 모드 ('SPEED': 정적 분석, 'PRECISION': LSP 분석)

    Returns:
        영향도 분석 결과 (캐시된 결과 또는 새로운 분석 결과)
    """
    return _cached_analysis(file_path, function_or_class, mode)


@tool
def analyze_multiple_files(
    file_paths: List[str],
    function_or_class: str,
    mode: Literal["SPEED", "PRECISION"] = "SPEED"
) -> str:
    """여러 파일을 병렬로 분석합니다.

    Args:
        file_paths: 분석할 Python 파일 경로 리스트
        function_or_class: 분석 대상 함수 또는 클래스 이름
        mode: 분석 모드

    Returns:
        모든 파일의 분석 결과 통합
    """
    if not file_paths:
        return "Error: No file paths provided"

    # 파일 개수가 적으면 순차 처리
    if len(file_paths) <= PARALLEL_THRESHOLD:
        results = []
        for fp in file_paths:
            result = _cached_analysis(fp, function_or_class, mode)
            results.append(result)

        output = f"# Multiple File Analysis Results ({len(file_paths)} files)\n\n"
        for i, (fp, result) in enumerate(zip(file_paths, results), 1):
            output += f"## {i}. {fp}\n\n{result}\n\n"
            output += "---\n\n"
        return output

    # 병렬 처리
    start_time = time.time()
    results = process_files_parallel(
        file_paths,
        lambda fp: _cached_analysis(fp, function_or_class, mode),
        max_workers=min(len(file_paths), MAX_PARALLEL_WORKERS)
    )
    elapsed = time.time() - start_time

    # 결과 정렬 (원본 순서 유지)
    file_to_result = {fp: result for fp, result in results}
    ordered_results = [(fp, file_to_result.get(fp, "Error: No result")) for fp in file_paths]

    # 성공/실패 분류
    successful = [(fp, r) for fp, r in ordered_results if not r.startswith("Error")]
    failed = [(fp, r) for fp, r in ordered_results if r.startswith("Error")]

    # 결과 포맷팅
    output = f"# Multiple File Analysis Results\n\n"
    output += f"**Target:** {function_or_class}\n"
    output += f"**Mode:** {mode}\n"
    output += f"**Files analyzed:** {len(successful)}/{len(file_paths)}\n"
    output += f"**Time elapsed:** {elapsed:.2f}s\n\n"

    if failed:
        output += f"## Failed Analyses ({len(failed)} files):\n\n"
        for fp, error in failed[:5]:
            output += f"- {fp}: {error}\n"
        if len(failed) > 5:
            output += f"- ... and {len(failed) - 5} more\n"
        output += "\n"

    # 상위 5개 결과만 상세 표시
    output += f"## Successful Analyses (showing top 5 of {len(successful)}):\n\n"
    for fp, result in successful[:5]:
        output += f"### {fp}\n{result}\n\n---\n\n"

    if len(successful) > 5:
        output += f"*({len(successful) - 5} more results available)*\n"

    return output


@tool
def get_cache_stats() -> str:
    """캐시 통계를 반환합니다.

    Returns:
        파일 캐시 및 분석 결과 캐시의 통계 정보
    """
    file_stats = file_cache.stats()
    analysis_stats = {
        "size": len(_analysis_result_cache),
        "max_size": MAX_ANALYSIS_CACHE_SIZE
    }

    output = "# Cache Statistics\n\n"
    output += "## File Cache\n\n"
    output += f"```json\n{json.dumps(file_stats, indent=2)}\n```\n\n"
    output += "## Analysis Result Cache\n\n"
    output += f"```json\n{json.dumps(analysis_stats, indent=2)}\n```\n\n"

    return output


@tool
def analyze_project(
    target_function: str,
    file_pattern: str = "**/*.py",
    mode: Literal["SPEED", "PRECISION"] = "SPEED"
) -> str:
    """프로젝트 전체를 병렬로 분석합니다.

    Args:
        target_function: 분석 대상 함수 또는 클래스 이름
        file_pattern: 검색할 파일 패턴 (glob 형식)
        mode: 분석 모드

    Returns:
        프로젝트 분석 결과 요약
    """
    # 파일 검색
    cwd = Path.cwd()
    files = list(cwd.glob(file_pattern))
    file_paths = [str(f) for f in files if f.is_file()]

    if not file_paths:
        return f"No files found matching pattern: {file_pattern}"

    # 컨텍스트 폭발 방지
    limited_files, warning = limit_file_list(file_paths, max_count=50)

    # 병렬 분석
    start_time = time.time()
    results = process_files_parallel(
        limited_files,
        lambda fp: _cached_analysis(fp, target_function, mode),
        max_workers=min(len(limited_files), MAX_PARALLEL_WORKERS)
    )
    elapsed = time.time() - start_time

    # 결과 집계
    successful = [r for r in results if not r[1].startswith("Error")]
    failed = [r for r in results if r[1].startswith("Error")]

    output = f"# Project Analysis Results\n\n"
    output += f"**Target:** {target_function}\n"
    output += f"**Mode:** {mode}\n"
    output += f"**Pattern:** {file_pattern}\n"
    output += f"**Files analyzed:** {len(successful)}/{len(files)} (completed in {elapsed:.2f}s)\n\n"

    if warning:
        output += f"{warning}\n\n"

    if failed:
        output += f"**Failed analyses:** {len(failed)} files\n"
        for fp, error in failed[:5]:
            output += f"- {fp}: {error[:100]}...\n"
        if len(failed) > 5:
            output += f"- ... and {len(failed) - 5} more\n"
        output += "\n"

    # 상위 5개 결과만 요약
    output += f"## Top 5 Results:\n\n"
    for fp, result in successful[:5]:
        # 결과를 제한
        truncated = (
            result[:TRUNCATE_RESULT_LENGTH] + "..."
            if len(result) > TRUNCATE_RESULT_LENGTH
            else result
        )
        output += f"### {fp}\n{truncated}\n\n"

    return output


# =============================================================================
# 도구: 외부 정보 검색 (Tavily)
# =============================================================================

@tool
def search_web(
    query: str,
    max_results: int = 5,
) -> str:
    """Tavily를 사용하여 웹에서 최신 정보를 검색합니다.

    내부 파일 시스템에 없는 외부 정보가 필요할 때 사용하세요:
    - 최신 라이브러리 API 문서
    - 프로그래밍 모범 사례
    - 버그 해결 방법
    - 기술 트렌드 및 비교

    Args:
        query: 검색할 쿼리 문자열
        max_results: 반환할 최대 결과 개수 (기본값: 5)

    Returns:
        검색 결과 요약 (제목, URL, 내용 스니펫 포함)
    """
    try:
        # Tavily API 키가 환경 변수에 설정되어 있는지 확인
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            return (
                "Error: TAVILY_API_KEY environment variable not set.\n"
                "Please set your Tavily API key to use web search.\n"
                "Get your API key at: https://tavily.com"
            )

        # TavilySearchResults 도구 초기화
        tavily_tool = TavilySearchResults(
            max_results=max_results,
            api_key=tavily_api_key,
            search_depth="advanced",  # 더 깊은 검색 수행
            include_answer=True,      # AI 생성 답변 포함
            include_raw_content=False, # 원본 콘텐츠 제외 (토큰 절약)
        )

        # 검색 실행
        results = tavily_tool.invoke({"query": query})

        if not results:
            return f"No results found for query: {query}"

        # 결과 포맷팅
        output = f"# Web Search Results for: {query}\n\n"

        # AI 생성 답변이 있으면 먼저 표시
        if isinstance(results, list) and len(results) > 0:
            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    output += f"## Result {i}\n\n"

                    if "url" in result:
                        output += f"**URL:** {result['url']}\n\n"

                    if "title" in result:
                        output += f"**Title:** {result['title']}\n\n"

                    if "content" in result:
                        # 내용을 적절한 길이로 제한
                        content = result["content"]
                        if len(content) > 500:
                            content = content[:500] + "..."
                        output += f"**Content:**\n{content}\n\n"

                    if "score" in result:
                        output += f"*Relevance: {result['score']:.2f}*\n\n"

                    output += "---\n\n"

        return output

    except ImportError:
        return (
            "Error: tavily-python package not installed.\n"
            "Install with: pip install tavily-python langchain-community"
        )
    except Exception as e:
        return f"Error performing web search: {e}"


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

# delete_file 도구는 이제 파일시스템 미들웨어에서 제공됨 (fs_tools에 포함)


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

        # FilesystemBackend의 cwd도 업데이트 (중요!)
        if _filesystem_backend is not None:
            _filesystem_backend.cwd = new_dir

        # 현재 작업 디렉토리 변경
        os.chdir(new_dir)

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

SPEED_ANALYZER_PROMPT = """You are a speed-focused code analyzer using AST parsing.

Analyze Python code dependencies quickly and return:
- Callers (functions/classes that call the target)
- Dependencies (what the target calls)
- Import statements

Provide results within seconds for rapid feedback."""

speed_analyzer_agent = {
    "name": "speed-analyzer",
    "description": "Fast static analysis of Python code dependencies using AST parsing. Use when quick analysis is needed without full type checking.",
    "system_prompt": SPEED_ANALYZER_PROMPT,
    "tools": [analyze_impact],
}


PRECISION_ANALYZER_PROMPT = """You are a precision-focused code analyzer using Pyright LSP.

Perform compiler-level type checking and provide:
- Type errors and warnings
- Accurate type information
- Dependency validation

If Pyright fails, immediately suggest SPEED mode fallback."""

precision_analyzer_agent = {
    "name": "precision-analyzer",
    "description": "Precise LSP-based analysis using Pyright. Use when accuracy is critical and build environment is ready. Falls back to SPEED mode on failure.",
    "system_prompt": PRECISION_ANALYZER_PROMPT,
    "tools": [analyze_impact],
}


CODE_REFACTOR_PROMPT = """You are an expert Python code refactorer with self-healing.

Process:
1. Read target file(s)
2. Apply changes using edit_file
3. Validate by executing code
4. Self-heal on errors (max 3 attempts)

Standards:
- PEP8 compliance
- Google-style docstrings
- Preserve existing style
- Test before finalizing"""

code_refactor_agent = {
    "name": "code-refactor",
    "description": "Expert at refactoring Python code with self-healing capabilities. Automatically fixes compilation errors up to 3 times.",
    "system_prompt": CODE_REFACTOR_PROMPT,
    "tools": [execute_python_code],
}


TEST_GENERATOR_PROMPT = """You are a pytest test generator specialist.

Generate comprehensive unit tests covering:
- Normal cases, edge cases, error cases
- AAA pattern (Arrange, Act, Assert)
- Meaningful test names with docstrings

Process: Read code → Generate tests → Run tests → Report results"""

test_generator_agent = {
    "name": "test-generator",
    "description": "Generates comprehensive pytest unit tests for Python code. Automatically validates generated tests.",
    "system_prompt": TEST_GENERATOR_PROMPT,
    "tools": [execute_python_code, run_pytest],
}


DOC_SYNC_PROMPT = """You are a documentation synchronization specialist.

Keep documentation in sync with code changes:
- Python docstrings (Google-style)
- README.md files
- API documentation
- Code comments

Process: Analyze needs → Propose changes → Update files → Verify consistency"""

doc_sync_agent = {
    "name": "doc-sync",
    "description": "Synchronizes documentation with code changes. Updates docstrings, README, and API docs.",
    "system_prompt": DOC_SYNC_PROMPT,
    "tools": [],
}


FILE_SUMMARIZER_PROMPT = """You are a file content summarizer for large files exceeding token limits.

Extract and summarize:
- Main functions and classes
- Key logic flows
- Important dependencies

Process: Read portions → Extract key info → Save summary → Report location

Use structured Markdown format."""

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

### 1. Impact Analysis
Two modes for analyzing code change impact:
- **SPEED Mode** (default): AST-based static analysis, completes in ~5s for 10k lines
- **PRECISION Mode**: Pyright LSP type checking, requires build environment

Tools:
- `analyze_impact_cached`: Single file with caching
- `analyze_multiple_files`: Parallel analysis (3+ files use 8 workers)
- `analyze_project`: Scans entire project with glob patterns
- `get_cache_stats`: View cache performance

If PRECISION fails, immediately suggest SPEED mode fallback.

### 2. Web Search (Tavily Integration)
- Use `search_web` to fetch external information when internal files are insufficient
- Ideal for:
  - Latest library API documentation
  - Programming best practices
  - Bug fixes and troubleshooting
  - Technology trends and comparisons
  - Framework-specific solutions
- Always check internal files first before searching externally
- Requires TAVILY_API_KEY environment variable

### 3. Autonomous Refactoring
- Make changes with edit_file
- Validate with execute_python_code
- Self-heal on errors (max 3 attempts)
- Report failures to user after 3 retries

### 4. Test Generation & Documentation
- Generate pytest tests for significant changes
- Sync docstrings, README, and API docs after changes
- Use Google-style docstrings

### 5. Project Directory Management
- Use `change_project_directory` to switch projects
- Security: Limited to WORKSPACE scope
- Updates all file operations context

## Workflow

**Before changes:** Understand context → Run impact analysis → Search web if needed → Create TODO for complex tasks
**During changes:** Incremental edits → Test each step → Use subagents for complex work
**After changes:** Generate tests → Update docs → Validate

## Quality Standards
- PEP8 compliance
- Type hints where beneficial
- Google-style docstrings
- Preserve existing style

## Human-in-the-Loop
Request approval for:
- PRECISION→SPEED mode switch
- Files >5000 lines
- Changes to >10 files
- Destructive operations
"""


# =============================================================================
# Agent 설정 및 생성
# =============================================================================

workspace_root = os.environ.get("WORKSPACE", os.getcwd())

model = ChatOpenAI(
    model=os.environ.get("MODEL", "moonshotai/kimi-k2-0905"),
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
    temperature=0,
)

_filesystem_backend = FilesystemBackend(
    root_dir=workspace_root,
    virtual_mode=True,
)

# 파일시스템 도구 생성 (ls, read_file, write_file, edit_file, glob, grep)
fs_tools = _get_filesystem_tools(_filesystem_backend)

agent = create_deep_agent(
    model=model,
    tools=[
        *fs_tools,  # 파일시스템 도구 추가 (ls, read_file, write_file, edit_file, glob, grep, delete_file)
        analyze_impact,
        analyze_impact_cached,
        analyze_multiple_files,
        get_cache_stats,
        analyze_project,
        execute_python_code,
        run_pytest,
        change_project_directory,
        search_web,
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
        "analyze_impact": False,
        "read_file": False,
        "write_file": False,
        "edit_file": False,
        "delete_file": True,  # 파괴적 작업은 반드시 승인 필요 (fs_tools에서 제공)
        "change_project_directory": False,  # WORKSPACE 내에서만 허용
    },
)


graph = agent  # LangGraph export
