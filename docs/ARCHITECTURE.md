# CodingAgent Architecture Documentation

이 문서는 DeepAgentic Code Assistant의 전체 아키텍처를 설명합니다.

## 📊 다이어그램 파일

- **DrawIO**: [`coding_agent_architecture.drawio`](coding_agent_architecture.drawio)
  - [Draw.io](https://app.diagrams.net)에서 열어서 편집 가능
  - XML 기반 형식
 ![Uploading image.png…]()

    

## 🏗️ 아키텍처 레이어

### 1. User Interface Layer (사용자 인터페이스 계층)

사용자와 시스템 간의 상호작용을 담당하는 계층입니다.

**구성 요소:**
- CLI (run_cli.py): 대화형 커맨드라인 인터페이스
- LangGraph Server: HTTP API 및 스트리밍 지원
- API Interface: 외부 애플리케이션 통합

### 2. Main Agent Layer (메인 에이전트 계층)

**DeepAgent Core**: LangGraph 기반 상태 그래프 실행, 도구 오케스트레이션

**Middleware Stack**:
- TodoListMiddleware: 작업 목록 관리
- FilesystemMiddleware: 파일 시스템 도구 제공
- SubAgentMiddleware: 서브에이전트 스폰
- SummarizationMiddleware: 컨텍스트 요약
- AnthropicPromptCachingMiddleware: 프롬프트 캐싱

**Backend Layer**:
- FilesystemBackend: 실제 파일 시스템
- StateBackend: LangGraph 상태 저장
- StoreBackend: 영구 저장소
- CompositeBackend: 하이브리드 라우팅

### 3. Tools Layer (도구 계층)

**Analysis Tools**: analyze_impact, analyze_impact_cached, analyze_multiple_files, analyze_project

**Execution Tools**: execute_python_code, run_pytest, search_web

**Filesystem Tools**: ls, read_file, write_file, edit_file, glob, grep, delete_file

**Project Tools**: change_project_directory, get_cache_stats

### 4. SubAgents Layer (서브에이전트 계층)

- Speed Analyzer: AST 기반 정적 분석 (~5s)
- Precision Analyzer: Pyright LSP 타입 체킹 (자동 폴백)
- Code Refactor: 자가 치유 (최대 3회)
- Test Generator: Pytest 테스트 자동 생성
- Doc Sync: 문서 동기화
- File Summarizer: 대용량 파일 요약

### 5. Performance Optimizations Layer (성능 최적화)

**Caching**: File cache (LRU), Analysis cache, Graph cache

**Parallel Processing**: ThreadPoolExecutor (8 workers, 3+ files)

**Context Control**: 파일 제한 (50개), 자동 잘라내기, 요약

**Monitoring**: 성능 추적, LangSmith 통합, 캐시 통계

## 🔄 Analysis Flow (SPEED Mode)

```
User Request → Cache Check → AST Parsing → Graph Building →
Dependency Analysis → Result Formatting → Cache Store → Return Result
```

## 🎯 설계 원칙

1. **모듈화**: 명확한 책임 분리, 플러그형 백엔드
2. **성능**: 다층 캐싱, 병렬 처리, 컨텍스트 최적화
3. **자율성**: 서브에이전트, 자가 치유, 자동 폴백
4. **확장성**: 도구/서브에이전트/백엔드 추가 용이

## 📈 성능 특성

- 파일 캐시 히트율: 70-80%
- 분석 캐시 히트율: 60-70%
- 병렬 처리: 3개 파일 ~2.5배, 10개 파일 ~6배 빠름
- SPEED 모드: 100-500ms (소형), 3-5s (대형)
- PRECISION 모드: 5-15s

## 🔒 보안

- 경로 정규화 및 트래버설 차단
- WORKSPACE 제한
- 승인 필요 작업 (delete_file)
- 타임아웃 및 샌드박싱

---

**Version**: 0.2.0
