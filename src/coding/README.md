# DeepAgentic Code Assistant

Python 코드 분석 및 리팩토링을 위한 지능형 에이전트입니다.

## 주요 기능

### 1. 영향도 분석 (Impact Analysis)

코드 변경이 프로젝트에 미치는 영향을 분석합니다.

#### SPEED 모드 (기본)
- **특징**: AST 파싱 기반 정적 분석
- **속도**: ~5초 (10k 라인 기준)
- **장점**: 빌드 환경 불필요, 빠른 피드백
- **단점**: 동적 코드에서 False Positive 가능

#### PRECISION 모드
- **특징**: LSP (Pyright) 기반 컴파일러 수준 분석
- **정확도**: 타입 체킹 및 정확한 참조 추적
- **요구사항**: Pyright 설치 및 올바른 빌드 환경
- **폴백**: 실패 시 자동으로 SPEED 모드 제안

### 2. 자율 코딩 및 복구 (Autonomous Coding & Recovery)

코드 변경 시 자동으로 검증하고 오류를 수정합니다.

**Self-Healing Loop:**
1. 코드 변경 적용 (`edit_file`)
2. 실행하여 검증 (`execute_python_code`)
3. 오류 발생 시 에러 분석
4. 타겟 수정 적용
5. 최대 3회 재시도
6. 3회 실패 시 사용자에게 보고

### 3. 테스트 자동 생성 (Test Generation)

변경된 코드에 대한 pytest 테스트를 자동 생성합니다.

**생성 기준:**
- AAA 패턴 (Arrange, Act, Assert)
- 정상 케이스, 엣지 케이스, 에러 케이스 커버
- 의미 있는 테스트 이름
- Google-style docstring 포함

### 4. 문서 동기화 (Documentation Sync)

코드 변경 시 관련 문서를 자동으로 업데이트합니다.

**대상:**
- Python Docstring (Google-style)
- README.md
- API 문서
- 코드 주석

### 5. 파일 삭제 (File Deletion)

파일을 안전하게 삭제합니다.

**보안 기능:**
- Human-in-the-Loop: 삭제 전 반드시 사용자 승인 필요
- 경로 검증: 현재 작업 디렉토리 내 파일만 삭제 가능
- 타입 확인: 파일만 삭제 가능 (디렉토리 삭제 차단)

**주의:** 삭제된 파일은 복구할 수 없습니다!

### 6. 프로젝트 디렉토리 변경 (Change Project Directory)

작업 중인 프로젝트 폴더를 동적으로 변경합니다.

**보안 기능:**
- WORKSPACE 제한: `.env`의 `WORKSPACE` 환경 변수로 정의된 루트 외부로 이동 불가
- 경로 검증: 상대/절대 경로 모두 지원하되, 항상 WORKSPACE 내부로만 제한
- 존재 여부 확인: 디렉토리가 실제로 존재하는지 검증

**사용 예:**
```python
# 상대 경로로 이동
change_project_directory("src/myproject")

# 절대 경로로 이동 (WORKSPACE 내부만)
change_project_directory("/home/ubuntu/DeepAgent/examples")

# 상위 디렉토리 이동 차단 (보안)
change_project_directory("../../etc")  # ❌ 거부됨
```

## 서브에이전트 (SubAgents)

### `speed-analyzer`
빠른 정적 분석 수행

### `precision-analyzer`
정밀한 LSP 기반 분석 (Pyright)

### `code-refactor`
코드 리팩토링 및 자가 복구

### `test-generator`
pytest 테스트 자동 생성

### `doc-sync`
문서 동기화

### `file-summarizer`
대용량 파일 요약

## 사용 방법

### 설치

#### 방법 1: uv로 설치 (가장 빠름, 권장) ⚡

```bash
cd src/coding

# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# dev 의존성 포함 설치
uv sync --extra dev

# CLI 실행
uv run python run_cli.py

# 또는 coding-assistant 명령어
uv run coding-assistant
```

#### 방법 2: pyproject.toml로 설치

```bash
cd src/coding

# 개발 모드로 설치 (소스 코드 변경 시 자동 반영)
pip install -e .

# dev 의존성 포함 설치
pip install -e ".[dev]"
```

#### 방법 2: requirements.txt로 설치

```bash
cd src/coding
pip install -r requirements.txt
```

#### 설치 후 명령어 사용

```bash
# uv 사용
uv run coding-assistant
# 또는
uv run python run_cli.py

# pip 설치 후
coding-assistant
# 또는
python run_cli.py
```

### 실행 방법

#### 1. LangGraph 서버 모드 (권장)

```bash
langgraph dev
```

웹 브라우저에서 `http://localhost:8123`로 접속하여 사용

#### 2. CLI 대화형 모드 (로컬 테스트)

```bash
python run_cli.py
```

터미널에서 직접 대화하며 테스트할 수 있습니다:

```
🧑 You: 현재 프로젝트 폴더 내 파이썬 프로그램 목록을 모두 보여줘

🤖 Assistant: [파일 목록 응답]
```

종료: `exit`, `quit`, 또는 `Ctrl+C`

### 환경 변수 설정

`.env` 파일에 다음 변수를 설정하세요:

```env
# OpenRouter API 사용 (권장)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=moonshotai/kimi-k2-0905  # 또는 다른 OpenRouter 모델

# 또는 OpenAI API 직접 사용
OPENAI_API_KEY=your_openai_api_key
MODEL=gpt-4o

# 작업 공간 루트 (프로젝트 디렉토리 변경 시 상위 제한)
WORKSPACE=/home/ubuntu/DeepAgent
```

**참고:** 기본 설정은 OpenRouter를 사용합니다. OpenAI API를 직접 사용하려면 `coding_agent.py`의 모델 설정을 수정하세요.

## 예제 사용 시나리오

### 시나리오 1: 함수 영향도 분석

```
User: "analyze_impact.py 파일의 process_data 함수를 수정하려고 합니다.
      영향도를 분석해주세요."

Agent:
1. SPEED 모드로 빠른 분석 수행
2. 호출자 목록 제공
3. 의존성 그래프 제공
4. 변경 시 주의사항 안내
```

### 시나리오 2: 코드 리팩토링

```
User: "utils.py의 calculate_total 함수를 리팩토링해주세요.
      리스트 컴프리헨션을 사용하도록 변경하고 테스트도 생성해주세요."

Agent:
1. utils.py 읽기
2. code-refactor 서브에이전트로 리팩토링
3. 변경 사항 검증 (실행)
4. test-generator로 테스트 생성
5. 테스트 실행 및 검증
6. doc-sync로 docstring 업데이트
```

### 시나리오 3: PRECISION 모드 폴백

```
User: "config.py의 타입 힌트를 정확히 분석해주세요."

Agent:
1. PRECISION 모드로 분석 시도
2. Pyright 빌드 에러 발생
3. 사용자에게 SPEED 모드 전환 제안
4. 승인 시 SPEED 모드로 분석 계속
```

### 시나리오 4: 프로젝트 디렉토리 변경

```
User: "examples/research 폴더로 이동해서 파일을 분석해줘"

Agent:
1. change_project_directory("examples/research") 실행
2. WORKSPACE 범위 내인지 검증
3. 디렉토리 변경 성공
4. 새 위치에서 파일 분석 수행

User: "../../etc로 이동해줘"

Agent:
❌ 보안 오류: WORKSPACE 외부로의 이동은 불가능합니다.
허용 범위: /home/ubuntu/DeepAgent
```

### 시나리오 5: 대용량 디렉토리 처리

```
User: "프로젝트를 /home/ubuntu/work/large_project로 변경해줘"

Agent: ✓ 프로젝트 디렉토리 변경 완료
       파일 개수: 1,245개

       ⚠️ 경고: 이 디렉토리에는 1,245개의 파일이 있습니다!
       파일 목록 조회 시 컨텍스트 오버플로우가 발생할 수 있습니다.

       권장사항:
       - 특정 파일만 직접 경로로 지정하여 읽기
       - glob 패턴으로 필요한 파일만 필터링 (예: *.py)
       - 하위 디렉토리로 더 좁게 이동

User: "src 폴더로 이동해줘"  (파일 개수 감소)

또는

User: "glob 패턴 '*.py'로 Python 파일만 찾아줘"  (필터링)
```

## 품질 기준

### 코드 스타일
- **PEP8** 준수
- **Google-style** Docstring
- 타입 힌트 적극 활용
- 명확하고 자기 문서화된 코드

### 테스트
- pytest 프레임워크 사용
- 최소 80% 커버리지 목표
- 단위 테스트 우선

## 아키텍처

```
coding_agent.py
├── Tools
│   ├── analyze_impact (SPEED/PRECISION)
│   ├── execute_python_code
│   ├── run_pytest
│   ├── delete_file (Human-in-the-Loop)
│   └── change_project_directory (WORKSPACE 제한)
├── SubAgents
│   ├── speed-analyzer
│   ├── precision-analyzer
│   ├── code-refactor
│   ├── test-generator
│   ├── doc-sync
│   └── file-summarizer
└── Backend
    └── FilesystemBackend (virtual_mode=True)
```

## Human-in-the-Loop 지점

다음 경우 사용자 승인을 요청합니다:

1. **파일 삭제**: 모든 `delete_file` 호출 (되돌릴 수 없는 파괴적 작업)
2. **PRECISION 모드 실패**: SPEED 모드 전환 제안
3. **대용량 파일 처리**: 5,000 라인 이상 파일
4. **대량 파일 수정**: 10개 이상 파일 변경

## 제한사항

- **언어**: Python만 지원
- **분석 깊이**: SPEED 모드는 단일 파일 내 분석
- **LSP 의존성**: PRECISION 모드는 Pyright 설치 필요
- **실행 환경**: 로컬 파일 시스템 접근 필요
- **컨텍스트 제한**: 대용량 디렉토리(100개 이상 파일)에서는 glob 패턴 사용 권장

## 대용량 프로젝트 작업 가이드

대규모 프로젝트에서 컨텍스트 오버플로우를 방지하려면:

1. **하위 디렉토리로 이동**: 작업 범위를 좁히기
   ```
   "src/api 폴더로 이동해줘"
   ```

2. **glob 패턴 사용**: 필요한 파일만 필터링
   ```
   "glob 패턴 '**/*.py'로 Python 파일만 찾아줘"
   ```

3. **직접 파일 경로 지정**: 특정 파일만 읽기
   ```
   "src/main.py 파일을 읽어줘"
   ```

4. **단계적 탐색**: 먼저 ls로 구조 파악 후 필요한 부분만 탐색
   ```
   Step 1: "현재 디렉토리의 폴더 구조를 보여줘"
   Step 2: "src 폴더로 이동해줘"
   Step 3: "Python 파일 목록 보여줘"
   ```

## 트러블슈팅

### Pyright 설치 오류

```bash
pip install pyright
```

### pytest 없음

```bash
pip install pytest
```

### 파일 권한 오류

FilesystemBackend는 `virtual_mode=True`로 설정되어 있어
현재 작업 디렉토리 외부로의 접근이 제한됩니다.

## 라이선스

이 프로젝트는 DeepAgent 0.2 아키텍처를 기반으로 합니다.

## 참고 문서

- [DeepAgent Blog](https://blog.langchain.com/deep-agents/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [PEP8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
