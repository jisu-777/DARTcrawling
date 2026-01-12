# DART 임원 정보 크롤링 및 분석 도구

DART(전자공시시스템)에서 기업 보고서를 가져와 임원 정보를 추출하고 AI를 활용하여 정규화된 형식으로 변환하는 도구입니다.

## 주요 기능

- **3단계 파이프라인**: 수집 → LLM 매핑 → 병합으로 분리하여 성능 및 안정성 향상
- **네트워크 I/O와 CPU 파싱 분리**: httpx 비동기 HTTP + ProcessPoolExecutor로 병렬 효율 극대화
- **배치 처리**: LLM 요청을 배치로 묶어 처리하여 효율성 향상
- **체크포인트 및 재시작**: 중단 후 재시작 가능
- DART OpenAPI를 통한 기업 보고서 자동 수집
- XML/HTML에서 임원 정보 테이블 자동 추출
- AI를 활용한 임원 정보 정규화 및 매핑
- 교수 정보 자동 추출 (학교, 학과, 직함)
- 엑셀 파일로 결과 출력

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd DARTcrawling
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```env
# DART API 설정
DART_API_KEY=your_dart_api_key
DART_API_BASE_URL=https://opendart.fss.or.kr/api

# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://your-openai-base-url.com/chat/completions

# OpenAI 모델 설정 (선택적, 기본값: azure.gpt-4o-mini)
OPENAI_MODEL=azure.gpt-4o-mini
```

### DART API 키 발급

1. [DART OpenAPI](https://opendart.fss.or.kr/)에 접속
2. 회원가입 후 API 키 발급

### OpenAI API 설정

- 내부 서버를 사용하는 경우 `OPENAI_API_BASE`에 베이스 URL 설정
- 사용 가능한 모델은 API 제공자에 따라 다를 수 있습니다

## 입력 파일 형식

`input.xlsx` 파일을 생성하고 다음 컬럼을 포함하세요:

| corp_code | corp_name | stock_code |
|-----------|-----------|------------|
| 00126380  | 삼성전자  | 005930     |
| 00258801  | 카카오    | 035720     |

- **corp_code**: DART 기업 고유번호 (8자리 숫자)
- **corp_name**: 회사명
- **stock_code**: 종목코드 (6자리 숫자)

컬럼명은 다음 변형도 지원됩니다:
- `corp_code` / `회사코드`
- `corp_name` / `회사명`
- `stock_code` / `종목코드`

## 사용 방법

### 3단계 파이프라인 (권장)

성능과 안정성을 위해 3단계 파이프라인으로 분리되어 있습니다:

#### Stage 1: 데이터 수집

```bash
python stage1_collect.py \
    --input input.xlsx \
    --out collected_rows.parquet \
    --years-back 3 \
    --workers-http 10 \
    --workers-cpu 4 \
    --checkpoint-interval 100 \
    --resume
```

**파라미터:**
- `--input`: 입력 엑셀 파일
- `--out`: 출력 파일 (parquet 또는 csv)
- `--years-back`: 검색할 연도 범위
- `--workers-http`: HTTP 동시 요청 수
- `--workers-cpu`: CPU 프로세스 워커 수
- `--checkpoint-interval`: N개 회사마다 체크포인트 저장
- `--resume`: 기존 파일에서 재시작

#### Stage 2: LLM 매핑

```bash
python stage2_map_llm.py \
    --in collected_rows.parquet \
    --out mapped.parquet \
    --concurrency 5 \
    --batch-size 10 \
    --checkpoint-interval 1000 \
    --resume
```

**파라미터:**
- `--in`: Stage1 출력 파일
- `--out`: 매핑 결과 파일
- `--concurrency`: LLM 동시 요청 수
- `--batch-size`: 배치 크기 (5-20 권장)
- `--checkpoint-interval`: N개 row마다 체크포인트 저장
- `--resume`: 기존 파일에서 재시작

#### Stage 3: 최종 병합

```bash
python stage3_merge_export.py \
    --input-xlsx input.xlsx \
    --mapped mapped.parquet \
    --out dart_result.xlsx \
    --no-dedupe  # 중복 제거 비활성화 (선택적)
```

**파라미터:**
- `--input-xlsx`: 원본 입력 엑셀 파일
- `--mapped`: Stage2 출력 파일
- `--out`: 최종 출력 엑셀 파일
- `--no-dedupe`: 중복 제거 비활성화

### 기존 방식 (단일 스크립트)

```python
from dart_fetcher import build_dart_result

output_path = build_dart_result(
    input_path="input.xlsx",
    output_path="dart_result.xlsx",
    years_back=3,
    debug=True,
)
```

또는

```bash
python main.py
```

## 출력 형식

출력 엑셀 파일에는 두 개의 시트가 생성됩니다:

### 1. 회사별 공통행

| 회사 | 종목코드 | 구분 | url |
|------|----------|------|-----|
| 삼성전자 | 005930 | 분기 | https://... |

### 2. 임원 현황

| 회사 | 종목코드 | 성명 | 담당업무 | 주요경력 | 학교 | 학과 | 교수 |
|------|----------|------|----------|----------|------|------|------|
| 삼성전자 | 005930 | 홍길동 | DX부문장, MX사업부장 | ... | 서울대학교 | 경영학과 | 교수 |
| 삼성전자 | 005930 | 김철수 | 대표이사 | ... | 해당없음 | 해당없음 | 해당없음 |

### 출력 컬럼 설명

- **회사**: 회사명
- **종목코드**: 종목코드
- **성명**: 임원 이름
- **담당업무**: 담당 업무 (특수기호 제거, 쉼표로 구분)
- **주요경력**: 주요 경력
- **학교**: 교수인 경우 소속 학교 (교수가 아니면 "해당없음")
- **학과**: 교수인 경우 소속 학과 (교수가 아니면 "해당없음")
- **교수**: 교수 직함 (교수, 부교수, 조교수 등, 교수가 아니면 "해당없음")

## 주요 기능 상세

### 1. 보고서 자동 검색

- 분기보고서 → 반기보고서 → 연간보고서 순으로 우선순위 검색
- 최근 N년간의 보고서를 검색

### 2. 임원 정보 추출

- XML/HTML에서 "가. 등기임원" 또는 "가. 임원 현황" 테이블 자동 탐지
- BeautifulSoup을 사용한 테이블 파싱

### 3. AI 기반 정보 정규화

- OpenAI API를 사용하여 임원 정보를 정규화된 형식으로 매핑
- 담당업무에서 특수기호 제거 및 정리
- 교수 정보 자동 추출 (담당업무 우선, 없으면 주요경력)

### 4. 병렬 처리

- 회사 단위 병렬 처리 (최대 3개 동시)
- 임원 행 단위 병렬 처리 (최대 3개 동시)
- 비동기 처리로 성능 최적화

## 체크포인트 기능

100개 회사마다 자동으로 체크포인트 파일을 저장합니다:
- 파일명: `dart_result_checkpoint_100.csv`, `dart_result_checkpoint_200.csv` 등

## 문제 해결

### Connection Error

- 네트워크 연결 문제 확인
- 방화벽 설정 확인
- 프록시 설정 필요 여부 확인

### API Key Error

- `.env` 파일에 모든 필수 환경 변수가 설정되어 있는지 확인
- API 키가 유효한지 확인

### 모델 접근 오류

- `OPENAI_MODEL` 환경 변수에 사용 가능한 모델명이 설정되어 있는지 확인
- 에러 메시지에서 사용 가능한 모델 목록 확인

## 의존성

- `pandas`: 데이터 처리
- `openpyxl`: 엑셀 파일 읽기/쓰기
- `requests`: HTTP 요청
- `python-dotenv`: 환경 변수 관리
- `beautifulsoup4`: HTML/XML 파싱
- `lxml`: XML 파서
- `openai`: OpenAI API 클라이언트

## 라이선스

이 프로젝트는 개인/내부 사용 목적으로 제작되었습니다.

## 참고사항

- DART API 사용량 제한에 주의하세요
- OpenAI API 사용량에 따른 비용이 발생할 수 있습니다
- 대량 데이터 처리 시 시간이 소요될 수 있습니다

