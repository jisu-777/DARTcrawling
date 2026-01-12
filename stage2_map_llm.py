"""
Stage 2: LLM을 사용한 임원 정보 매핑
- 배치 처리 지원 (5-20개 row 묶어서 처리)
- 재시도 및 백오프 강화
- parquet/csv로 결과 저장
"""
import os
import json
import asyncio
import argparse
import time
from typing import List, Dict, Optional, Any
import logging

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_llm_prompt(row_data: Dict, columns_info: List[str]) -> str:
    """LLM 프롬프트 생성"""
    return f"""다음은 DART 보고서에서 추출한 임원 정보 테이블의 한 행입니다.
컬럼명: {json.dumps(columns_info, ensure_ascii=False)}
행 데이터: {json.dumps(row_data, ensure_ascii=False)}

이 데이터를 다음 형식의 JSON으로 매핑해주세요. 값이 없으면 빈 문자열("")로 설정하세요.

필수 출력 형식:
{{
    "성명": "",
    "성별": "",
    "출생년월": "",
    "직위": "",
    "등기임원여부": "",
    "상근여부": "",
    "담당업무": "",
    "주요경력": "",
    "학교": "",
    "학과": "",
    "교수": "",
    "소유주식수(의결권O)": "",
    "소유주식수(의결권X)": "",
    "최대주주와의 관계": "",
    "재직기간": "",
    "임기만료일": ""
}}

컬럼명을 분석하여 적절한 필드에 매핑하세요. 예를 들어:
- "성명", "이름", "name" 등 -> "성명"
- "직위", "position", "직책" 등 -> "직위"
- "소유주식수"와 "의결권" 관련 컬럼 -> "소유주식수(의결권O)" 또는 "소유주식수(의결권X)"
- "임기", "만료일" 등 -> "임기만료일"

중요: "학교", "학과", "교수" 필드는 "담당업무" 또는 "주요경력" 필드에서 추출해야 합니다. "담당업무"를 우선적으로 확인하세요.

1. 추출 조건:
   - 담당업무 또는 주요경력에 교수 직함(교수, 부교수, 조교수, 전임강사 등)이 포함된 경우에만 추출
   - 교수 직함이 없으면 "학교": "해당없음", "학과": "해당없음", "교수": "해당없음"으로 설정

2. 추출 방법:
   - 담당업무를 먼저 확인하고, 없으면 주요경력을 확인하세요
   - 교수 직함이 있는 문장에서 교수 직함 앞에 나오는 학교명과 학과명을 찾으세요
   - 교수 직함 자체를 "교수" 필드에 입력하세요 (예: "교수", "부교수", "조교수", "전임강사" 등)
   - 예: "한성대 AI응용학과 교수" -> 학교: "한성대" 또는 "한성대학교", 학과: "AI응용학과", 교수: "교수"
   - 예: "서울대학교 경영학과 부교수" -> 학교: "서울대학교", 학과: "경영학과", 교수: "부교수"

3. 추출 규칙:
   - 학교명은 "한성대", "서울대", "고려대" 같은 약칭이거나 "한성대학교", "서울대학교" 같은 전체명일 수 있습니다. 원문에 명시된 대로 그대로 추출하세요.
   - 학과명은 "AI응용학과", "경영학과", "경영대학", "의과대학" 등으로 명시된 대로 추출하세요.
   - 교수 필드에는 정확한 직함을 입력하세요: "교수", "부교수", "조교수", "전임강사", "임용교수" 등

4. 오류 방지:
   - 교수 직함이 없는 경우: "학교": "해당없음", "학과": "해당없음", "교수": "해당없음"
   - 교수 직함이 있지만 학교명이나 학과명을 찾을 수 없는 경우: 가능한 정보만 추출하고 나머지는 빈 문자열("")

5. 담당업무 필드 정리:
   - 담당업무 필드에 특수기호(ㆍ, ·, • 등)나 불필요한 구분자가 포함되어 있으면 제거하고 쉼표(,)로 구분하세요
   - 예: "ㆍDX부문장 직무대행ㆍMX사업부장" -> "DX부문장 직무대행, MX사업부장"
   - 여러 업무가 나열되어 있을 때는 쉼표로 구분하여 깔끔하게 정리하세요

JSON만 반환하고 다른 설명은 포함하지 마세요."""


async def map_row_with_llm(
    row_data: Dict,
    columns_info: List[str],
    client: AsyncOpenAI,
    model_name: str,
    max_retries: int = 3,
) -> tuple[Optional[Dict], int, float, Optional[str]]:
    """
    단일 row를 LLM으로 매핑
    
    Returns:
        (mapped_data, attempts, latency_ms, error)
    """
    prompt = get_llm_prompt(row_data, columns_info)
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "당신은 DART 보고서의 임원 정보를 정규화된 형식으로 매핑하는 전문가입니다. JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=30.0
            )
            
            latency_ms = (time.time() - start_time) * 1000
            result_text = response.choices[0].message.content.strip()
            
            # JSON 파싱
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join([line for line in lines if not line.strip().startswith("```")])
            
            mapped_data = json.loads(result_text)
            
            # 필수 필드 확인
            required_fields = [
                "성명", "성별", "출생년월", "직위", "등기임원여부", "상근여부",
                "담당업무", "주요경력", "학교", "학과", "교수",
                "소유주식수(의결권O)", "소유주식수(의결권X)",
                "최대주주와의 관계", "재직기간", "임기만료일"
            ]
            
            for field in required_fields:
                if field not in mapped_data:
                    mapped_data[field] = ""
            
            # 교수 정보 후처리
            duty_text = mapped_data.get("담당업무", "") or ""
            career_text = mapped_data.get("주요경력", "") or ""
            
            professor_keywords = ["교수", "부교수", "조교수", "전임강사", "임용교수"]
            has_professor = False
            professor_title = ""
            
            for keyword in professor_keywords:
                if keyword in duty_text:
                    has_professor = True
                    professor_title = keyword
                    break
            
            if not has_professor:
                for keyword in professor_keywords:
                    if keyword in career_text:
                        has_professor = True
                        professor_title = keyword
                        break
            
            if not has_professor:
                if not mapped_data.get("학교") or mapped_data.get("학교", "").strip() == "":
                    mapped_data["학교"] = "해당없음"
                if not mapped_data.get("학과") or mapped_data.get("학과", "").strip() == "":
                    mapped_data["학과"] = "해당없음"
                if not mapped_data.get("교수") or mapped_data.get("교수", "").strip() == "":
                    mapped_data["교수"] = "해당없음"
            else:
                if not mapped_data.get("교수") or mapped_data.get("교수", "").strip() == "":
                    mapped_data["교수"] = professor_title
            
            return mapped_data, attempt + 1, latency_ms, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # 백오프
                continue
            latency_ms = (time.time() - start_time) * 1000
            return None, attempt + 1, latency_ms, str(e)
    
    latency_ms = (time.time() - start_time) * 1000
    return None, max_retries, latency_ms, "Max retries exceeded"


async def map_batch_with_llm(
    batch_rows: List[Dict],
    client: AsyncOpenAI,
    model_name: str,
    concurrency: int,
) -> List[Dict]:
    """배치를 병렬로 매핑"""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_row(row_dict):
        async with semaphore:
            row_data = json.loads(row_dict["row_json"])
            columns_info = json.loads(row_dict["columns_json"])
            
            mapped_data, attempts, latency_ms, error = await map_row_with_llm(
                row_data, columns_info, client, model_name
            )
            
            if mapped_data and mapped_data.get("성명") and mapped_data.get("성명").strip():
                result = {
                    "회사": row_dict["회사"],
                    "종목코드": row_dict["종목코드"],
                    "구분": row_dict["구분"],
                    "url": row_dict["url"],
                    **mapped_data,
                    # 소스 키
                    "company_id": row_dict["company_id"],
                    "rcept_no": row_dict["rcept_no"],
                    "source_xml_index": row_dict["source_xml_index"],
                    "row_index": row_dict["row_index"],
                    # LLM 메타데이터
                    "llm_model": model_name,
                    "llm_latency_ms": latency_ms,
                    "llm_attempts": attempts,
                    "error": error or "",
                }
                return result
            return None
    
    tasks = [process_row(row) for row in batch_rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    mapped_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Batch mapping error: {result}")
            continue
        if result is not None:
            mapped_results.append(result)
    
    return mapped_results


def create_batches(df: pd.DataFrame, batch_size: int, max_tokens_per_batch: int = 50000) -> List[List[Dict]]:
    """동적 배치 생성 (토큰 수 고려)"""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for _, row in df.iterrows():
        row_text = row.get("row_text", "")
        # 간단한 토큰 추정 (한글 기준 약 1.5배)
        estimated_tokens = len(row_text) * 2
        
        if len(current_batch) >= batch_size or (current_tokens + estimated_tokens) > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.append(row.to_dict())
        current_tokens += estimated_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


async def map_all_rows(
    input_path: str,
    output_path: str,
    model_name: str,
    concurrency: int,
    checkpoint_interval: int = 1000,
    resume: bool = False,
) -> None:
    """모든 row를 LLM으로 매핑"""
    # 입력 파일 읽기
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # 이미 매핑된 row 확인 (resume 모드)
    mapped_ids = set()
    if resume and os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path) if output_path.endswith('.parquet') else pd.read_csv(output_path)
            if 'company_id' in existing_df.columns and 'row_index' in existing_df.columns:
                mapped_ids = set(zip(existing_df['company_id'], existing_df['source_xml_index'], existing_df['row_index']))
                logger.info(f"Resume mode: {len(mapped_ids)} rows already mapped")
        except Exception as e:
            logger.warning(f"Failed to read existing file for resume: {e}")
    
    # Resume 모드: 이미 매핑된 row 제외
    if resume and mapped_ids:
        df = df[~df.apply(
            lambda row: (row.get('company_id'), row.get('source_xml_index'), row.get('row_index')) in mapped_ids,
            axis=1
        )]
        logger.info(f"Remaining rows to map: {len(df)}")
    
    if df.empty:
        logger.info("No rows to map")
        return
    
    # OpenAI 클라이언트 생성
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    
    openai_base_url = os.getenv("OPENAI_API_BASE", "").strip()
    if openai_base_url:
        if '/chat/completions' in openai_base_url:
            openai_base_url = openai_base_url.replace('/chat/completions', '').rstrip('/')
        if openai_base_url.endswith('/v1'):
            openai_base_url = openai_base_url[:-3].rstrip('/')
        if openai_base_url:
            openai_base_url = openai_base_url.rstrip('/')
            client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        else:
            client = AsyncOpenAI(api_key=openai_api_key)
    else:
        client = AsyncOpenAI(api_key=openai_api_key)
    
    # 모든 row를 50개씩 동시 처리 (배치 없이)
    logger.info(f"Concurrency: {concurrency} (up to {concurrency} simultaneous LLM requests)")
    logger.info(f"Processing all {len(df)} rows with {concurrency} concurrent requests")
    
    all_mapped = []
    completed = 0
    start_time = time.time()
    first_completion_time = None
    last_checkpoint = 0
    
    # 세마포어로 동시 요청 수 제한
    semaphore = asyncio.Semaphore(concurrency)
    active_tasks = set()
    
    async def process_row_with_tracking(row_dict, row_idx):
        """단일 row 처리 + 진행률 추적"""
        nonlocal all_mapped, completed, first_completion_time, last_checkpoint
        
        async with semaphore:
            row_data = json.loads(row_dict["row_json"])
            columns_info = json.loads(row_dict["columns_json"])
            
            mapped_data, attempts, latency_ms, error = await map_row_with_llm(
                row_data, columns_info, client, model_name
            )
            
            if first_completion_time is None:
                first_completion_time = time.time()
                elapsed = first_completion_time - start_time
                logger.info(
                    f"✓ First LLM request completed in {elapsed:.2f}s "
                    f"({len(active_tasks)} still active - CONCURRENCY CONFIRMED!)"
                )
            
            if mapped_data and mapped_data.get("성명") and mapped_data.get("성명").strip():
                result = {
                    "회사": row_dict["회사"],
                    "종목코드": row_dict["종목코드"],
                    "구분": row_dict["구분"],
                    "url": row_dict["url"],
                    **mapped_data,
                    # 소스 키
                    "company_id": row_dict["company_id"],
                    "rcept_no": row_dict["rcept_no"],
                    "source_xml_index": row_dict["source_xml_index"],
                    "row_index": row_dict["row_index"],
                    # LLM 메타데이터
                    "llm_model": model_name,
                    "llm_latency_ms": latency_ms,
                    "llm_attempts": attempts,
                    "error": error or "",
                }
                all_mapped.append(result)
            
            completed += 1
            
            # 진행률 로그 (10개마다 또는 5초마다)
            current_time = time.time()
            elapsed = current_time - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            
            if completed % 10 == 0 or (current_time - start_time) % 5 < 0.1 or completed == len(df):
                logger.info(
                    f"LLM progress: {completed}/{len(df)} rows "
                    f"({rate:.1f} rows/sec, {len(active_tasks)} active tasks)"
                )
            
            # 체크포인트 저장
            if len(all_mapped) - last_checkpoint >= checkpoint_interval:
                checkpoint_df = pd.DataFrame(all_mapped)
                if output_path.endswith('.parquet'):
                    checkpoint_df.to_parquet(output_path, index=False)
                else:
                    checkpoint_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"Checkpoint saved: {completed}/{len(df)} rows processed, {len(all_mapped)} mapped")
                last_checkpoint = len(all_mapped)
    
    # 모든 row를 동시에 처리 (concurrency 제한 내에서)
    tasks = []
    for idx, row in df.iterrows():
        task = asyncio.create_task(process_row_with_tracking(row.to_dict(), idx))
        tasks.append(task)
        active_tasks.add(task)
    
    # 완료된 태스크 추적
    for task in asyncio.as_completed(tasks):
        active_tasks.discard(task)
        try:
            await task
        except Exception as e:
            logger.error(f"Row processing error: {e}")
            completed += 1
    
    total_elapsed = time.time() - start_time
    logger.info(
        f"✓ All LLM requests completed in {total_elapsed:.2f}s "
        f"(avg {total_elapsed/len(df):.2f}s per row, "
        f"throughput: {len(df)/total_elapsed:.2f} rows/sec)"
    )
    
    # 최종 저장
    if all_mapped:
        result_df = pd.DataFrame(all_mapped)
        if output_path.endswith('.parquet'):
            result_df.to_parquet(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(all_mapped)} mapped rows to {output_path}")
    else:
        logger.warning("No rows were successfully mapped")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM 매핑")
    parser.add_argument("--in", dest="input_path", required=True, help="Input file from Stage1")
    parser.add_argument("--out", required=True, help="Output file (mapped.parquet or .csv)")
    parser.add_argument("--concurrency", type=int, default=50, help="LLM concurrent requests (default: 50, max: 50)")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint every N mapped rows")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    load_dotenv()
    model_name = os.getenv("OPENAI_MODEL", "azure.gpt-4o-mini")
    
    logger.info(f"Starting LLM mapping")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.out}")
    logger.info(f"Model: {model_name}")
    # concurrency 제한 (최대 50)
    concurrency = min(args.concurrency, 50)
    logger.info(f"Concurrency: {concurrency} (up to {concurrency} simultaneous LLM requests)")
    
    asyncio.run(map_all_rows(
        input_path=args.input_path,
        output_path=args.out,
        model_name=model_name,
        concurrency=concurrency,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    ))
    
    logger.info("Stage 2 complete!")


if __name__ == "__main__":
    main()

