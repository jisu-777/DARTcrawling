import os
import re
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from dart_executive_html import extract_registered_executives, map_executive_row_with_ai
from openai import AsyncOpenAI

DART_MAIN_URL = "https://dart.fss.or.kr/dsaf001/main.do"


def normalize_stock_code(val) -> str:
    """Stock code should be 6 digits (e.g., 36720 -> 036720)."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)  # if Excel read as float like 36720.0
    s = re.sub(r"\D", "", s)
    return s.zfill(6) if s else ""


def normalize_corp_code(val) -> str:
    """Corp code must be 8 digits.
    - 6 digits: add "00" prefix -> 8 digits
    - 7 digits: add "0" prefix -> 8 digits
    - 8 digits: use as is (including leading zeros)
    """
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)  # if Excel read as float like 123456.0
    s = re.sub(r"\D", "", s)  # remove non-digits
    
    if not s:
        return ""
    
    # Check original length (before removing leading zeros)
    original_length = len(s)
    
    if original_length == 6:
        return "00" + s
    elif original_length == 7:
        return "0" + s
    elif original_length == 8:
        return s
    else:
        # If not 6, 7, or 8 digits, pad to 8 with zeros from left
        return s.zfill(8)


def _fetch_reports_by_corp_code(api_key: str, base_url: str, years_back: int, debug: bool, 
                                corp_code: str):
    """
    Internal function to fetch reports using corp_code.
    Returns (all_items, identifier_string) or (None, None) on error.
    """
    if not corp_code:
        return None, None
    
    identifier = f"corp_code={corp_code}"
    
    end_de = datetime.today().strftime("%Y%m%d")
    bgn_de = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y%m%d")

    url = f"{base_url.rstrip('/')}/list.json"
    
    if debug:
        print(f"  [DEBUG] API URL: {url}")
        print(f"  [DEBUG] Date range: {bgn_de} ~ {end_de}")
    
    # Collect all items across pages
    all_items = []
    page_no = 1
    page_count = 100
    max_pages = 10  # safety limit
    
    while page_no <= max_pages:
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "pblntf_ty": "A",   # periodic disclosure
            "page_no": page_no,
            "page_count": page_count,
        }

        if debug and page_no == 1:
            # Print full URL with params for first page
            full_url = f"{url}?" + "&".join([f"{k}={v}" for k, v in params.items() if k != "crtfc_key"])
            print(f"  [DEBUG] Full request URL (without API key): {full_url}")
        
        try:
            # 재시도 로직 추가 (네트워크 오류 대응)
            for retry in range(3):
                try:
                    r = requests.get(url, params=params, timeout=20)
                    r.raise_for_status()
                    data = r.json()
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if retry < 2:
                        if debug:
                            print(f"  [DEBUG] 네트워크 오류, 재시도 {retry + 1}/3: {e}")
                        import time
                        time.sleep(1)  # 1초 대기 후 재시도
                        continue
                    else:
                        raise
            else:
                # 마지막 재시도 실패
                raise requests.exceptions.RequestException("Max retries exceeded")
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"  [DEBUG] Request error for {identifier}: {e}")
            return None, None
        except Exception as e:
            if debug:
                print(f"  [DEBUG] Unexpected error for {identifier}: {e}")
            return None, None

        status = str(data.get("status", ""))
        if status != "000":
            message = data.get("message", "Unknown error")
            if debug:
                print(f"  [DEBUG] API error for {identifier}: status={status}, message={message}")
            # Don't return None immediately - might be pagination issue
            if status in ["013", "014"]:  # No data found
                break
            # For other errors, try to continue or return None based on error
            if page_no == 1:  # Only return None on first page error
                return None, None

        page_items = data.get("list", []) or []
        if not page_items:
            break
        
        all_items.extend(page_items)
        
        # Check if there are more pages
        total_count = data.get("total_count", 0)
        if len(all_items) >= total_count or len(page_items) < page_count:
            break
        
        page_no += 1

    if not all_items:
        if debug:
            print(f"  [DEBUG] No items found for {identifier}")
        return None, None
    
    return all_items, identifier


def fetch_latest_periodic_report(corp_code: str, api_key: str, base_url: str, 
                                 years_back: int = 3, debug: bool = False):
    """
    Find latest quarterly report first; if none, latest half-year report; if none, return None.
    Uses DART list.json (periodic disclosures) and filters by report_nm text.
    Uses corp_code only (must be 8 digits).
    Handles pagination and provides better error handling.
    """
    if not corp_code:
        if debug:
            print(f"  [DEBUG] corp_code is empty")
        return None

    # Normalize corp_code to 8 digits
    corp_code = normalize_corp_code(corp_code)
    if not corp_code or len(corp_code) != 8:
        if debug:
            print(f"  [DEBUG] Invalid corp_code after normalization: {corp_code}")
        return None

    if debug:
        print(f"  [DEBUG] Using corp_code: {corp_code}")

    all_items, identifier = _fetch_reports_by_corp_code(
        api_key=api_key,
        base_url=base_url,
        years_back=years_back,
        debug=debug,
        corp_code=corp_code
    )

    if not all_items:
        return None

    def is_quarterly(name: str) -> bool:
        return "분기보고서" in name or "분기" in name

    def is_halfyear(name: str) -> bool:
        return "반기보고서" in name or "반기" in name

    def is_annual(name: str) -> bool:
        return "사업보고서" in name or "연간" in name

    # Sort by date (newest first)
    items_sorted = sorted(
        all_items,
        key=lambda x: (x.get("rcept_dt", ""), x.get("rcept_no", "")),
        reverse=True,
    )

    if debug:
        print(f"  [DEBUG] Found {len(all_items)} total reports for {identifier}")
        if len(all_items) > 0:
            print(f"  [DEBUG] Sample report names: {[it.get('report_nm', '') for it in items_sorted[:3]]}")

    # reprt_code 매핑: 분기=11013, 반기=11012, 연간(사업보고서)=11011
    REPRT_CODE_MAP = {
        "분기": "11013",
        "반기": "11012",
        "연간": "11011",
    }

    # 1) quarterly first
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_quarterly(nm):
            rcp = it.get("rcept_no", "") or ""
            bsns_year = it.get("bsns_year", "") or ""
            return {
                "type": "분기",
                "rcept_no": rcp,
                "url": f"{DART_MAIN_URL}?rcpNo={rcp}",
                "bsns_year": bsns_year,
                "reprt_code": REPRT_CODE_MAP["분기"],
                "report_nm": nm,
            }

    # 2) half-year
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_halfyear(nm):
            rcp = it.get("rcept_no", "") or ""
            bsns_year = it.get("bsns_year", "") or ""
            return {
                "type": "반기",
                "rcept_no": rcp,
                "url": f"{DART_MAIN_URL}?rcpNo={rcp}",
                "bsns_year": bsns_year,
                "reprt_code": REPRT_CODE_MAP["반기"],
                "report_nm": nm,
            }

    # 3) annual as fallback
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_annual(nm):
            rcp = it.get("rcept_no", "") or ""
            bsns_year = it.get("bsns_year", "") or ""
            return {
                "type": "연간",
                "rcept_no": rcp,
                "url": f"{DART_MAIN_URL}?rcpNo={rcp}",
                "bsns_year": bsns_year,
                "reprt_code": REPRT_CODE_MAP["연간"],
                "report_nm": nm,
            }

    if debug:
        print(f"  [DEBUG] No matching report type found for {identifier}")
    return None


def build_dart_result(input_path: str = "input.xlsx", output_path: str = "dart_result.xlsx", years_back: int = 3, debug: bool = False):
    """
    Read input Excel (corp_code, corp_name, stock_code),
    create output Excel with columns: 회사, 종목코드, 구분, url.
    """
    load_dotenv()
    api_key = os.getenv("DART_API_KEY", "").strip()
    base_url = os.getenv("DART_API_BASE_URL", "").strip()

    if not api_key or not base_url:
        raise RuntimeError("Missing env. Please set DART_API_KEY and DART_API_BASE_URL in .env")

    # Read Excel with flexible column handling
    df = pd.read_excel(input_path)
    
    # Normalize column names (handle whitespace, case variations)
    df.columns = df.columns.str.strip()
    
    # Map common column name variations
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'corp_code' in col_lower or '회사코드' in col:
            col_mapping[col] = 'corp_code'
        elif 'corp_name' in col_lower or '회사명' in col or 'corp_eng_name' in col_lower:
            if 'corp_name' not in [v for v in col_mapping.values()]:
                col_mapping[col] = 'corp_name'
        elif 'stock_code' in col_lower or '종목코드' in col:
            col_mapping[col] = 'stock_code'
    
    # Rename columns
    df = df.rename(columns=col_mapping)
    
    # Ensure required columns exist
    required_cols = {"corp_code", "corp_name", "stock_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Input file is missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}")

    # Convert corp_code to string, handling NaN and float values
    df["corp_code"] = df["corp_code"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df["corp_name"] = df["corp_name"].astype(str).str.strip()
    
    results = []
    all_executives = []  # 임원 정보 리스트 (스트리밍 방식)
    total = len(df)
    
    # OpenAI API 키 확인 및 클라이언트 생성 (1번만 생성하여 재사용)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    
    # OpenAI API Base URL 확인 (OPENAI_API_BASE 환경 변수 사용)
    openai_base_url = os.getenv("OPENAI_API_BASE", "").strip()
    
    # base_url에서 /chat/completions 같은 엔드포인트 경로 제거
    if openai_base_url:
        # /v1 또는 /chat/completions 같은 경로 제거하여 베이스 URL만 남김
        if '/chat/completions' in openai_base_url:
            openai_base_url = openai_base_url.replace('/chat/completions', '').rstrip('/')
        # 마지막에 /v1이 있으면 제거 (SDK가 자동 추가)
        if openai_base_url.endswith('/v1'):
            openai_base_url = openai_base_url[:-3].rstrip('/')
        # 빈 문자열이 아니면 베이스 URL로 사용
        if openai_base_url:
            # 마지막 슬래시 제거
            openai_base_url = openai_base_url.rstrip('/')
            client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
            if debug:
                print(f"  [DEBUG] OpenAI Base URL 설정: {openai_base_url}")
        else:
            client = AsyncOpenAI(api_key=openai_api_key)
    else:
        client = AsyncOpenAI(api_key=openai_api_key)
    
    # 배치 파이프라인: 1단계 - 모든 회사의 테이블 먼저 수집
    async def process_company_collect(idx, row):
        """회사 1개를 처리하여 테이블만 추출 (AI 매핑 제외)"""
        corp_code_raw = str(row.get("corp_code", "")).strip()
        corp_name = str(row.get("corp_name", "")).strip()
        stock_code = normalize_stock_code(row.get("stock_code"))
        
        corp_code = normalize_corp_code(corp_code_raw)
        
        if not corp_code or len(corp_code) != 8:
            if debug:
                print(f"[{idx+1}/{total}] Skipping row {idx+1}: invalid corp_code")
            return {
                "result": {"회사": corp_name, "종목코드": stock_code, "구분": "코드오류", "url": ""},
                "exec_df": None
            }

        if debug:
            print(f"[{idx+1}/{total}] Processing: {corp_name} (corp_code: {corp_code}, stock_code: {stock_code})")

        report_meta = await asyncio.to_thread(
            fetch_latest_periodic_report,
            corp_code=corp_code,
            api_key=api_key,
            base_url=base_url,
            years_back=years_back,
            debug=debug,
        )

        if report_meta is None:
            return {
                "result": {"회사": corp_name, "종목코드": stock_code, "구분": "정보없음", "url": ""},
                "exec_df": None
            }
        
        result_item = {
            "회사": corp_name,
            "종목코드": stock_code,
            "구분": report_meta["type"],
            "url": report_meta["url"]
        }
        
        if debug:
            print(f"  -> Found: {report_meta['type']} (rcept_no: {report_meta['rcept_no']})")
        
        # 테이블만 추출 (AI 매핑은 나중에)
        exec_df = None
        if report_meta.get("rcept_no") and report_meta.get("url"):
            if debug:
                print(f"  [임원정보 테이블 추출] {corp_name} (rcp_no: {report_meta['rcept_no']})")
            
            try:
                exec_df = await asyncio.to_thread(
                    extract_registered_executives,
                    rcp_no=report_meta["rcept_no"],
                    main_url=report_meta["url"],
                    corp_name=corp_name,
                    stock_code=stock_code,
                    report_type=report_meta["type"],
                    api_key=api_key,
                    base_url=base_url,
                    debug=debug,
                )
                
                if exec_df is not None and not exec_df.empty:
                    if debug:
                        print(f"  -> 테이블 추출 성공: {len(exec_df)}행")
                else:
                    if debug:
                        print(f"  -> 테이블 추출 실패: 빈 DataFrame")
            except Exception as e:
                if debug:
                    print(f"  -> 테이블 추출 실패: {e}")
        
        return {"result": result_item, "exec_df": exec_df}
    
    # 배치 파이프라인: 1단계 - 모든 회사의 테이블 먼저 수집
    async def collect_all_tables():
        """모든 회사의 테이블을 먼저 수집 (AI 매핑 제외)"""
        company_semaphore = asyncio.Semaphore(50)  # 회사 단위 동시성 제한 (50개)
        
        async def process_with_semaphore(idx, row):
            async with company_semaphore:
                return await process_company_collect(idx, row)
        
        # 모든 회사 처리
        tasks = [process_with_semaphore(idx, row) for idx, row in df.iterrows()]
        collected_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 수집
        all_tables = []  # (result_item, exec_df, company_meta) 튜플 리스트
        for result in collected_results:
            if isinstance(result, Exception):
                if debug:
                    print(f"  [DEBUG] 회사 처리 중 오류: {result}")
                continue
            if isinstance(result, dict):
                results.append(result["result"])
                if result["exec_df"] is not None and not result["exec_df"].empty:
                    company_meta = {
                        "회사": result["result"]["회사"],
                        "종목코드": result["result"]["종목코드"],
                        "구분": result["result"]["구분"],
                        "url": result["result"]["url"]
                    }
                    all_tables.append((result["result"], result["exec_df"], company_meta))
        
        if debug:
            print(f"\n[1단계 완료] 총 {len(all_tables)}개 회사의 테이블 수집 완료")
        
        return all_tables
    
    # 배치 파이프라인: 2단계 - 모든 테이블을 LLM에 일괄 처리 (체크포인트 포함)
    async def map_all_tables_with_ai(all_tables, checkpoint_interval: int = 100):
        """수집된 모든 테이블을 LLM에 일괄 처리 (중간 저장 포함)"""
        if debug:
            print(f"\n[2단계 시작] {len(all_tables)}개 회사의 테이블을 LLM에 일괄 처리...")
        
        # 모든 테이블의 모든 행을 하나의 리스트로 모음
        all_rows_to_map = []
        for result_item, exec_df, company_meta in all_tables:
            for idx, row in exec_df.iterrows():
                all_rows_to_map.append((row, company_meta))
        
        if debug:
            print(f"  총 {len(all_rows_to_map)}개 행을 LLM에 처리합니다")
        
        # 체크포인트 파일 경로
        checkpoint_path = output_path.replace(".xlsx", "_llm_checkpoint.csv")
        last_checkpoint_count = 0
        
        # 세마포어로 동시 요청 수 제한 (50개)
        semaphore = asyncio.Semaphore(50)
        completed_count = 0
        
        async def map_row_with_semaphore(row, company_meta):
            nonlocal completed_count, last_checkpoint_count
            async with semaphore:
                result = await map_executive_row_with_ai(row, company_meta, client, debug=debug)
                completed_count += 1
                
                # 결과 저장
                if result is not None:
                    all_executives.append(result)
                    
                    # 체크포인트 저장 (일정 간격마다)
                    if len(all_executives) - last_checkpoint_count >= checkpoint_interval:
                        try:
                            checkpoint_df = pd.DataFrame(all_executives)
                            checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
                            last_checkpoint_count = len(all_executives)
                            if debug:
                                print(f"  [체크포인트 저장] {completed_count}/{len(all_rows_to_map)} 행 처리, {len(all_executives)}명 매핑 완료 → {checkpoint_path}")
                        except Exception as e:
                            if debug:
                                print(f"  [DEBUG] 체크포인트 저장 실패: {e}")
                
                # 진행률 로그
                if completed_count % 50 == 0 or completed_count == len(all_rows_to_map):
                    if debug:
                        print(f"  [진행률] {completed_count}/{len(all_rows_to_map)} 행 처리 완료 ({len(all_executives)}명 매핑)")
                
                return result
        
        # 모든 행을 병렬로 처리 (50개씩 배치로 나눠서 처리)
        batch_size = 500  # 500개씩 배치로 나눠서 처리
        total_batches = (len(all_rows_to_map) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_rows_to_map))
            batch_rows = all_rows_to_map[start_idx:end_idx]
            
            if debug:
                print(f"  [배치 {batch_idx + 1}/{total_batches}] {len(batch_rows)}개 행 처리 중...")
            
            # 배치 내에서 병렬 처리
            tasks = [map_row_with_semaphore(row, company_meta) for row, company_meta in batch_rows]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 배치 완료 후 체크포인트 저장
            if len(all_executives) > last_checkpoint_count:
                try:
                    checkpoint_df = pd.DataFrame(all_executives)
                    checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
                    last_checkpoint_count = len(all_executives)
                    if debug:
                        print(f"  [배치 완료] 체크포인트 저장: {len(all_executives)}명 매핑 완료")
                except Exception as e:
                    if debug:
                        print(f"  [DEBUG] 배치 체크포인트 저장 실패: {e}")
        
        if debug:
            print(f"\n[2단계 완료] 총 {len(all_executives)}명의 임원 정보 추출 완료")
    
    # 배치 파이프라인 실행
    async def run_pipeline():
        # 1단계: 모든 테이블 수집
        all_tables = await collect_all_tables()
        
        # 1단계 완료 후 체크포인트 저장 (테이블 데이터)
        if all_tables:
            checkpoint_tables_path = output_path.replace(".xlsx", "_tables_checkpoint.pkl")
            try:
                import pickle
                with open(checkpoint_tables_path, 'wb') as f:
                    pickle.dump(all_tables, f)
                if debug:
                    print(f"  [1단계 체크포인트 저장] {len(all_tables)}개 회사의 테이블 데이터 저장 → {checkpoint_tables_path}")
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] 테이블 체크포인트 저장 실패: {e}")
        
        # 2단계: LLM 매핑 (체크포인트 포함)
        if all_tables:
            await map_all_tables_with_ai(all_tables, checkpoint_interval=100)
    
    asyncio.run(run_pipeline())
    
    if debug and all_executives:
        print(f"\n[AI 매핑 완료] 총 {len(all_executives)}명의 임원 정보 추출")

    out_df = pd.DataFrame(results, columns=["회사", "종목코드", "구분", "url"])
    
    # Excel 파일에 여러 시트로 저장
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 기존 결과 시트
        out_df.to_excel(writer, sheet_name='회사별 공통행', index=False)
        
        # 임원 현황 시트 - 지정된 컬럼 순서로 생성
        column_order = [
            "회사", "종목코드", "성명", "담당업무", "주요경력", "학교", "학과", "교수"
        ]
        
        if all_executives:
            exec_df = pd.DataFrame(all_executives)
            # 지정된 컬럼만 선택 (존재하지 않는 컬럼은 빈 값으로 채움)
            for col in column_order:
                if col not in exec_df.columns:
                    exec_df[col] = ""
            exec_df = exec_df[column_order]
            exec_df.to_excel(writer, sheet_name='임원 현황', index=False)
            print(f"\n임원 정보: {len(all_executives)}명 추출")
        else:
            # 빈 시트 생성 (지정된 컬럼만 포함)
            empty_df = pd.DataFrame(columns=column_order)
            empty_df.to_excel(writer, sheet_name='임원 현황', index=False)
    
    # Print summary
    summary = out_df["구분"].value_counts()
    print(f"\n=== Summary ===")
    print(summary)
    print(f"\nTotal processed: {len(results)}")
    print(f"Saved to: {output_path}")
    
    return output_path