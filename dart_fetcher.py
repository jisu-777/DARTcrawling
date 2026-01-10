import os
import re
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from dart_executive_html import extract_registered_executives, map_executives_with_ai

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
    all_executive_dfs = []  # DataFrame 리스트로 변경
    total = len(df)
    
    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    
    # 단일 루프로 통합: 각 회사에 대해 보고서 조회 및 임원 정보 수집
    for idx, row in df.iterrows():
        corp_code_raw = str(row.get("corp_code", "")).strip()
        corp_name = str(row.get("corp_name", "")).strip()
        stock_code = normalize_stock_code(row.get("stock_code"))  # Only for output display
        
        # Normalize corp_code to 8 digits
        corp_code = normalize_corp_code(corp_code_raw)
        
        # Skip invalid corp_code
        if not corp_code or len(corp_code) != 8:
            if debug:
                print(f"[{idx+1}/{total}] Skipping row {idx+1}: invalid corp_code (raw: {corp_code_raw}, normalized: {corp_code})")
            results.append({"회사": corp_name, "종목코드": stock_code, "구분": "코드오류", "url": ""})
            continue

        if debug:
            print(f"[{idx+1}/{total}] Processing: {corp_name} (corp_code: {corp_code}, stock_code: {stock_code})")

        # Use corp_code only (must be 8 digits)
        report_meta = fetch_latest_periodic_report(
            corp_code=corp_code,
            api_key=api_key,
            base_url=base_url,
            years_back=years_back,
            debug=debug,
        )

        if report_meta is None:
            results.append({"회사": corp_name, "종목코드": stock_code, "구분": "정보없음", "url": ""})
            if debug:
                print(f"  -> No report found")
        else:
            results.append({
                "회사": corp_name,
                "종목코드": stock_code,
                "구분": report_meta["type"],
                "url": report_meta["url"]
            })
            if debug:
                print(f"  -> Found: {report_meta['type']} (rcept_no: {report_meta['rcept_no']})")
            
            # 임원 테이블 추출 (DataFrame 반환)
            if report_meta.get("rcept_no") and report_meta.get("url"):
                if debug:
                    print(f"  [임원정보 테이블 추출] {corp_name} (rcp_no: {report_meta['rcept_no']})")
                
                try:
                    exec_df = extract_registered_executives(
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
                        # 회사 메타데이터와 함께 저장
                        all_executive_dfs.append({
                            "df": exec_df,
                            "company_meta": {
                                "회사": corp_name,
                                "종목코드": stock_code,
                                "구분": report_meta["type"],
                                "url": report_meta["url"]
                            }
                        })
                        if debug:
                            print(f"  -> 테이블 추출 성공: {len(exec_df)}행")
                    else:
                        if debug:
                            print(f"  -> 테이블 추출 실패: 빈 DataFrame")
                except Exception as e:
                    if debug:
                        print(f"  -> 테이블 추출 실패: {e}")
    
    # 모든 DataFrame을 AI로 병렬 매핑
    if all_executive_dfs:
        if debug:
            print(f"\n[AI 매핑 시작] 총 {len(all_executive_dfs)}개 회사의 임원 정보를 AI로 매핑합니다...")
        
        async def process_all_executives():
            all_executives = []
            tasks = []
            
            for item in all_executive_dfs:
                task = map_executives_with_ai(
                    df=item["df"],
                    company_meta=item["company_meta"],
                    openai_api_key=openai_api_key,
                    debug=debug,
                    max_concurrent=3  # 동시 요청 수를 줄여서 Connection error 방지
                )
                tasks.append(task)
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results_list:
                if isinstance(result, Exception):
                    if debug:
                        print(f"  [DEBUG] AI 매핑 중 오류: {result}")
                    continue
                if isinstance(result, list):
                    all_executives.extend(result)
            
            return all_executives
        
        # 비동기 실행
        all_executives = asyncio.run(process_all_executives())
        
        if debug:
            print(f"\n[AI 매핑 완료] 총 {len(all_executives)}명의 임원 정보 추출")
    else:
        all_executives = []

    out_df = pd.DataFrame(results, columns=["회사", "종목코드", "구분", "url"])
    
    # Excel 파일에 여러 시트로 저장
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 기존 결과 시트
        out_df.to_excel(writer, sheet_name='회사별 공통행', index=False)
        
        # 임원 현황 시트 - 지정된 컬럼 순서로 생성
        column_order = [
            "회사", "url", "성명", "담당업무", "주요경력", "학교", "학과"
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