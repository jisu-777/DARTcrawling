"""
Stage 1: DART API를 통한 원천 데이터 수집
- requests + ThreadPoolExecutor로 HTTP I/O 처리 (최대 50개 동시)
- ProcessPoolExecutor로 CPU 파싱 처리
- HTTP와 CPU 작업을 완전히 분리
"""
import os
import json
import argparse
import queue
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import multiprocessing

import pandas as pd
import requests
import zipfile
import io
from dotenv import load_dotenv

from utils import normalize_corp_code, normalize_stock_code, normalize_input_dataframe
from dart_executive_html import parse_registered_executives_from_xml

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DART_MAIN_URL = "https://dart.fss.or.kr/dsaf001/main.do"


def extract_rows_from_xml(xml_text: str, xml_index: int, table_hint: str = "") -> List[Dict[str, Any]]:
    """
    XML에서 테이블을 파싱하여 row 단위 dict 리스트로 반환.
    ProcessPoolExecutor에서 실행되므로 독립적인 함수여야 함.
    
    Returns:
        List of dicts with keys: columns_json, row_json, row_text
    """
    try:
        df = parse_registered_executives_from_xml(xml_text, debug=False)
        if df is None or df.empty:
            return []
        
        rows = []
        for idx, row in df.iterrows():
            # 컬럼명과 값을 dict로 변환
            row_dict = row.to_dict()
            # None/NaN 제거
            row_dict = {k: ("" if pd.isna(v) else str(v).strip()) for k, v in row_dict.items()}
            
            # JSON 문자열로 변환
            columns_json = json.dumps(list(df.columns.tolist()), ensure_ascii=False)
            row_json = json.dumps(row_dict, ensure_ascii=False)
            
            # 사람이 읽을 수 있는 텍스트 (디버깅용)
            row_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items() if v])
            
            rows.append({
                "xml_index": xml_index,
                "table_hint": table_hint,
                "row_index": int(idx),
                "columns_json": columns_json,
                "row_json": row_json,
                "row_text": row_text,
            })
        
        return rows
    except Exception as e:
        logger.error(f"XML 파싱 오류 (xml_index={xml_index}): {e}")
        return []


def extract_xml_from_zip(zip_content: bytes) -> List[Tuple[int, str, str]]:
    """
    ZIP 파일에서 XML 텍스트 리스트 추출.
    ProcessPoolExecutor에서 실행.
    
    Returns:
        List of (xml_index, xml_text, table_hint) tuples
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_content))
        fnames = sorted([info.filename for info in zf.infolist()])
        
        xml_list = []
        for idx, fname in enumerate(fnames):
            try:
                xml_data = zf.read(fname)
                try:
                    xml_text = xml_data.decode('euc-kr')
                except UnicodeDecodeError:
                    try:
                        xml_text = xml_data.decode('utf-8')
                    except UnicodeDecodeError:
                        xml_text = xml_data.decode('utf-8', errors='ignore')
                
                # 간단히 키워드 확인 (정확한 파싱은 extract_rows_from_xml에서)
                table_hint = ""
                if "등기임원" in xml_text or "임원 현황" in xml_text:
                    if "가. 등기임원" in xml_text:
                        table_hint = "가. 등기임원"
                    elif "가. 임원 현황" in xml_text:
                        table_hint = "가. 임원 현황"
                    elif "등기임원" in xml_text:
                        table_hint = "등기임원"
                    else:
                        table_hint = "임원 현황"
                
                xml_list.append((idx, xml_text, table_hint))
            except Exception as e:
                logger.error(f"Failed to extract XML from {fname}: {e}")
                continue
        
        return xml_list
    except Exception as e:
        logger.error(f"Failed to extract ZIP: {e}")
        return []


def fetch_list_json(session: requests.Session, api_key: str, base_url: str, 
                    corp_code: str, years_back: int) -> Optional[Dict]:
    """list.json을 동기로 가져오기 (ThreadPoolExecutor에서 실행)"""
    end_de = datetime.today().strftime("%Y%m%d")
    bgn_de = (datetime.today() - timedelta(days=365 * years_back)).strftime("%Y%m%d")
    
    url = f"{base_url.rstrip('/')}/list.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "pblntf_ty": "A",
        "page_no": 1,
        "page_count": 100,
    }
    
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            status = str(data.get("status", ""))
            if status != "000":
                if status in ["013", "014"]:
                    return None
                if attempt == 2:
                    logger.warning(f"API error for corp_code={corp_code}: status={status}")
                    return None
                time.sleep(1)
                continue
            
            return data
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
                continue
            logger.error(f"Request error for corp_code={corp_code}: {e}")
            return None
    
    return None


def select_latest_report(all_items: List[Dict]) -> Optional[Dict]:
    """분기 > 반기 > 연간 순으로 최신 보고서 선택"""
    if not all_items:
        return None
    
    def is_quarterly(name: str) -> bool:
        return "분기보고서" in name or "분기" in name
    
    def is_halfyear(name: str) -> bool:
        return "반기보고서" in name or "반기" in name
    
    def is_annual(name: str) -> bool:
        return "사업보고서" in name or "연간" in name
    
    items_sorted = sorted(
        all_items,
        key=lambda x: (x.get("rcept_dt", ""), x.get("rcept_no", "")),
        reverse=True,
    )
    
    # 분기 > 반기 > 연간 순
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_quarterly(nm):
            return {"type": "분기", **it}
    
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_halfyear(nm):
            return {"type": "반기", **it}
    
    for it in items_sorted:
        nm = it.get("report_nm", "") or ""
        if is_annual(nm):
            return {"type": "연간", **it}
    
    return None


def fetch_document_xml(session: requests.Session, api_key: str, base_url: str, 
                       rcept_no: str) -> Optional[bytes]:
    """document.xml ZIP을 동기로 가져오기 (ThreadPoolExecutor에서 실행)"""
    url = f"{base_url.rstrip('/')}/document.xml"
    params = {
        "crtfc_key": api_key,
        "rcept_no": rcept_no,
    }
    
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.content
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
                continue
            logger.error(f"Failed to fetch document.xml for rcept_no={rcept_no}: {e}")
            return None
    
    return None


def process_company_http(
    api_key: str,
    base_url: str,
    corp_code: str,
    corp_name: str,
    stock_code: str,
    years_back: int,
    cpu_queue: queue.Queue,
) -> Optional[Dict]:
    """
    회사 1개 처리: HTTP 요청만 수행 (ThreadPoolExecutor에서 실행)
    CPU 파싱은 큐에 넣고 반환
    """
    thread_id = threading.current_thread().ident
    start_time = time.time()
    
    # 스레드별 Session 생성 (커넥션 재사용)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    try:
        company_id = normalize_corp_code(corp_code)
        if not company_id or len(company_id) != 8:
            logger.warning(f"Invalid corp_code: {corp_code}")
            return None
        
        # 1. list.json 가져오기
        list_start = time.time()
        data = fetch_list_json(session, api_key, base_url, company_id, years_back)
        list_time = time.time() - list_start
        
        if not data:
            return None
        
        all_items = data.get("list", [])
        if not all_items:
            return None
        
        # 2. 최신 보고서 선택
        report = select_latest_report(all_items)
        if not report:
            return None
        
        rcept_no = report.get("rcept_no", "")
        report_type = report.get("type", "")
        url = f"{DART_MAIN_URL}?rcpNo={rcept_no}"
        
        # 3. document.xml ZIP 가져오기
        doc_start = time.time()
        zip_content = fetch_document_xml(session, api_key, base_url, rcept_no)
        doc_time = time.time() - doc_start
        
        if not zip_content:
            return None
        
        total_time = time.time() - start_time
        logger.debug(
            f"[Thread-{thread_id}] {corp_name}: "
            f"list.json={list_time:.2f}s, document.xml={doc_time:.2f}s, total={total_time:.2f}s"
        )
        
        # 4. CPU 파싱 큐에 넣기 (즉시 파싱하지 않음)
        cpu_queue.put({
            "company_id": company_id,
            "회사": corp_name,
            "종목코드": stock_code,
            "구분": report_type,
            "url": url,
            "rcept_no": rcept_no,
            "zip_content": zip_content,
        })
        
        return {
            "company_id": company_id,
            "회사": corp_name,
            "status": "collected",
        }
    except Exception as e:
        logger.error(f"Error processing {corp_name}: {e}")
        return None
    finally:
        session.close()


def process_cpu_queue(
    cpu_queue: queue.Queue,
    result_queue: queue.Queue,
    cpu_executor: ProcessPoolExecutor,
    stop_event: threading.Event,
):
    """
    CPU 파싱 워커: 큐에서 ZIP을 가져와서 파싱
    별도 스레드에서 실행
    """
    while not stop_event.is_set() or not cpu_queue.empty():
        try:
            # 타임아웃으로 stop_event 체크 가능하게
            item = cpu_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        try:
            company_id = item["company_id"]
            zip_content = item["zip_content"]
            
            # ZIP에서 XML 추출 (ProcessPoolExecutor)
            future = cpu_executor.submit(extract_xml_from_zip, zip_content)
            xml_list = future.result(timeout=300)  # 5분 타임아웃
            
            if not xml_list:
                cpu_queue.task_done()
                continue
            
            # 각 XML에서 row 추출
            all_rows = []
            for xml_index, xml_text, table_hint in xml_list:
                # table_hint가 없어도 파싱 시도 (parse_registered_executives_from_xml이 내부에서 찾음)
                # CPU 작업: XML 파싱 (ProcessPoolExecutor)
                future = cpu_executor.submit(extract_rows_from_xml, xml_text, xml_index, table_hint or "")
                rows = future.result(timeout=300)
                
                # 메타데이터 추가
                for row in rows:
                    row.update({
                        "company_id": company_id,
                        "회사": item["회사"],
                        "종목코드": item["종목코드"],
                        "구분": item["구분"],
                        "url": item["url"],
                        "rcept_no": item["rcept_no"],
                        "source_xml_index": xml_index,
                        "collected_at": datetime.now().isoformat(),
                    })
                
                all_rows.extend(rows)
            
            # 결과 큐에 넣기
            if all_rows:
                result_queue.put(all_rows)
                logger.debug(f"CPU processed {len(all_rows)} rows for company_id={company_id}")
            
        except Exception as e:
            logger.error(f"CPU processing error for company_id={item.get('company_id')}: {e}")
        finally:
            cpu_queue.task_done()


def collect_all_companies(
    df: pd.DataFrame,
    api_key: str,
    base_url: str,
    years_back: int,
    workers_http: int,
    workers_cpu: int,
    output_path: str,
    checkpoint_interval: int = 100,
    resume: bool = False,
) -> None:
    """모든 회사 수집 (ThreadPoolExecutor + ProcessPoolExecutor)"""
    # 이미 수집된 company_id 확인 (resume 모드)
    collected_ids = set()
    if resume and os.path.exists(output_path):
        try:
            if output_path.endswith('.parquet'):
                existing_df = pd.read_parquet(output_path)
            else:
                existing_df = pd.read_csv(output_path)
            if 'company_id' in existing_df.columns:
                collected_ids = set(existing_df['company_id'].unique())
                logger.info(f"Resume mode: {len(collected_ids)} companies already collected")
        except Exception as e:
            logger.warning(f"Failed to read existing file for resume: {e}")
    
    # 큐 생성
    cpu_queue = queue.Queue(maxsize=workers_http * 2)  # 버퍼 크기
    result_queue = queue.Queue()
    stop_event = threading.Event()
    
    # ProcessPoolExecutor 생성 (CPU 코어 수 - 1, 최소 1)
    cpu_workers = max(1, min(workers_cpu, multiprocessing.cpu_count() - 1))
    cpu_executor = ProcessPoolExecutor(max_workers=cpu_workers)
    logger.info(f"CPU executor created with {cpu_workers} workers")
    
    # CPU 워커 스레드 시작 (여러 개)
    cpu_worker_threads = []
    for _ in range(workers_cpu):
        thread = threading.Thread(
            target=process_cpu_queue,
            args=(cpu_queue, result_queue, cpu_executor, stop_event),
            daemon=True
        )
        thread.start()
        cpu_worker_threads.append(thread)
    
    all_collected_rows = []
    completed = 0
    last_checkpoint = 0
    
    try:
        # HTTP 작업: ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers_http) as http_executor:
            # 모든 회사에 대해 HTTP 작업 제출
            futures = {}
            for idx, row in df.iterrows():
                corp_code = str(row.get("corp_code", "")).strip()
                company_id = normalize_corp_code(corp_code)
                
                # Resume 모드: 이미 수집된 회사는 스킵
                if resume and company_id in collected_ids:
                    logger.info(f"[{idx+1}/{len(df)}] Skipping {row.get('corp_name')} (already collected)")
                    continue
                
                future = http_executor.submit(
                    process_company_http,
                    api_key=api_key,
                    base_url=base_url,
                    corp_code=corp_code,
                    corp_name=str(row.get("corp_name", "")).strip(),
                    stock_code=normalize_stock_code(row.get("stock_code")),
                    years_back=years_back,
                    cpu_queue=cpu_queue,
                )
                futures[future] = (idx + 1, row.get("corp_name"))
            
            total_tasks = len(futures)
            logger.info(f"Submitted {total_tasks} HTTP tasks (max_workers={workers_http})")
            logger.info(f"Expected concurrent HTTP requests: up to {min(workers_http, total_tasks)}")
            
            # 진행률 추적
            start_time = time.time()
            last_log_time = start_time
            
            # HTTP 작업 완료 대기 및 결과 수집 (병렬로 CPU 결과도 수집)
            def collect_cpu_results():
                """별도 스레드에서 CPU 결과 수집 (메모리 효율적)"""
                nonlocal all_collected_rows, last_checkpoint
                batch_for_checkpoint = []
                
                while not stop_event.is_set() or not result_queue.empty():
                    try:
                        rows = result_queue.get(timeout=1)
                        all_collected_rows.extend(rows)
                        batch_for_checkpoint.extend(rows)
                        
                        # 체크포인트 저장 (일정량 모이면 저장하고 메모리에서 해제)
                        if len(batch_for_checkpoint) >= checkpoint_interval * 10:  # 대략적인 row 수
                            save_checkpoint(batch_for_checkpoint, output_path, append=True)
                            last_checkpoint += len(batch_for_checkpoint)
                            batch_for_checkpoint = []  # 메모리에서 해제
                    except queue.Empty:
                        continue
                
                # 남은 데이터 저장
                if batch_for_checkpoint:
                    save_checkpoint(batch_for_checkpoint, output_path, append=True)
            
            # CPU 결과 수집 스레드 시작
            collector_thread = threading.Thread(target=collect_cpu_results, daemon=True)
            collector_thread.start()
            
            # HTTP 작업 완료 대기
            active_tasks = set(futures.keys())
            first_completion_time = None
            for future in as_completed(futures):
                idx, corp_name = futures[future]
                active_tasks.discard(future)
                
                if first_completion_time is None:
                    first_completion_time = time.time()
                    first_elapsed = first_completion_time - start_time
                    logger.info(
                        f"✓ First HTTP request completed in {first_elapsed:.2f}s "
                        f"({len(active_tasks)} still active - CONCURRENCY CONFIRMED!)"
                    )
                
                try:
                    result = future.result()
                    if result:
                        completed += 1
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        
                        # 10개마다 또는 5초마다 로그
                        current_time = time.time()
                        if completed % 10 == 0 or (current_time - last_log_time) >= 5 or completed == total_tasks:
                            logger.info(
                                f"HTTP progress: {completed}/{total_tasks} companies "
                                f"({rate:.1f} companies/sec, {len(active_tasks)} active tasks)"
                            )
                            last_log_time = current_time
                except Exception as e:
                    logger.error(f"HTTP task failed for {corp_name}: {e}")
                    completed += 1
            
            total_elapsed = time.time() - start_time
            logger.info(
                f"✓ All HTTP tasks completed in {total_elapsed:.2f}s "
                f"(avg {total_elapsed/total_tasks:.2f}s per company, "
                f"throughput: {total_tasks/total_elapsed:.2f} companies/sec)"
            )
            
            logger.info(f"All HTTP tasks completed: {completed}/{total_tasks}")
            
            # CPU 큐에 더 이상 작업이 없을 때까지 대기
            cpu_queue.join()
            stop_event.set()
            
            # 남은 결과 수집
            while not result_queue.empty():
                rows = result_queue.get()
                all_collected_rows.extend(rows)
            
            # CPU 워커 스레드 및 수집 스레드 종료 대기
            for thread in cpu_worker_threads:
                thread.join(timeout=5)
            collector_thread.join(timeout=5)
        
        # 최종 결과 저장 (이미 체크포인트로 저장된 것 제외)
        if all_collected_rows:
            # 마지막으로 남은 데이터가 있으면 저장
            remaining = all_collected_rows[last_checkpoint:] if last_checkpoint < len(all_collected_rows) else []
            if remaining:
                save_checkpoint(remaining, output_path, append=True)
            logger.info(f"Final save complete: {len(all_collected_rows)} total rows to {output_path}")
        else:
            logger.warning("No rows collected")
    
    finally:
        cpu_executor.shutdown(wait=True)
        stop_event.set()


def save_checkpoint(rows: List[Dict], output_path: str, append: bool = True):
    """체크포인트 저장 (append 모드로 메모리 효율성 향상)"""
    if not rows:
        return
    
    try:
        result_df = pd.DataFrame(rows)
        if output_path.endswith('.parquet'):
            if append and os.path.exists(output_path):
                # 기존 파일 읽기
                existing_df = pd.read_parquet(output_path)
                # 병합
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
            result_df.to_parquet(output_path, index=False)
        else:
            if append and os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Checkpoint saved: {len(result_df)} total rows")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
    
    finally:
        cpu_executor.shutdown(wait=True)
        stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="Stage 1: DART API 데이터 수집")
    parser.add_argument("--input", required=True, help="Input Excel file (input.xlsx)")
    parser.add_argument("--out", required=True, help="Output file (collected_rows.parquet or .csv)")
    parser.add_argument("--years-back", type=int, default=3, help="Years to look back")
    parser.add_argument("--workers-http", type=int, default=50, help="HTTP concurrent workers")
    parser.add_argument("--workers-cpu", type=int, default=4, help="CPU process workers")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint every N companies")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    load_dotenv()
    api_key = os.getenv("DART_API_KEY", "").strip()
    base_url = os.getenv("DART_API_BASE_URL", "").strip()
    
    if not api_key or not base_url:
        raise RuntimeError("Missing DART_API_KEY or DART_API_BASE_URL in .env")
    
    # 입력 파일 읽기
    df = pd.read_excel(args.input)
    df = normalize_input_dataframe(df)
    
    logger.info(f"Starting collection for {len(df)} companies")
    logger.info(f"Output: {args.out}")
    logger.info(f"HTTP workers: {args.workers_http}, CPU workers: {args.workers_cpu}")
    
    collect_all_companies(
        df=df,
        api_key=api_key,
        base_url=base_url,
        years_back=args.years_back,
        workers_http=args.workers_http,
        workers_cpu=args.workers_cpu,
        output_path=args.out,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
    )
    
    logger.info("Stage 1 complete!")


if __name__ == "__main__":
    main()
