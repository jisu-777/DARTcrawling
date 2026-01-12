"""
동시 요청 테스트 스크립트
실제로 50개 동시 요청이 작동하는지 확인
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DART_API_KEY", "").strip()
base_url = os.getenv("DART_API_BASE_URL", "").strip()

def test_request(corp_code: str, idx: int):
    """단일 요청 테스트"""
    thread_id = threading.current_thread().ident
    start = time.time()
    
    session = requests.Session()
    try:
        url = f"{base_url.rstrip('/')}/list.json"
        params = {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": "20240101",
            "end_de": "20260101",
            "pblntf_ty": "A",
            "page_no": 1,
            "page_count": 10,
        }
        
        response = session.get(url, params=params, timeout=20)
        elapsed = time.time() - start
        
        print(f"[{idx}] Thread-{thread_id}: {elapsed:.2f}s - Status: {response.status_code}")
        return {"idx": idx, "elapsed": elapsed, "status": response.status_code}
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{idx}] Thread-{thread_id}: {elapsed:.2f}s - Error: {e}")
        return {"idx": idx, "elapsed": elapsed, "error": str(e)}
    finally:
        session.close()

# 테스트용 corp_code 리스트 (10개)
test_codes = [
    "00126380", "00258801", "01244601", "01133217", "01205709",
    "00427483", "00161383", "00291231", "00126380", "00258801"
] * 5  # 50개로 확장

if __name__ == "__main__":
    print(f"Testing {len(test_codes)} concurrent requests with ThreadPoolExecutor(max_workers=50)")
    print("=" * 60)
    
    start_time = time.time()
    active_count = 0
    lock = threading.Lock()
    
    def track_active():
        global active_count
        with lock:
            active_count += 1
            print(f"  [ACTIVE] {active_count} requests running simultaneously")
    
    def track_complete():
        global active_count
        with lock:
            active_count -= 1
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {}
        for idx, corp_code in enumerate(test_codes):
            future = executor.submit(test_request, corp_code, idx + 1)
            futures[future] = idx + 1
            track_active()
        
        results = []
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results.append(result)
            track_complete()
    
    total_time = time.time() - start_time
    avg_time = sum(r.get("elapsed", 0) for r in results) / len(results) if results else 0
    
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average request time: {avg_time:.2f}s")
    print(f"Throughput: {len(test_codes) / total_time:.2f} requests/sec")
    print(f"Expected sequential time: {avg_time * len(test_codes):.2f}s")
    print(f"Speedup: {avg_time * len(test_codes) / total_time:.2f}x")

