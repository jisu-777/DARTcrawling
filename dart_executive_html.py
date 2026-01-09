"""
DART 보고서 HTML(viewer.do) 또는 XML에서 "가. 등기임원" 테이블을 파싱하여 임원 정보를 추출합니다.
"""
from typing import List, Dict, Optional
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import zipfile
import io
import xml.etree.ElementTree as ET
import asyncio
import json
import os
from openai import AsyncOpenAI


def get_viewer_url(main_url: str, debug: bool = False) -> Optional[str]:
    """
    main.do HTML에서 iframe(id=ifrm)의 src(viewer.do URL)를 추출합니다.
    여러 방법을 시도합니다: iframe src, JavaScript viewDoc, 정규식 패턴 매칭.
    
    Args:
        main_url: main.do URL
        debug: 디버그 로그 출력 여부
    
    Returns:
        viewer.do URL (절대경로) 또는 None
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(main_url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        if debug:
            print(f"  [DEBUG] Failed to fetch main.do: {e}")
        return None
    
    # 방법 1: iframe 태그에서 직접 찾기
    soup = BeautifulSoup(html, 'html.parser')
    iframe = soup.find('iframe', {'id': 'ifrm'}) or soup.find('iframe')
    
    if iframe:
        viewer_src = iframe.get('src', '')
        if viewer_src:
            viewer_url = _normalize_viewer_url(viewer_src)
            if viewer_url:
                if debug:
                    print(f"  [DEBUG] viewer_url (iframe): {viewer_url}")
                return viewer_url
    
    # 방법 2: 정규식으로 iframe src 찾기
    match = re.search(r'<iframe[^>]+src=["\']([^"\']*viewer\.do[^"\']*)["\']', html, re.IGNORECASE)
    if match:
        viewer_src = match.group(1)
        viewer_url = _normalize_viewer_url(viewer_src)
        if viewer_url:
            if debug:
                print(f"  [DEBUG] viewer_url (regex iframe): {viewer_url}")
            return viewer_url
    
    # 방법 3: JavaScript에서 viewDoc 호출 찾기
    # viewDoc(rcpNo, dcmNo, eleId, offset, length, dtd) 패턴
    viewdoc_patterns = [
        r'viewDoc\s*\(\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?([^"\']+)["\']?\s*\)',
        r'viewDoc\s*\(\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*,\s*["\']?(\d+)["\']?\s*\)',
    ]
    
    for pattern in viewdoc_patterns:
        match = re.search(pattern, html)
        if match:
            groups = match.groups()
            if len(groups) >= 5:
                rcp_no = groups[0]
                dcm_no = groups[1]
                ele_id = groups[2]
                offset = groups[3]
                length = groups[4]
                dtd = groups[5] if len(groups) > 5 else "dart3.xsd"
                
                viewer_url = f"https://dart.fss.or.kr/report/viewer.do?rcpNo={rcp_no}&dcmNo={dcm_no}&eleId={ele_id}&offset={offset}&length={length}&dtd={dtd}"
                if debug:
                    print(f"  [DEBUG] viewer_url (viewDoc): {viewer_url}")
                return viewer_url
    
    # 방법 4: HTML에서 viewer.do 관련 링크나 스크립트 찾기
    # viewer.do가 포함된 모든 링크 찾기 (JavaScript 코드 제외)
    viewer_links = re.findall(r'["\']([^"\']*viewer\.do[^"\']*)["\']', html, re.IGNORECASE)
    for link in viewer_links:
        # JavaScript 코드나 잘못된 URL 제외
        if 'viewer.do' in link and not link.startswith(');') and '?' in link:
            # URL에 파라미터가 있는지 확인 (실제 URL인지 확인)
            if 'rcpNo=' in link or 'dcmNo=' in link:
                viewer_url = _normalize_viewer_url(link)
                if viewer_url and viewer_url.startswith('http'):
                    if debug:
                        print(f"  [DEBUG] viewer_url (link found): {viewer_url}")
                    return viewer_url
    
    # 방법 5: main.do URL에서 rcpNo 추출하여 직접 viewer.do 구성 시도
    # (이 방법은 dcmNo 등을 모르므로 마지막 수단)
    rcp_match = re.search(r'rcpNo=(\d+)', main_url)
    if rcp_match:
        rcp_no = rcp_match.group(1)
        # 기본 파라미터로 시도 (실패할 수 있음)
        viewer_url = f"https://dart.fss.or.kr/report/viewer.do?rcpNo={rcp_no}"
        if debug:
            print(f"  [DEBUG] viewer_url (fallback from rcpNo): {viewer_url}")
        return viewer_url
    
    if debug:
        print(f"  [DEBUG] viewer.do URL 추출 실패 (모든 방법 시도)")
    return None


def _normalize_viewer_url(viewer_src: str) -> Optional[str]:
    """
    viewer.do 상대경로를 절대경로로 변환합니다.
    
    Args:
        viewer_src: viewer.do 상대경로 또는 절대경로
    
    Returns:
        절대경로 URL 또는 None
    """
    if not viewer_src:
        return None
    
    # 이미 절대경로인 경우
    if viewer_src.startswith('http'):
        return viewer_src
    
    # 상대경로 처리
    if viewer_src.startswith('/'):
        return f"https://dart.fss.or.kr{viewer_src}"
    elif viewer_src.startswith('viewer.do') or '/viewer.do' in viewer_src:
        return f"https://dart.fss.or.kr/report/{viewer_src.lstrip('/')}"
    else:
        return f"https://dart.fss.or.kr/report/{viewer_src}"


def _normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다 (따옴표 제거, 공백/개행/탭 제거).
    
    Args:
        text: 원본 텍스트
    
    Returns:
        정규화된 텍스트
    """
    # 따옴표 제거
    text = text.replace('"', '').replace("'", '')
    # 공백/개행/탭 제거
    text = re.sub(r"\s+", "", text)
    return text


def _count_tokens(normalized_text: str) -> int:
    """
    정규화된 텍스트에서 토큰 개수를 세어 반환합니다.
    
    Args:
        normalized_text: 정규화된 텍스트
    
    Returns:
        토큰 개수
    """
    tokens = [
        "성명", "성별", "출생년월", "직위", "등기임원", "상근",
        "담당", "주요경력", "소유주식수", "의결권있는주식", "의결권없는주식",
        "최대주주와의관계", "재직기간", "임기만료일"
    ]
    count = sum(1 for token in tokens if token in normalized_text)
    return count


def parse_registered_executives_from_xml(xml_text: str, debug: bool = False) -> Optional[pd.DataFrame]:
    """
    XML 텍스트에서 "가. 등기임원" 또는 "가. 임원 현황" 테이블을 파싱하여 DataFrame으로 반환합니다.
    
    Args:
        xml_text: XML 텍스트 내용
        debug: 디버그 로그 출력 여부
    
    Returns:
        파싱된 DataFrame 또는 None
    """
    try:
        if debug:
            print(f"  [DEBUG] XML 텍스트 크기: {len(xml_text)} bytes")
            # XML 내용의 일부 확인
            xml_preview = xml_text[:500] if len(xml_text) > 500 else xml_text
            print(f"  [DEBUG] XML 시작 부분: {xml_preview[:200]}...")
        
        # BeautifulSoup으로 XML 파싱 (더 관대한 파서)
        # 'html.parser'를 사용하면 XML도 파싱 가능하고 더 관대함
        # 경고 무시
        import warnings
        from bs4 import XMLParsedAsHTMLWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(xml_text, 'html.parser')
        
        # "가. 등기임원" 또는 "가. 임원 현황" 텍스트를 포함하는 요소 찾기
        target_table = None
        keywords = ["가. 등기임원", "가. 임원 현황", "등기임원", "임원 현황"]
        
        # 모든 텍스트 노드에서 키워드 찾기
        for keyword in keywords:
            # 텍스트로 검색 (정규식 사용)
            pattern = re.compile(re.escape(keyword))
            for element in soup.find_all(string=pattern):
                if debug:
                    print(f"  [DEBUG] 키워드 '{keyword}' 발견")
                
                # 이 요소 이후의 첫 번째 table 찾기
                for table in element.find_all_next('table'):
                    table_text = ''.join(table.stripped_strings)
                    norm_text = _normalize_text(table_text)
                    
                    # 토큰 개수 확인
                    token_count = _count_tokens(norm_text)
                    has_bonus = (
                        "성명" in norm_text and
                        "직위" in norm_text and
                        ("임기" in norm_text or "임기만료일" in norm_text)
                    )
                    
                    if debug:
                        print(f"  [DEBUG] 테이블 후보: 토큰={token_count}, 가산점={has_bonus}, 텍스트 샘플={norm_text[:100]}")
                    
                    if token_count >= 6 or has_bonus:
                        target_table = table
                        if debug:
                            print(f"  [DEBUG] XML에서 타겟 테이블 탐지 성공 (토큰: {token_count})")
                        break
                if target_table:
                    break
            if target_table:
                break
        
        if not target_table:
            if debug:
                # XML에서 "임원" 관련 텍스트가 있는지 확인
                all_text = soup.get_text()
                if "임원" in all_text:
                    print(f"  [DEBUG] XML에 '임원' 텍스트는 있지만 테이블을 찾지 못함")
                else:
                    print(f"  [DEBUG] XML에 '임원' 관련 텍스트가 없음")
            return None
        
        # 단순 파싱: tbody의 tr에서 td를 순서대로 추출
        rows = []
        
        if debug:
            table_tag = target_table.name
            print(f"  [DEBUG] 테이블 태그: {table_tag}")
        
        # tbody에서 모든 tr 찾기 (thead는 무시)
        tbody = target_table.find('tbody')
        if tbody:
            all_trs = tbody.find_all('tr')
        else:
            # tbody가 없으면 thead를 제외한 모든 tr 찾기
            thead = target_table.find('thead')
            if thead:
                all_trs_in_table = target_table.find_all('tr')
                thead_trs = set(thead.find_all('tr'))
                all_trs = [tr for tr in all_trs_in_table if tr not in thead_trs]
            else:
                all_trs = target_table.find_all('tr')
        
        if debug:
            print(f"  [DEBUG] 찾은 행 수: {len(all_trs)}")
            if all_trs:
                first_tr = all_trs[0]
                # tr의 모든 직접 자식 요소 확인
                direct_children = [c for c in first_tr.children if hasattr(c, 'name')]
                child_tags = [c.name for c in direct_children[:10]]
                print(f"  [DEBUG] 첫 번째 tr의 직접 자식 태그: {child_tags}")
                first_tds = first_tr.find_all('td')
                first_ths = first_tr.find_all('th')
                print(f"  [DEBUG] 첫 번째 tr의 td 수: {len(first_tds)}, th 수: {len(first_ths)}")
                if first_tds:
                    print(f"  [DEBUG] 첫 번째 tr의 td 샘플: {[td.get_text(strip=True)[:20] for td in first_tds[:3]]}")
        
        # 각 tr에서 셀 찾기 (td, th, 또는 다른 태그명)
        for i, tr in enumerate(all_trs):
            # 먼저 td 찾기
            tds = tr.find_all('td')
            
            # td가 없으면 th 찾기
            if not tds:
                ths = tr.find_all('th')
                if ths:
                    # 헤더 행이면 건너뛰기
                    if debug and i == 0:
                        print(f"  [DEBUG] 헤더 행으로 판단 (th만 있음), 건너뜀")
                    continue
                else:
                    # td도 th도 없으면 tr의 모든 직접 자식 요소에서 텍스트 추출
                    # XML에서는 태그명이 다를 수 있음
                    cells = []
                    for child in tr.children:
                        if hasattr(child, 'name') and child.name:
                            # 태그가 있으면 텍스트 추출
                            text = child.get_text(strip=True) if hasattr(child, 'get_text') else str(child).strip()
                            if text:
                                cells.append(text)
                    
                    if debug and i == 0:
                        print(f"  [DEBUG] td/th 없음, 자식 요소에서 추출한 셀 수: {len(cells)}")
                        if cells:
                            print(f"  [DEBUG] 첫 번째 셀 샘플: {cells[:3]}")
                    
                    if cells:
                        rows.append(cells)
                    continue
            
            row_data = [td.get_text(strip=True) for td in tds]
            
            # 빈 행이 아니고, 최소한 하나의 셀에 내용이 있으면 추가
            if row_data and any(cell.strip() for cell in row_data):
                rows.append(row_data)
        
        if debug:
            print(f"  [DEBUG] 추출된 데이터 행 수: {len(rows)}")
            if rows:
                print(f"  [DEBUG] 첫 번째 데이터 행 샘플: {rows[0][:5]}")
                print(f"  [DEBUG] 모든 행을 데이터로 처리 (첫 번째 행 포함): 총 {len(rows)}행")
        
        if not rows:
            if debug:
                print(f"  [DEBUG] 테이블에서 데이터 행을 찾지 못함")
            return None
        
        # 첫 번째 행도 데이터로 처리 (헤더로 인식하지 않음)
        # 모든 행을 AI가 분석하도록 함 - 첫 번째 행 제거하지 않음!
        header_row = None
        
        # 행 길이 맞추기
        if rows:
            max_cols = max(len(row) for row in rows)
            for i, row in enumerate(rows):
                rows[i] = row + [''] * (max_cols - len(row))
        
        # DataFrame 생성 (헤더 없이 숫자 인덱스로) - 모든 행 포함
        df = pd.DataFrame(rows)
        
        if debug:
            print(f"  [DEBUG] 직접 파싱 완료: {len(df)}행, {len(df.columns)}열 (첫 번째 행 포함)")
            if len(df) > 0:
                print(f"  [DEBUG] 첫 번째 행 샘플 (제거되지 않음): {list(df.iloc[0])[:5]}")
                print(f"  [DEBUG] 마지막 행 샘플: {list(df.iloc[-1])[:5] if len(df) > 1 else 'N/A'}")
        
        # MultiIndex 헤더 처리
        if isinstance(df.columns, pd.MultiIndex):
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    parts = [str(c).strip() for c in col if pd.notna(c) and str(c).strip()]
                    col_str = "".join(parts)
                    col_str = _normalize_text(col_str)
                    
                    if "소유" in col_str and "의결권" in col_str:
                        if "의결권있는주식" in col_str or "있는" in col_str or "O" in col_str:
                            col_str = "소유주식수(의결권O)"
                        elif "의결권없는주식" in col_str or "없는" in col_str or "X" in col_str:
                            col_str = "소유주식수(의결권X)"
                    new_columns.append(col_str)
                else:
                    new_columns.append(_normalize_text(str(col)))
            df.columns = new_columns
        else:
            df.columns = [_normalize_text(str(col)) for col in df.columns]
        
        if debug:
            print(f"  [DEBUG] XML 테이블 파싱 완료: {len(df)}행, 헤더: {list(df.columns)[:5]}")
        
        return df
    except Exception as e:
        if debug:
            import traceback
            print(f"  [DEBUG] XML 파싱 실패: {e}")
            print(f"  [DEBUG] 에러 상세: {traceback.format_exc()[:500]}")
        return None


def parse_registered_executives(viewer_html: str, debug: bool = False) -> Optional[pd.DataFrame]:
    """
    viewer.do HTML에서 "가. 등기임원" 테이블을 파싱하여 DataFrame으로 반환합니다.
    
    Args:
        viewer_html: viewer.do HTML 내용
        debug: 디버그 로그 출력 여부
    
    Returns:
        파싱된 DataFrame 또는 None
    """
    soup = BeautifulSoup(viewer_html, 'html.parser')
    all_tables = soup.find_all('table')
    
    target_table = None
    
    # 우선순위 1: "가.등기임원" 또는 "등기임원" 텍스트 이후 첫 번째 의미 있는 table
    keywords = ["가.등기임원", "등기임원"]
    
    for keyword in keywords:
        # 정규화된 키워드
        norm_keyword = _normalize_text(keyword)
        
        # 텍스트로 검색
        for element in soup.find_all(string=re.compile(re.escape(keyword.replace(".", "\\.")))):
            # 이 요소 이후의 모든 table 검색
            for table in element.find_all_next('table'):
                table_text = table.get_text()
                norm_text = _normalize_text(table_text)
                
                # 토큰 개수 확인 (최소 6개)
                token_count = _count_tokens(norm_text)
                
                # 가산점: 성명 + 직위 + 임기(또는 임기만료일) 동시 포함
                has_bonus = (
                    "성명" in norm_text and
                    "직위" in norm_text and
                    ("임기" in norm_text or "임기만료일" in norm_text)
                )
                
                if token_count >= 6 or has_bonus:
                    target_table = table
                    if debug:
                        print(f"  [DEBUG] 타겟 테이블 탐지 성공 (토큰: {token_count}, 가산점: {has_bonus})")
                    break
            if target_table:
                break
        if target_table:
            break
    
    # Fallback: 모든 table을 훑어서 성명+직위+임기(또는 재직기간) 포함하는 table 선택
    if not target_table:
        if debug:
            print(f"  [DEBUG] 우선순위 1 실패, fallback 시도 (전체 table 수: {len(all_tables)})")
        
        best_table = None
        best_score = 0
        
        for table in all_tables:
            table_text = table.get_text()
            norm_text = _normalize_text(table_text)
            
            # 성명과 직위가 있고 임기 또는 재직기간이 있는지 확인
            has_name = "성명" in norm_text
            has_position = "직위" in norm_text
            has_term = "임기" in norm_text or "재직기간" in norm_text
            
            if has_name and has_position and has_term:
                token_count = _count_tokens(norm_text)
                if token_count > best_score:
                    best_score = token_count
                    best_table = table
        
        if best_table:
            target_table = best_table
            if debug:
                print(f"  [DEBUG] Fallback으로 타겟 테이블 탐지 성공 (토큰: {best_score})")
    
    if not target_table:
        if debug:
            # 전체 table 개수와 상위 3개 토큰 수 출력
            table_scores = []
            for table in all_tables:
                table_text = table.get_text()
                norm_text = _normalize_text(table_text)
                token_count = _count_tokens(norm_text)
                table_scores.append((token_count, table))
            
            table_scores.sort(reverse=True, key=lambda x: x[0])
            top3 = table_scores[:3]
            print(f"  [DEBUG] 타겟 테이블 탐지 실패 (전체 table 수: {len(all_tables)})")
            print(f"  [DEBUG] 상위 3개 table 토큰 수: {[s[0] for s in top3]}")
            if top3:
                # 상위 1개 테이블의 정규화된 텍스트 일부 출력
                top_table = top3[0][1]
                top_text = _normalize_text(top_table.get_text()[:500])  # 처음 500자만
                print(f"  [DEBUG] 상위 테이블 텍스트 샘플: {top_text[:200]}")
        return None
    
    try:
        # BeautifulSoup으로 직접 테이블 파싱 (pandas.read_html 대신)
        rows = []
        header_row = None
        
        # thead가 있으면 헤더 행 찾기
        thead = target_table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            if header_rows:
                header_cells = []
                for hr in header_rows:
                    cells = hr.find_all(['td', 'th'])
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        if text:
                            header_cells.append(text)
                if header_cells:
                    header_row = header_cells
        
        # tbody 또는 모든 tr 찾기
        tbody = target_table.find('tbody')
        data_rows = tbody.find_all('tr') if tbody else target_table.find_all('tr')
        
        # thead가 있으면 헤더로 사용, 없으면 헤더 없이 모든 행을 데이터로 처리
        # 첫 번째 행도 데이터로 처리하여 AI가 모두 분석하도록 함
        
        # 데이터 행 파싱 (모든 행 포함)
        for tr in data_rows:
            cells = tr.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if row_data and any(cell.strip() for cell in row_data):  # 빈 행 제외
                rows.append(row_data)
        
        if not rows:
            if debug:
                print(f"  [DEBUG] 테이블에서 데이터 행을 찾지 못함")
            return None
        
        # DataFrame 생성 (thead가 있으면 헤더 사용, 없으면 모든 행을 데이터로 처리)
        if header_row:
            # 헤더와 행 길이 맞추기
            max_cols = max(len(header_row), max(len(row) for row in rows) if rows else 0)
            header_row = header_row + [''] * (max_cols - len(header_row))
            for i, row in enumerate(rows):
                rows[i] = row + [''] * (max_cols - len(row))
            df = pd.DataFrame(rows, columns=header_row)
        else:
            # 헤더가 없으면 모든 행을 데이터로 처리 (숫자 인덱스로 컬럼명 생성)
            max_cols = max(len(row) for row in rows) if rows else 0
            for i, row in enumerate(rows):
                rows[i] = row + [''] * (max_cols - len(row))
            df = pd.DataFrame(rows)
        
        if debug:
            print(f"  [DEBUG] 직접 파싱 완료: {len(df)}행, {len(df.columns)}열")
        
        # MultiIndex 헤더 처리
        if isinstance(df.columns, pd.MultiIndex):
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # 튜플 요소들을 strip 후 join (구분자 없이)
                    parts = [str(c).strip() for c in col if pd.notna(c) and str(c).strip()]
                    col_str = "".join(parts)
                    # 공백 제거
                    col_str = _normalize_text(col_str)
                    
                    # 소유주식수 매핑
                    if "소유" in col_str and "의결권" in col_str:
                        if "의결권있는주식" in col_str or "있는" in col_str or "O" in col_str:
                            col_str = "소유주식수(의결권O)"
                        elif "의결권없는주식" in col_str or "없는" in col_str or "X" in col_str:
                            col_str = "소유주식수(의결권X)"
                    
                    new_columns.append(col_str)
                else:
                    col_str = str(col)
                    new_columns.append(_normalize_text(col_str))
            df.columns = new_columns
        else:
            # 일반 컬럼도 정규화
            df.columns = [_normalize_text(str(col)) for col in df.columns]
        
        if debug:
            print(f"  [DEBUG] 파싱된 테이블 헤더 (일부): {list(df.columns)[:5]}")
            print(f"  [DEBUG] 파싱된 테이블 행 수: {len(df)}")
            if len(df) > 0:
                print(f"  [DEBUG] 첫 번째 행 샘플: {dict(df.iloc[0].head(5))}")
        
        return df
    except Exception as e:
        if debug:
            print(f"  [DEBUG] 테이블 파싱 실패: {e}")
        return None


def normalize_executive_row(row: pd.Series, company_meta: Dict) -> Dict:
    """
    DataFrame의 한 행을 최종 컬럼 형식으로 정규화합니다.
    
    Args:
        row: DataFrame의 한 행
        company_meta: 회사 메타데이터
    
    Returns:
        정규화된 임원 정보 딕셔너리 (성명이 없으면 None)
    """
    exec_data = {}
    
    # 회사 메타데이터 추가
    exec_data["회사"] = company_meta.get("회사", "")
    exec_data["종목코드"] = company_meta.get("종목코드", "")
    exec_data["구분"] = company_meta.get("구분", "")
    exec_data["url"] = company_meta.get("url", "")
    
    # 컬럼명을 정규화해서 매핑 (순서대로 되어 있으므로 인덱스도 고려)
    def get_value(*token_keys):
        """
        토큰 기반으로 값을 찾습니다.
        token_keys는 찾을 토큰 리스트입니다 (예: ["성명"] -> "성명" 포함 컬럼에서 찾음).
        """
        for token_key in token_keys:
            norm_token = _normalize_text(token_key)
            for col in row.index:
                col_str = str(col)
                norm_col = _normalize_text(col_str)
                
                # 토큰이 컬럼명에 포함되어 있는지 확인
                if norm_token in norm_col:
                    val = row[col]
                    if pd.notna(val):
                        return str(val).strip()
        return ""
    
    # 순서대로 매핑 (일반적인 순서: 성명, 성별, 출생년월, 직위, 등기임원여부, 상근여부, 담당업무, 주요경력, 소유주식수, 최대주주와의관계, 재직기간, 임기만료일, 의결권있는 주식, 의결권없는 주식)
    # 하지만 컬럼명이 있으므로 컬럼명으로 먼저 찾고, 없으면 인덱스로 접근
    exec_data["성명"] = get_value("성명", "이름", "name") or (str(row.iloc[0]).strip() if len(row) > 0 else "")
    exec_data["성별"] = get_value("성별", "gender", "sex") or (str(row.iloc[1]).strip() if len(row) > 1 else "")
    exec_data["출생년월"] = get_value("출생년월", "생년월일", "birth", "출생") or (str(row.iloc[2]).strip() if len(row) > 2 else "")
    exec_data["직위"] = get_value("직위", "position", "직책") or (str(row.iloc[3]).strip() if len(row) > 3 else "")
    exec_data["등기임원여부"] = get_value("등기임원") or (str(row.iloc[4]).strip() if len(row) > 4 else "")
    exec_data["상근여부"] = get_value("상근") or (str(row.iloc[5]).strip() if len(row) > 5 else "")
    exec_data["담당업무"] = get_value("담당", "업무", "responsibilities") or (str(row.iloc[6]).strip() if len(row) > 6 else "")
    exec_data["주요경력"] = get_value("경력", "주요경력", "career") or (str(row.iloc[7]).strip() if len(row) > 7 else "")
    
    # 소유주식수 매핑 (의결권O/X 구분)
    voting_shares = get_value("의결권있는주식", "의결권O", "의결권있는", "voting")
    if not voting_shares:
        for col in row.index:
            col_str = str(col)
            norm_col = _normalize_text(col_str)
            if "소유" in norm_col and "의결권" in norm_col:
                if "있는" in norm_col or "O" in norm_col or "의결권있는주식" in norm_col:
                    val = row[col]
                    if pd.notna(val):
                        voting_shares = str(val).strip()
                        break
    if not voting_shares and len(row) > 12:
        voting_shares = str(row.iloc[12]).strip()
    exec_data["소유주식수(의결권O)"] = voting_shares
    
    non_voting_shares = get_value("의결권없는주식", "의결권X", "의결권없는", "non_voting")
    if not non_voting_shares:
        for col in row.index:
            col_str = str(col)
            norm_col = _normalize_text(col_str)
            if "소유" in norm_col and "의결권" in norm_col:
                if "없는" in norm_col or "X" in norm_col or "의결권없는주식" in norm_col:
                    val = row[col]
                    if pd.notna(val):
                        non_voting_shares = str(val).strip()
                        break
    if not non_voting_shares and len(row) > 13:
        non_voting_shares = str(row.iloc[13]).strip()
    exec_data["소유주식수(의결권X)"] = non_voting_shares
    
    exec_data["최대주주와의 관계"] = get_value("최대주주", "관계", "relationship", "주주") or (str(row.iloc[9]).strip() if len(row) > 9 else "")
    exec_data["재직기간"] = get_value("재직", "재직기간", "tenure", "기간") or (str(row.iloc[10]).strip() if len(row) > 10 else "")
    
    # 임기만료일
    term_expiration = get_value("임기", "만료", "expiration")
    if not term_expiration:
        for col in row.index:
            col_str = str(col)
            norm_col = _normalize_text(col_str)
            if "임기" in norm_col and "만료" in norm_col:
                val = row[col]
                if pd.notna(val):
                    term_expiration = str(val).strip()
                    break
    if not term_expiration and len(row) > 11:
        term_expiration = str(row.iloc[11]).strip()
    exec_data["임기만료일"] = term_expiration
    
    # 모든 최종 컬럼이 포함되었는지 확인 (없으면 빈 문자열로 채움)
    final_columns = [
        "회사", "종목코드", "구분", "url", "성명", "성별", "출생년월",
        "직위", "등기임원여부", "상근여부", "담당업무", "주요경력",
        "소유주식수(의결권O)", "소유주식수(의결권X)", "최대주주와의 관계",
        "재직기간", "임기만료일"
    ]
    for col in final_columns:
        if col not in exec_data:
            exec_data[col] = ""
    
    # 성명이 있는 경우에만 반환
    if exec_data["성명"]:
        return exec_data
    return None


def fetch_report_xml_from_api(rcp_no: str, api_key: str, base_url: str, debug: bool = False) -> Optional[List[str]]:
    """
    DART OpenAPI의 document.xml을 통해 보고서 전문 XML 리스트를 가져옵니다.
    document.xml은 ZIP 파일을 반환하며, 압축 해제하면 여러 XML 파일이 나옵니다.
    
    Args:
        rcp_no: 보고서 접수번호
        api_key: DART API 키
        base_url: DART API 베이스 URL
        debug: 디버그 로그 출력 여부
    
    Returns:
        XML 텍스트 리스트 또는 None
    """
    try:
        url = f"{base_url.rstrip('/')}/document.xml"
        params = {
            "crtfc_key": api_key,
            "rcept_no": rcp_no,
        }
        
        if debug:
            print(f"  [DEBUG] API로 보고서 전문 ZIP 가져오기: {url}")
        
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        
        # ZIP 파일인지 확인
        if r.content[:2] != b'PK':
            # ZIP이 아니면 XML 응답일 수 있음
            try:
                tree = ET.fromstring(r.text)
                status = tree.find('status')
                if status is not None and status.text != '000':
                    message = tree.find('message')
                    if debug:
                        print(f"  [DEBUG] API 오류: status={status.text}, message={message.text if message is not None else 'Unknown'}")
                    return None
            except:
                pass
            
            if debug:
                print(f"  [DEBUG] ZIP 파일이 아님, XML로 처리")
            return [r.text]
        
        # ZIP 파일 압축 해제
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        info_list = zf.infolist()
        fnames = sorted([info.filename for info in info_list])
        
        if debug:
            print(f"  [DEBUG] ZIP 파일 압축 해제: {len(fnames)}개 파일")
        
        xml_text_list = []
        for fname in fnames:
            xml_data = zf.read(fname)
            try:
                xml_text = xml_data.decode('euc-kr')
            except UnicodeDecodeError:
                try:
                    xml_text = xml_data.decode('utf-8')
                except UnicodeDecodeError:
                    xml_text = xml_data.decode('utf-8', errors='ignore')
            xml_text_list.append(xml_text)
        
        if debug:
            print(f"  [DEBUG] {len(xml_text_list)}개 XML 파일 추출 완료")
        
        return xml_text_list
    except zipfile.BadZipFile:
        # ZIP이 아닌 경우 XML 응답일 수 있음
        try:
            tree = ET.fromstring(r.text)
            status = tree.find('status')
            if status is not None and status.text != '000':
                message = tree.find('message')
                if debug:
                    print(f"  [DEBUG] API 오류: status={status.text}, message={message.text if message is not None else 'Unknown'}")
                return None
        except:
            pass
        
        # XML로 직접 처리
        if debug:
            print(f"  [DEBUG] ZIP이 아님, XML로 직접 처리")
        try:
            xml_text = r.text
            return [xml_text]
        except:
            pass
        
        if debug:
            print(f"  [DEBUG] document.xml 응답 처리 실패")
        return None
    except Exception as e:
        if debug:
            print(f"  [DEBUG] document.xml API 실패: {e}")
        return None


def extract_registered_executives(
    rcp_no: str,
    main_url: str,
    corp_name: str,
    stock_code: str,
    report_type: str,
    api_key: str = "",
    base_url: str = "",
    debug: bool = False
) -> Optional[pd.DataFrame]:
    """
    DART 보고서에서 등기임원 테이블을 추출합니다.
    DART OpenAPI document.xml을 통해 XML을 가져와 파싱합니다.
    
    Args:
        rcp_no: 보고서 접수번호
        main_url: main.do URL
        corp_name: 회사명
        stock_code: 종목코드
        report_type: 보고서 유형
        api_key: DART API 키
        base_url: DART API 베이스 URL
        debug: 디버그 로그 출력 여부
    
    Returns:
        파싱된 DataFrame 또는 None
    """
    df = None
    
    # DART OpenAPI document.xml로 보고서 전문 가져오기
    if api_key and base_url:
        xml_list = fetch_report_xml_from_api(rcp_no, api_key, base_url, debug=debug)
        if xml_list:
            # 각 XML 파일에서 테이블 찾기
            for xml_text in xml_list:
                df = parse_registered_executives_from_xml(xml_text, debug=debug)
                if df is not None and not df.empty:
                    if debug:
                        print(f"  [DEBUG] XML에서 임원 테이블 찾음: {len(df)}행")
                    break
    
    if df is None or df.empty:
        if debug:
            print(f"  [DEBUG] 테이블 파싱 결과: None 또는 빈 DataFrame")
        return None
    
    # 회사 메타데이터를 DataFrame에 추가
    df['_회사'] = corp_name
    df['_종목코드'] = stock_code
    df['_구분'] = report_type
    df['_url'] = main_url
    
    return df


async def map_executive_row_with_ai(
    row: pd.Series,
    company_meta: Dict,
    client: AsyncOpenAI,
    debug: bool = False
) -> Optional[Dict]:
    """
    AI를 사용하여 임원 행 데이터를 정규화된 형식으로 매핑합니다.
    
    Args:
        row: DataFrame의 한 행
        company_meta: 회사 메타데이터
        client: OpenAI AsyncClient
        debug: 디버그 로그 출력 여부
    
    Returns:
        정규화된 임원 정보 딕셔너리 또는 None
    """
    try:
        # 행 데이터를 딕셔너리로 변환 (메타데이터 컬럼 제외)
        # row가 Series이므로 to_dict()로 변환
        row_dict_raw = row.to_dict()
        row_dict = {}
        
        for col, val in row_dict_raw.items():
            if str(col).startswith('_'):
                continue
            
            # 값이 유효한지 확인 (Series 체크 제거, 이미 dict이므로)
            if val is not None and pd.notna(val):
                val_str = str(val).strip()
                if val_str:
                    row_dict[str(col)] = val_str
        
        if not row_dict:
            return None
        
        # 컬럼명 정보도 포함
        columns_info = list(row.index)
        columns_info = [str(c) for c in columns_info if not str(c).startswith('_')]
        
        # AI 프롬프트 구성
        prompt = f"""다음은 DART 보고서에서 추출한 임원 정보 테이블의 한 행입니다.
컬럼명: {json.dumps(columns_info, ensure_ascii=False)}
행 데이터: {json.dumps(row_dict, ensure_ascii=False)}

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

JSON만 반환하고 다른 설명은 포함하지 마세요."""

        # API 호출 재시도 로직
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 DART 보고서의 임원 정보를 정규화된 형식으로 매핑하는 전문가입니다. JSON 형식으로만 응답하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=30.0
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    if debug:
                        print(f"  [DEBUG] API 호출 실패 (시도 {attempt + 1}/{max_retries}), 재시도 중...")
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    raise
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 (코드 블록 제거)
        if result_text.startswith("```"):
            # 코드 블록 제거
            lines = result_text.split("\n")
            result_text = "\n".join([line for line in lines if not line.strip().startswith("```")])
        
        mapped_data = json.loads(result_text)
        
        # 필수 필드 확인 및 기본값 설정
        required_fields = [
            "성명", "성별", "출생년월", "직위", "등기임원여부", "상근여부",
            "담당업무", "주요경력", "소유주식수(의결권O)", "소유주식수(의결권X)",
            "최대주주와의 관계", "재직기간", "임기만료일"
        ]
        
        # 누락된 필드 추가
        for field in required_fields:
            if field not in mapped_data:
                mapped_data[field] = ""
        
        # 회사 메타데이터 추가
        final_data = {
            "회사": company_meta.get("회사", ""),
            "종목코드": company_meta.get("종목코드", ""),
            "구분": company_meta.get("구분", ""),
            "url": company_meta.get("url", ""),
            **mapped_data
        }
        
        # 성명이 있는 경우에만 반환
        if final_data.get("성명") and final_data.get("성명").strip():
            return final_data
        
        return None
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"  [DEBUG] AI 응답 JSON 파싱 실패: {e}")
        return None
    except Exception as e:
        if debug:
            print(f"  [DEBUG] AI 매핑 실패: {e}")
        return None


async def map_executives_with_ai(
    df: pd.DataFrame,
    company_meta: Dict,
    openai_api_key: str,
    debug: bool = False,
    max_concurrent: int = 10
) -> List[Dict]:
    """
    DataFrame의 모든 행을 AI를 사용하여 병렬로 매핑합니다.
    
    Args:
        df: 임원 테이블 DataFrame
        company_meta: 회사 메타데이터
        openai_api_key: OpenAI API 키
        debug: 디버그 로그 출력 여부
        max_concurrent: 최대 동시 요청 수
    
    Returns:
        정규화된 임원 정보 리스트
    """
    if df is None or df.empty:
        return []
    
    client = AsyncOpenAI(api_key=openai_api_key)
    executives = []
    
    # 세마포어로 동시 요청 수 제한 (너무 많은 동시 요청 방지)
    semaphore = asyncio.Semaphore(min(max_concurrent, 7))  # 최대 5개로 제한
    
    async def process_row(idx, row):
        async with semaphore:
            try:
                if debug and (idx == 0 or idx % 5 == 0):
                    print(f"  [AI 매핑] {idx+1}/{len(df)} 처리 중...")
                # row는 이미 Series이므로 그대로 전달
                result = await map_executive_row_with_ai(row, company_meta, client, debug=debug)
                return result
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] 행 {idx} 처리 중 오류: {type(e).__name__}: {e}")
                return None
    
    # 모든 행을 병렬로 처리
    tasks = [process_row(idx, row) for idx, row in df.iterrows()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 결과 수집
    for result in results:
        if isinstance(result, Exception):
            if debug:
                print(f"  [DEBUG] 행 처리 중 예외: {type(result).__name__}: {result}")
            continue
        if result is not None:
            executives.append(result)
    
    if debug:
        print(f"  [DEBUG] AI 매핑 완료: {len(executives)}명 추출")
    
    return executives
