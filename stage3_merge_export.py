"""
Stage 3: 최종 엑셀 병합 및 출력
- Stage2 결과를 읽어서 엑셀 파일로 저장
- 회사별 공통행 시트와 임원 현황 시트 생성
"""
import argparse
import logging
from typing import Set, Tuple

import pandas as pd
from utils import normalize_input_dataframe

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_company_summary(input_df: pd.DataFrame, mapped_df: pd.DataFrame) -> pd.DataFrame:
    """회사별 공통행 시트 생성"""
    # mapped_df에서 회사별 정보 추출
    company_info = mapped_df.groupby('company_id').agg({
        '회사': 'first',
        '종목코드': 'first',
        '구분': 'first',
        'url': 'first',
    }).reset_index()
    
    # input_df와 병합하여 모든 회사 포함
    input_df = input_df.copy()
    input_df['company_id'] = input_df['corp_code'].apply(lambda x: str(x).zfill(8))
    
    summary = input_df[['company_id', 'corp_name', 'stock_code']].merge(
        company_info[['company_id', '구분', 'url']],
        on='company_id',
        how='left'
    )
    
    summary['회사'] = summary['corp_name']
    summary['종목코드'] = summary['stock_code'].apply(lambda x: str(x).zfill(6) if pd.notna(x) else "")
    summary['구분'] = summary['구분'].fillna('정보없음')
    summary['url'] = summary['url'].fillna('')
    
    return summary[['회사', '종목코드', '구분', 'url']]


def create_executive_sheet(mapped_df: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """임원 현황 시트 생성"""
    # 성명이 없는 행 제외
    exec_df = mapped_df[mapped_df['성명'].notna() & (mapped_df['성명'].str.strip() != '')].copy()
    
    if exec_df.empty:
        logger.warning("No executives found after filtering")
        return pd.DataFrame(columns=[
            "회사", "종목코드", "성명", "담당업무", "주요경력", "학교", "학과", "교수"
        ])
    
    # 중복 제거 (선택적)
    if dedupe:
        # (회사, 성명, 직위) 기준으로 중복 제거
        before_count = len(exec_df)
        exec_df = exec_df.drop_duplicates(
            subset=['회사', '성명', '직위'],
            keep='first'
        )
        after_count = len(exec_df)
        if before_count != after_count:
            logger.info(f"Deduplicated: {before_count} -> {after_count} rows")
    
    # 컬럼 순서 지정
    column_order = [
        "회사", "종목코드", "성명", "담당업무", "주요경력", "학교", "학과", "교수"
    ]
    
    # 존재하지 않는 컬럼은 빈 값으로 채움
    for col in column_order:
        if col not in exec_df.columns:
            exec_df[col] = ""
    
    return exec_df[column_order]


def merge_and_export(
    input_xlsx: str,
    mapped_file: str,
    output_xlsx: str,
    dedupe: bool = True,
) -> None:
    """최종 병합 및 엑셀 저장"""
    # 입력 파일 읽기
    input_df = pd.read_excel(input_xlsx)
    input_df = normalize_input_dataframe(input_df)
    logger.info(f"Loaded {len(input_df)} companies from {input_xlsx}")
    
    # 매핑 결과 읽기
    if mapped_file.endswith('.parquet'):
        mapped_df = pd.read_parquet(mapped_file)
    else:
        mapped_df = pd.read_csv(mapped_file)
    logger.info(f"Loaded {len(mapped_df)} mapped rows from {mapped_file}")
    
    # 회사별 공통행 생성
    company_summary = create_company_summary(input_df, mapped_df)
    logger.info(f"Created company summary: {len(company_summary)} companies")
    
    # 임원 현황 생성
    executive_sheet = create_executive_sheet(mapped_df, dedupe=dedupe)
    logger.info(f"Created executive sheet: {len(executive_sheet)} executives")
    
    # 엑셀 파일로 저장
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        company_summary.to_excel(writer, sheet_name='회사별 공통행', index=False)
        executive_sheet.to_excel(writer, sheet_name='임원 현황', index=False)
    
    logger.info(f"Saved to {output_xlsx}")
    
    # 요약 출력
    summary = company_summary["구분"].value_counts()
    print(f"\n=== Summary ===")
    print(summary)
    print(f"\nTotal companies: {len(company_summary)}")
    print(f"Total executives: {len(executive_sheet)}")
    print(f"Saved to: {output_xlsx}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: 최종 엑셀 병합 및 출력")
    parser.add_argument("--input-xlsx", required=True, help="Input Excel file (input.xlsx)")
    parser.add_argument("--mapped", required=True, help="Mapped file from Stage2")
    parser.add_argument("--out", required=True, help="Output Excel file (dart_result.xlsx)")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    merge_and_export(
        input_xlsx=args.input_xlsx,
        mapped_file=args.mapped,
        output_xlsx=args.out,
        dedupe=not args.no_dedupe,
    )
    
    logger.info("Stage 3 complete!")


if __name__ == "__main__":
    main()






