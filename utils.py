"""
공통 유틸리티 함수들
"""
import re
import pandas as pd


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


def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """입력 DataFrame의 컬럼명을 정규화하고 필수 컬럼 확인"""
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
    
    return df

