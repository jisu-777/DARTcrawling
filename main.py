from dart_fetcher import build_dart_result

def main():
    output_path = build_dart_result(
        input_path="input.xlsx",
        output_path="dart_result.xlsx",
        years_back=2,  # 5년치 데이터 검색 (필요시 조정)
        debug=True,    # 디버그 정보 출력 (False로 설정하면 출력 안함)
    )

if __name__ == "__main__":
    main()
