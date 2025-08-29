import os
import csv
import json
import logging
import random
import datetime
import string
import io
import zipfile

# 이 파일은 다양한 파일 형식을 쓰고/읽는 콘솔 기반 파일 관리 프로그램입니다.
# 핵심 특징
# - 파일 시그니처(FILE_SIGNATURES) 기반의 바이너리 형식 식별 (jpeg, png, gif, bmp, zip, pdf)
# - 텍스트 형식(json, csv, txt)의 단계적 파싱 (JSON → CSV → 일반 텍스트)
# - 지원 확장자 화이트리스트를 통한 안전성 확보
# - 프로그램 실행 디렉터리와 스크립트 디렉터리가 달라도 동작하도록 경로 보정(resolve_path)
# - 내부 처리는 절대 경로 사용, 로그에는 스크립트 기준 상대 경로 표시

# logger 객체 생성
logger = logging.getLogger(__name__)
# 현재 모듈(__name__) 이름을 로거의 이름으로 사용합니다. 이렇게 하면 여러 모듈의 로그를 구분할 수 있습니다.

# 로그 레벨 설정
logger.setLevel(logging.INFO)
# 로거가 처리할 최소 로그 레벨을 INFO로 설정합니다.
# 이 설정보다 낮은 레벨(DEBUG)의 로그는 무시됩니다.
# 레벨 순서: DEBUG < INFO < WARNING < ERROR < CRITICAL

# 파일 핸들러 생성 및 설정
file_handler = logging.FileHandler('app.log', encoding='utf-8')
# 로그 메시지를 'app.log' 파일에 기록하는 핸들러를 만듭니다.
# encoding='utf-8'은 한글(유니코드)이 포함된 로그 메시지가 깨지지 않도록 합니다.
file_handler.setLevel(logging.INFO)
# 파일 핸들러가 처리할 최소 로그 레벨을 INFO로 설정합니다.

# 스트림 핸들러 생성 및 설정
stream_handler = logging.StreamHandler()
# 로그 메시지를 콘솔(터미널)에 출력하는 핸들러를 만듭니다.
stream_handler.setLevel(logging.INFO)
# 스트림 핸들러가 처리할 최소 로그 레벨을 INFO로 설정합니다.

# 포맷터 생성 및 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# 로그 메시지의 형식을 지정하는 포맷터를 만듭니다.
# %(asctime)s: 로그가 기록된 시간 (예: 2025-08-29 15:31:41,703)
# %(levelname)s: 로그 레벨 (예: INFO, ERROR)
# %(message)s: 실제 로그 메시지
file_handler.setFormatter(formatter)
# 파일 핸들러에 위에서 만든 포맷터를 적용합니다.
stream_handler.setFormatter(formatter)
# 스트림 핸들러에도 포맷터를 적용합니다.

# 핸들러를 로거에 추가
logger.addHandler(file_handler)
# 파일에 로그를 기록하는 기능을 로거에 연결합니다.
logger.addHandler(stream_handler)
# 콘솔에 로그를 출력하는 기능을 로거에 연결합니다.

# --- 파일 시그니처 및 지원 확장자 사전 ---
# 바이너리 파일의 매직 넘버(헤더)와 형식 매핑 테이블
FILE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'BM': 'bmp',
    b'PK\x03\x04': 'zip',
    b'%PDF': 'pdf',
}

# 지원하는 모든 확장자 목록
# 처리 대상 확장자 화이트리스트 (이외 확장자는 거부)
SUPPORTED_EXTENSIONS = [
    '.jpeg', '.png', '.gif', '.bmp', '.zip', '.pdf',
    '.json', '.csv', '.txt'
]

# --- 경로 유틸 ---
def resolve_path(user_input_path):
    """사용자 입력 경로를 스크립트 기준 절대 경로로 변환합니다.

    동작 배경:
    - 콘솔에서 프로그램을 임의 디렉터리에서 실행하면 상대 경로 해석 기준이 달라질 수 있습니다.
    - 이를 방지하기 위해, 입력이 절대 경로가 아닌 경우 스크립트 파일이 있는 디렉터리를 기준으로 합성합니다.

    Args:
        user_input_path: 사용자가 입력한 파일 경로(상대/절대 가능)

    Returns:
        절대 경로 문자열
    """
    if os.path.isabs(user_input_path):
        return user_input_path
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, user_input_path)

# --- 파일 쓰기 함수 (이전과 동일) ---
def write_data(filename, data):
    """파일 확장자에 따라 데이터를 적절한 방식으로 기록합니다.

    처리 원칙:
    - 지원 확장자만 허용(SUPPORTED_EXTENSIONS)
    - 기존 파일 존재 여부에 따라 쓰기 모드 자동 결정('w' 또는 'a')
    - json: 기존 리스트/단일 객체를 일관된 리스트 구조로 유지하여 덮어쓰기
    - csv: csv.writer 사용; 'a' 모드시 행/행렬 형태 모두 대응
    - 바이너리(jpeg/png/gif/bmp/pdf): 시그니처 + 페이로드를 바이너리 모드로 기록
    - txt: UTF-8 텍스트로 기록

    로그 표기:
    - 내부는 절대 경로로 처리하되, 로그에는 스크립트 기준 상대 경로를 출력합니다.

    Args:
        filename: 절대 경로 파일명 (resolve_path를 통해 사전에 정규화)
        data: 기록할 데이터(형식별 요구 사항 상이)
    """
    try:
        # 1. 파일 확장자 추출 및 유효성 검사
        display_name = os.path.relpath(filename, os.path.dirname(__file__))
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 지원하지 않는 파일 형식인 경우 함수를 즉시 종료
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error(f"'{ext}' 형식은 지원하지 않습니다. 지원 형식: {SUPPORTED_EXTENSIONS}")
            return
        
        # 2. 파일 쓰기 모드 자동 선택
        # 파일이 존재하면 이어쓰기('a'), 없으면 새로쓰기('w') 모드 선택
        mode = 'a' if os.path.exists(filename) else 'w'

        # 3. 확장자에 따른 데이터 쓰기 로직 분기
        # .csv 및 .zip 파일 처리 (CSV는 텍스트로, ZIP은 바이너리 헤더로 처리)
        if ext in ('.csv',):
            # CSV는 텍스트 기반이지만 writer를 사용해 안전하게 기록
            with open(filename, mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if mode == 'a':
                    # 2차원 리스트면 여러 행 추가, 1차원 리스트면 단일 행 추가
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                        writer.writerows(data)
                    elif isinstance(data, list):
                        writer.writerow(data)
                else:
                    # 새 파일에는 전체 데이터를 한 번에 기록
                    writer.writerows(data)
            logger.info(f"'{display_name}' 파일에 CSV 데이터 쓰기 완료 (모드: {mode}).")
        
        # .json 파일 처리 (가장 복잡한 로직)
        elif ext == '.json':
            # JSON은 항상 리스트 형태로 보존하여 Append 시 일관성 유지
            current_data = []
            # 추가 모드일 때만 기존 데이터를 불러옴
            if mode == 'a':
                with open(filename, 'r', encoding='utf-8') as f:
                    try:
                        # 기존 JSON 데이터를 파이썬 객체로 로드
                        current_data = json.load(f)
                        if not isinstance(current_data, list):
                            # 기존 데이터가 리스트가 아니면, 리스트로 변환하여 이어쓰기 준비
                            current_data = [current_data]
                    except (json.JSONDecodeError, FileNotFoundError):
                        # 파일이 깨졌거나 비어있으면 빈 리스트로 초기화
                        logger.error(f"'{display_name}' 파일이 유효한 JSON 형식이 아니거나 비어있습니다. 새 파일로 생성합니다.")
                        current_data = []
            
            # 새 데이터를 현재 데이터 리스트에 추가 (모드가 'a'든 'w'든 동일)
            if isinstance(data, dict):
                # 새 객체를 리스트에 추가
                current_data.append(data)
            else:
                logger.warning("JSON 데이터는 딕셔너리 형태여야 합니다. 무시됩니다.")
                
            # 전체 데이터를 파일에 다시 씀 (모드가 'a'여도 파일은 덮어쓰기)
            with open(filename, 'w', encoding='utf-8') as f:
                # indent=4로 사람이 읽기 좋은 형태로 저장
                json.dump(current_data, f, indent=4, ensure_ascii=False)
            logger.info(f"'{display_name}' 파일에 JSON 데이터 쓰기 완료 (총 {len(current_data)}개).")
        
        # 바이너리 파일 처리 (.jpeg, .png, .gif, .bmp, .pdf)
        elif ext in ('.jpeg', '.png', '.gif', '.bmp', '.pdf'):
            # 확장자 문자열에서 시그니처 바이트를 역매핑하여 헤더를 구성
            signature_map = {v: k for k, v in FILE_SIGNATURES.items()}
            file_signature = signature_map.get(ext.replace('.', ''), b'')
            
            # 헤더 + 페이로드 결합 (실제 환경에선 데이터가 이미 올바른 바이너리일 수 있음)
            data = file_signature + data

            with open(filename, 'ab' if mode == 'a' else 'wb') as f:
                f.write(data)
            logger.info(f"'{display_name}' 파일에 바이너리 데이터 쓰기 완료.")

        elif ext == '.zip':
            # ZIP은 유효한 ZIP 바이너리(bytes)를 그대로 기록해야 하며, 임의 헤더를 덧붙이면 안 됨
            with open(filename, 'ab' if mode == 'a' else 'wb') as f:
                f.write(data)
            logger.info(f"'{display_name}' 파일에 ZIP 데이터 쓰기 완료.")
        
        else:
            # 일반 텍스트(.txt)
            with open(filename, 'a' if mode == 'a' else 'w', encoding='utf-8') as f:
                f.write(data)
            logger.info(f"'{display_name}' 파일에 일반 텍스트 데이터 쓰기 완료.")
    
    except PermissionError:
        logger.error(f"권한 오류: '{display_name}'에 파일을 쓸 권한이 없습니다.")
    except Exception as e:
        logger.error(f"알 수 없는 오류가 발생했습니다: {e}")

# --- 파일 읽기 함수 (최종 수정) ---
def read_data_by_signature(filename):
    """파일을 단계적으로 검사하여 형식을 식별하고 데이터를 반환합니다.

    판별 순서:
    1) 바이너리 시그니처 검사: 헤더 startswith 매칭으로 jpeg/png/gif/bmp/zip/pdf 식별
       - 이미지/PDF는 전체 바이너리 데이터를 반환, ZIP은 문자열 'zip' 반환
    2) 텍스트 파싱:
       - JSON: json.load 성공 시 dict/list 반환
       - CSV: 샘플에서 구분자 존재 확인 → Sniffer 시도 → 최소 한 줄 이상 2컬럼 이상 확인
       - TXT: UTF-8로 전체 읽기 가능하면 'txt' 반환
    3) 그 외: 'unknown'

    로그 표기:
    - 스크립트 기준 상대 경로와 최종 식별 형식을 INFO 레벨로 남깁니다.

    Args:
        filename: 절대 경로 파일명 (resolve_path로 정규화된 값)

    Returns:
        형식별 데이터(jpeg/png/gif/bmp/pdf는 bytes, json은 객체, csv는 list[list[str]], txt/unknown/zip은 문자열)
    """
    try:
        # 1. 파일 확장자 추출 및 유효성 검사
        display_name = os.path.relpath(filename, os.path.dirname(__file__))
        # 사용자가 입력한 경로는 이미 절대 경로로 정규화되어 있음
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # 지원하는 확장자인지 확인하여 지원하지 않으면 오류를 기록하고 함수 종료
        if ext not in SUPPORTED_EXTENSIONS:
            logger.error(f"'{ext}' 형식은 지원하지 않습니다. 지원 형식: {SUPPORTED_EXTENSIONS}")
            return

        # 2. 파일 존재 여부 확인
        # 파일이 존재하지 않으면 FileNotFoundError 예외를 발생
        if not os.path.exists(filename):
            raise FileNotFoundError

        file_type = None  # 최종 판별된 형식 문자열 저장
        data = None       # 최종 반환 데이터 저장 (형식에 따라 타입 다름)

        # 확장자가 .zip인 경우에는 시그니처 유무와 무관하게 CSV 등 텍스트 파싱을 시도하지 않도록 즉시 처리
        if ext == '.zip':
            file_type = 'zip'
            data = 'zip'
            logger.info(f"'{display_name}'을(를) '{file_type}' 형식으로 식별했습니다.")
            return data

        # 1. 파일 헤더만 읽어서 파일 형식 식별
        # 3. 1단계: 바이너리 시그니처 확인 (가장 확실한 방법)
        # 파일을 바이너리 모드('rb')로 열어 파일의 시작 부분(헤더)을 읽습니다.
        with open(filename, 'rb') as f:
            header = f.read(8)  # 충분한 길이의 매직 넘버 비교를 위해 8바이트 읽기
            for signature, sig_type in FILE_SIGNATURES.items():
                if header.startswith(signature):
                    file_type = sig_type
                    break
        
        # 2. 파일 형식이 식별되면 해당 형식에 맞게 전체 데이터 읽기
        # 4. 2단계: 식별된 유형에 따라 데이터 읽기
        # 1단계에서 파일 유형이 식별된 경우, 해당 유형에 맞는 방식으로 데이터를 읽어 반환
        if file_type:
            if file_type in ('jpeg', 'png', 'gif', 'bmp', 'pdf'):
                # 이미지/PDF는 원시 바이트를 그대로 반환
                with open(filename, 'rb') as f:
                    data = f.read()
            elif file_type == 'zip':
                # ZIP은 내용 파싱 대신 형식 정보만 반환
                data = 'zip'
        
        # 3. 바이너리 시그니처가 없으면 텍스트 파싱 시도
        # JSON 파싱
        if not file_type:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # JSON 파싱 시도
                    file_type = 'json'
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # CSV 파싱
        if not file_type:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    sample = f.read(2048)  # 샘플 텍스트로 구분자 존재 여부 확인
                    f.seek(0)              # 파일 포인터 복원
                    # CSV로 판단할 수 있는 구분자가 전혀 없으면 CSV가 아님
                    if not any(d in sample for d in [',', ';', '\t']):
                        raise csv.Error
                    # 가능한 경우 Sniffer로 다이얼렉트 추정
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        f.seek(0)
                        reader = csv.reader(f, dialect)
                    except csv.Error:
                        f.seek(0)
                        reader = csv.reader(f)
                    data = list(reader)  # 전체를 메모리로 읽어 간단 검증
                    # 적어도 한 줄 이상에서 2개 이상의 컬럼이 있어야 CSV로 인정
                    if not any(len(row) > 1 for row in data):
                        raise csv.Error
                    file_type = 'csv'
            except (csv.Error, UnicodeDecodeError):
                pass

        # TXT 파싱
        if not file_type:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    f.read()  # 단순히 읽기 성공 여부로 텍스트 판단
                    file_type = 'txt'
                    data = 'txt'
            except UnicodeDecodeError:
                file_type = 'unknown'
                data = 'unknown'
        
        # 4. 최종 결과 로깅 및 반환
        if file_type:
            logger.info(f"'{display_name}'을(를) '{file_type}' 형식으로 식별했습니다.")
            return data
        else:
            return None

    except FileNotFoundError:
        logger.error(f"파일이 존재하지 않습니다: '{display_name}'")
    except Exception as e:
        logger.error(f"파일 검사 중 오류 발생: {e}")
    
    return None

# --- 메인 메뉴 함수 ---
def main_menu():
    """콘솔 기반 메인 메뉴 UI.

    기능:
    - 1: 파일 쓰기(확장자에 맞는 랜덤 데이터 생성 후 write_data 호출)
    - 2: 파일 읽기(read_data_by_signature 호출)
    - 3: 종료
    """
    while True:
        print("\n" + "="*40)
        print(" 파일 관리 프로그램 ")
        print("="*40)
        print("1. 파일 쓰기")
        print("2. 파일 읽기")
        print("3. 종료")
        print("="*40)
        
        choice = input("원하는 작업을 선택하세요 (1/2/3): ")  # 기본 입력 루프
        
        if choice == '1':
            filename = input("생성/수정할 파일명을 입력하세요 (예: data.csv, user.json): ")
            filename = resolve_path(filename)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if ext == '.csv':
                # CSV: 간단한 랜덤 CSV 데이터 생성 (헤더 포함)
                num_rows = random.randint(2, 5)
                data_to_write = [['이름', '나이']] + [[''.join(random.choices(string.ascii_letters, k=10)), random.randint(1,100)] for _ in range(num_rows)]
            elif ext == '.zip':
                # ZIP: 메모리 상에서 랜덤 파일들(1~3개)을 담아 유효한 ZIP 바이트 생성
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, 'w') as z:
                    for _ in range(random.randint(1, 3)):
                        name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10))) + '.txt'
                        content = 'rand: ' + ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 40)))
                        z.writestr(name, content)
                data_to_write = buf.getvalue()
            elif ext == '.json':
                # 샘플 JSON 객체 생성
                data_to_write = {
                    'timestamp': str(datetime.datetime.now()), 
                    'id': random.randint(100, 999),
                    'random_data': ''.join(random.choices(string.ascii_letters, k=10))
                }
            elif ext in ('.jpeg', '.png', '.gif', '.bmp', '.pdf'):
                # 더미 바이너리 페이로드 생성 (헤더는 write_data에서 결합)
                data_to_write = os.urandom(random.randint(50, 200))
            elif ext == '.txt':
                 # 임의 문자열 한 줄 작성
                 data_to_write = f"랜덤 데이터: {random.randint(1,100)} - " + ''.join(random.choices(string.ascii_letters, k=20)) + "\n"
            else:
                # 지원하지 않는 확장자일 때 None 전달(함수에서 거부 처리)
                data_to_write = None

            write_data(filename, data_to_write)
            
        elif choice == '2':
            filename = input("읽을 파일명을 입력하세요 (예: data.csv): ")
            filename = resolve_path(filename)
            read_data_by_signature(filename)
            
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("잘못된 입력입니다. 1, 2, 3 중 하나를 입력하세요.")

if __name__ == "__main__":
    main_menu()