import spacy

# EntityExtractor 클래스: 자연어 텍스트에서 개체(Entity)를 추출하고 커스텀 개체 인식 모델을 훈련하는 클래스
# NLP를 통해 텍스트에서 인명, 지명, 날짜, 시간 등의 특정 개체를 식별하고 분류
class EntityExtractor:
    def __init__(self, lang='ko_core_news_sm'):
        try:
            # spaCy 언어 모델 로드: 지정된 언어 모델을 메모리에 로드
            # 텍스트 분석 및 개체 인식을 위한 사전 훈련된 모델 사용
            self.nlp = spacy.load(lang)
        
        # 지정된 언어 모델이 설치되지 않은 경우
        # 대안으로 영어 모델 사용
        except OSError:
            self.nlp = spacy.load('en_core_web_sm')
        
    # 개체 추출 함수: 입력된 텍스트에서 명명된 개체(Named Entity)를 추출
    # 매개변수 text: 분석할 텍스트 문자열
    # 반환값: 추출된 개체들의 정보를 담은 딕셔너리 리스트
    # 텍스트에서 사람명, 지명, 조직명, 날짜 등을 자동으로 식별
    def extract_entities(self, text):
        # 텍스트 처리: spaCy 모델을 사용하여 텍스트를 분석하고 Doc 객체 생성
        # spaCy의 NLP 파이프라인을 통해 토큰화, 품사 태깅, 개체 인식 등을 수행
        doc = self.nlp(text)

        entities = []

        # doc.ents : spacy가 인식한 명명딘 개체들의 컬렉션
        for ent in doc.ents:
            entities.append({
                'text': ent.text,          # 개체의 실제 텍스트
                'label': ent.label_,       # 개체의 유형 라벨 (PERSON, ORG, DATE 등)
                'start': ent.start_char,   # 원본 텍스트에서 개체의 시작 위치
                'end': ent.end_char,       # 원본 텍스트에서 개체의 끝 위치
                'description': spacy.explain(ent.label_)  # 라벨에 대한 설명
            })
            # 개체 정보 저장: 각 개체의 상세 정보를 딕셔너리로 구성하여 리스트에 추가
            # 개체의 텍스트, 유형, 위치 정보를 구조화하여 후처리나 분석에 활용

        return entities

# 사용 예시

# EntityExtractor 객체 생성: 개체 추출기 인스턴스 생성
# 텍스트에서 개체를 추출하기 위한 도구 준비
extractor = EntityExtractor()

# 텍스트에서 개체 추출
text = "김철수는 내일 오후 3시에 서울역에서 만나자고 했다"
# 분석할 샘플 텍스트: 인명, 시간, 장소 정보가 포함된 한국어 문장
# 개체 추출 기능 테스트를 위한 예시 데이터

# 개체 추출 실행: 샘플 텍스트에서 명명된 개체들을 추출
entities = extractor.extract_entities(text)

for entity in entities:
    # 추출된 개체 출력: 각 개체의 정보를 콘솔에 표시
    print(f"개체: {entity['text']}, 유형: {entity['label']}, 설명: {entity['description']}")
    # 출력 형식: 개체 텍스트, 유형 라벨, 유형 설명을 사용자 친화적으로 표시
    # 개체 추출 결과를 시각적으로 확인
