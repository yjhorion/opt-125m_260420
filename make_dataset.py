## 학습용 dataset을 직접 만드는 dataset생성기
# raw_dataset.py 를 기반으로, 학습용 데이터셋을 생성하여 train_dataset.jsonl에 저장

import json
import random

# 1. 아까 수집한 데이터 로드
with open("raw_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 질문 템플릿 정의 (데이터 증강용)
templates = ["{} 가는 길 알려줘", "{} 어떻게 가?", "{} 경로 검색", "{}까지 얼마나 걸려?", "{} 가는 방법"]
locations = ["석촌호수", "강남역", "우리집", "에버랜드", "서울역"]

train_set = []

# 3. 데이터 생성 로직
for item in raw_data:
    if item["text"] == "길찾기": # 우리가 찾은 보물(Index 1)
        for _ in range(100): # 20개의 변형 문장 생성
            loc = random.choice(locations)
            q = random.choice(templates).format(loc)
            
            # AI 학습 포맷 구축
            train_set.append({
                "instruction": q,
                "input": item["html_snippet"],
                "output": "click"
            })

# 4. JSONL 파일로 저장
with open("train_dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in train_set:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ {len(train_set)}개의 학습 데이터가 생성되었습니다!")