## 준비된 raw-data를 바탕으로, 좌표 정보가 포함된 학습 데이터를 생성하는 스크립트.

import json
import random

# 데이터 경로 설정
RAW_DATA_PATH = "data/raw_v2.json"
TRAIN_DATA_PATH = "data/train_v2.jsonl"

with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = []

# 질문 템플릿
templates = {
    "click": ["{} 눌러줘", "{} 클릭", "{} 메뉴 보여줘", "{} 어디있어?", "{} 선택해"],
    "type": ["{}에 강남역 검색", "{}에 입력해줘", "{}에서 맛집 찾아줘"]
}

for item in raw_data:
    text = item["text"].strip()
    rect = item["rect"]
    html = item["html_snippet"]
    
    # 텍스트가 없으면 '검색창' 등으로 보정
    if not text:
        if "input_search" in html: text = "검색창"
        elif "btn_clear" in html: text = "삭제 버튼"
        else: continue

    # 액션 결정
    action = "type" if "input" in html else "click"
    
    # 30개씩 변형 데이터 생성
    for _ in range(30):
        instruction = random.choice(templates[action]).format(text)
        
        # 정답 포맷: 액션 + 좌표 정보를 JSON으로 출력하게 학습
        # 이것이 나중에 비전 모델과 연동되는 '핵심 고리'가 됩니다.
        response = {
            "action": action,
            "target": text,
            "location": {
                "point": [rect["x"] + rect["width"]/2, rect["y"] + rect["height"]/2], # 중심점
                "bbox": [rect["x"], rect["y"], rect["x"]+rect["width"], rect["y"]+rect["height"]]
            }
        }
        
        dataset.append({
            "instruction": instruction,
            "input": html,
            "output": json.dumps(response, ensure_ascii=False)
        })

with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 좌표 정보가 포함된 {len(dataset)}개의 데이터가 생성되었습니다.")