## chain of thought과 행동 계획을 포함하는 지능형 학습 데이터를 생성하는 스크립트.
## Qwen-2.5-3B-v2의 collect_v3_semantic.py에서 수집된 데이터를 활용하여, 모델이 '구조적 특징'을 중심으로 학습할 수 있도록 학습 데이터를 생성합니다. 
## 이 스크립트는 각 요소에 대해 '현재 화면에서 보이는가?' 여부에 따라 다른 사고 과정과 행동 계획을 포함하는 JSON 출력을 생성합니다.

import json

# 데이터 로드
with open("data/raw_v3_semantic.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 메인과 더보기 메뉴 분리
main_elements = [el for el in raw_data if el['context'] == 'main_screen']
more_elements = [el for el in raw_data if el['context'] == 'more_menu_layer']

# 더보기 버튼 식별 (v3 데이터 기반 - aria_label 활용이 가장 정확함)
try:
    more_button = next(el for el in main_elements if "더보기" in el['info']['aria_label'] or "btn_bubble_more" in el['info']['class'])
except StopIteration:
    # 혹시라도 못 찾을 경우를 대비한 기본값
    more_button = main_elements[0] 

train_dataset = []

def create_entry(instruction, target_el, is_hidden=False):
    info = target_el['info']
    
    # 1. 사고 과정(Thought) 정의: 모델에게 논리적 근거를 가르침
    if is_hidden:
        thought = f"현재 화면에 '{info['text']}' 메뉴가 보이지 않습니다. 하지만 '더보기' 버튼을 눌러 숨겨진 메뉴를 확인할 수 있습니다."
        action = {
            "thought": thought,
            "action": "click",
            "target": "더보기 버튼",
            "selector": f".{more_button['info']['class'].split()[0]}", # 첫 번째 클래스명만 사용
            "next_step": f"wait_for_layer_and_click_{info['text']}"
        }
    else:
        thought = f"사용자가 '{info['text']}' 메뉴를 요청했습니다. 현재 화면에 해당 요소가 존재하므로 즉시 클릭합니다."
        action = {
            "thought": thought,
            "action": "click",
            "target": info['text'],
            "selector": f".{info['class'].split()[0]}" if info['class'] else f"tag:{info['tag']}",
            "text_match": info['text']
        }

    return {
        "instruction": f"{info['text']} {instruction}",
        "input": target_el['html'],
        "output": json.dumps(action, ensure_ascii=False)
    }

# 데이터 생성 루프
for el in main_elements:
    if el['info']['text']:
        for inst in ["눌러줘", "클릭해", "보여줘", "어디있어"]:
            train_dataset.append(create_entry(inst, el, is_hidden=False))

for el in more_elements:
    if el['info']['text']:
        for inst in ["찾아줘", "눌러줘", "어디있니", "예약할래"]:
            train_dataset.append(create_entry(inst, el, is_hidden=True))

# 결과 저장
with open("data/train_v3_semantic.jsonl", "w", encoding="utf-8") as f:
    for entry in train_dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 지능형 학습 데이터 {len(train_dataset)}개 생성 완료!")