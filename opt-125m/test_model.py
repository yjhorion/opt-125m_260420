## 모델이 학습한 내용을 기반으로 테스트를 진행하는 코드

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def ask_ai(instruction, input_html):
    model_id = "facebook/opt-125m"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 우리가 학습시킨 '뇌'를 불러와서 합칩니다.
    model = PeftModel.from_pretrained(base_model, "./m4_web_helper_model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # AI가 배운 양식 그대로 질문을 구성합니다.
    prompt = f"### Instruction: {instruction}\n### Input: {input_html}\n### Response:"
    #AI가 학습한 그대로의 양식을 프롬프트로 제공 
    
    # 수정 후 (AI에게 힌트를 더 줌)
    #prompt = f"Below is an HTML element and a user request. Predict the action (click or type).\nRequest: {instruction}\nElement: {input_html}\nAction:"


    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# 테스트 실행
test_q = "잠실역 가는 길 알려줘"
test_html = '<button class="btn_navbar">길찾기</button>'

print("--- AI 지능 테스트 ---")
print(f"질문: {test_q}")
print(f"결과: {ask_ai(test_q, test_html)}")