## Qwen-2.5-3B-v2 모델을 활용하여, 메인 화면에 없는 '약국' 요청에 대한 모델의 판단과 행동 계획을 테스트하는 스크립트입니다.
## Qwen-2.5-3B-v2의 generate_v3.py에서 생성된 학습 데이터를 활용하여, 모델이 '구조적 특징'을 중심으로 학습한 결과를 검증하는 시나리오.

## 정적인 이미 가지고 있는 HTML 데이터를 기반으로 답변을 줌. (동적인 테스트는 inference_realtime.py 에서 진행)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def run_v2_inference():
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    adapter_model_path = "./qwen_v2_semantic_model"

    print("🧠 v2 지능형 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model.eval()

    # 시나리오: 메인 화면에 없는 '약국' 요청
    test_instruction = "약국 찾아줘"
    # 입력 HTML은 메인 화면의 컨텍스트임을 알려줍니다.
    test_html = '<div class="main_screen_context">네이버 지도 메인 화면</div>' 

    prompt = (
        f"<|im_start|>system\nYou are a strategic web agent. Analyze the element and decide the next action based on the context.<|im_end|>\n"
        f"<|im_start|>user\nTask: {test_instruction}\nContext HTML: {test_html}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    print(f"\n💬 질문: {test_instruction}")
    print("\n🚀 AI의 전략적 판단 중...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.1,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print("-" * 30)
    print(f"🤖 모델 응답:\n{response}")
    print("-" * 30)

if __name__ == "__main__":
    run_v2_inference()