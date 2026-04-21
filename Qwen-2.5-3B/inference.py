import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def run_inference():
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    # 우리가 학습시킨 LoRA 가중치가 저장된 경로
    adapter_model_path = "./qwen_web_helper_model"

    print("🧠 모델 및 어댑터 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # 1. 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. 학습시킨 LoRA 어댑터 결합
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model.eval() # 추론 모드로 전환

    # 3. 테스트할 시나리오 (실제 데이터셋에 있던 것 중 하나)
    # 124번 엘리먼트: 음식점 버튼 (x: 468, y: 25)
    test_instruction = "약국 버튼 눌러줘"
    test_html = '<button class="StyledBubbleKeywordButton-sc-1nm3szv-0 dtCpYL" type="button"><img width="20" height="20" class="StyledImg-sc-t1g0ob-0" src="https://map.pstatic.net/resource/api/v2/image/maps/around-category/dining_category_pc.png?version=103"><span class="bubble_keyword_text">음식점</span></button>'

    # 4. 프롬프트 구성 (학습 때와 동일한 포맷)
    prompt = (
        f"<|im_start|>system\nYou are a web automation assistant that outputs JSON coordinates.<|im_end|>\n"
        f"<|im_start|>user\nTask: {test_instruction}\nElement: {test_html}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("mps") # M4 GPU 가속

    print(f"\n💬 질문: {test_instruction}")
    print(f"🖥️ 대상 HTML: {test_html[:100]}...")
    print("\n🚀 AI 응답 생성 중...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            temperature=0.1, # 일관된 JSON 출력을 위해 낮게 설정
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print("-" * 30)
    print(f"🤖 모델 응답:\n{response}")
    print("-" * 30)

    # JSON 파싱 시도 (정상적인 JSON인지 검증)
    try:
        res_json = json.loads(response)
        print("✅ 성공: 모델이 올바른 JSON 형식을 뱉었습니다.")
        print(f"📍 클릭 타겟 좌표: {res_json['location']['point']}")
    except:
        print("⚠️ 주의: 모델 응답이 순수한 JSON이 아닙니다. 후처리가 필요할 수 있습니다.")

if __name__ == "__main__":
    run_inference()