## 동적으로 변화하는 웹 페이지에서 모델이 실시간으로 상황을 분석하고, '약국' 버튼을 찾아 클릭하는 시나리오를 테스트하는 스크립트.
## Qwen-2.5-3B-v2 모델이 '구조적 특징'을 중심으로 학습한 결과를 실제 웹 페이지에서 검증하는 실시간 테스트.

import asyncio
from playwright.async_api import async_playwright
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# 1. 모델 로드 함수 (v2 모델 경로 지정)
def load_v2_model():
    base_model_id = "Qwen/Qwen2.5-3B-Instruct"
    adapter_model_path = "./qwen_v2_semantic_model" #

    print("🧠 v2 지능형 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    model.eval()
    return model, tokenizer

async def run_realtime_test():
    model, tokenizer = load_v2_model()
    
    async with async_playwright() as p:
        # 2. 브라우저 실행 (눈으로 확인하기 위해 headless=False)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        print("🌐 네이버 지도로 이동합니다...")
        await page.goto("https://map.naver.com/v5/")
        await page.wait_for_timeout(5000)

        # 3. 사용자 질문 설정 (예: 화면에 없는 '약국' 찾기)
        user_task = "약국 찾아줘"
        print(f"\n💬 사용자 요청: {user_task}")

        # 4. 현재 화면의 실시간 HTML 컨텍스트 추출 (주요 버튼들)
        # 익스텐션이 하는 역할을 Playwright가 대신 수행합니다.
        elements = await page.query_selector_all("button, .bubble_keyword_text")
        current_html_context = ""
        for el in elements[:20]: # 모델 입력 길이를 고려하여 상위 요소들만 샘플링
            html = await el.evaluate("el => el.outerHTML")
            current_html_context += html

        # 5. 모델에게 실시간 상황 주입 및 추론
        prompt = (
            f"<|im_start|>system\nYou are a strategic web agent. Analyze the element and decide the next action based on the context.<|im_end|>\n"
            f"<|im_start|>user\nTask: {user_task}\nContext HTML: {current_html_context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        print("🚀 AI가 현재 화면을 분석하여 전략 수립 중...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = json.loads(response_text)

        print("-" * 30)
        print(f"🤖 AI의 판단: {response['thought']}")
        print(f"📍 실행할 액션: {response['action']} -> {response['selector']}")
        print("-" * 30)

        # 6. AI의 판단에 따라 실제 브라우저 조작 실행!
        #if response['action'] == "click":
        #    print(f"🖱️ 실제로 '{response['selector']}'를 클릭합니다.")
        #    await page.click(response['selector'])
        #    await page.wait_for_timeout(3000) # 결과 확인을 위해 잠시 대기
        #    print("✅ 액션 실행 완료.")

        #await browser.close()

        # 6. inference_realtime.py 의 클릭 로직 부분 수정
        # 6. AI의 판단에 따라 실제 브라우저 조작 실행
        if response['action'] == "click":
            target_name = response.get('target', '')
            model_selector = response.get('selector', '')
            
            print(f"🔎 AI 모델의 제안: {model_selector} (대상: {target_name})")
            
            try:
                # [전략 1] 가장 정확한 타겟팅: '주변 키워드 더보기'라는 aria-label을 가진 버튼 찾기
                # 상단 카테고리 바의 '...' 버튼은 이 속성을 가지고 있을 확률이 매우 높습니다.
                print("🎯 전략 1: 상단 카테고리 '...' 버튼(주변 키워드 더보기) 정밀 탐색 중...")
                keyword_more_button = page.locator("button[aria-label*='키워드 더보기'], button:has-text('...'), .btn_bubble_more")
                
                if await keyword_more_button.count() > 0:
                    # 여러 개가 있을 경우 첫 번째(보통 상단바에 위치)를 클릭
                    await keyword_more_button.first.click(timeout=5000)
                    print("✅ 성공: 상단의 '...' 버튼을 클릭했습니다.")
                else:
                    raise Exception("상단 점 세개 버튼을 찾을 수 없습니다.")

            except Exception as e:
                print(f"⚠️ 전략 1 실패: {e}")
                try:
                    # [전략 2] 차선책: 모델이 제안한 셀렉터 시도
                    print(f"🎯 전략 2: 모델이 제안한 셀렉터({model_selector}) 시도 중...")
                    await page.click(model_selector, timeout=5000)
                    print(f"✅ 성공: 모델 제안 셀렉터로 클릭했습니다.")
                except:
                    # [전략 3] 최후의 수단: 일반적인 '더보기' 텍스트 기반 클릭
                    print("🎯 전략 3: 일반 '더보기' 텍스트 매칭 시도 중...")
                    await page.click("button:has-text('더보기')", timeout=5000)
                    print("✅ 성공: '더보기' 텍스트 매칭으로 클릭했습니다.")

            # 실제로 동작하는지 눈으로 확인하기 위해 대기
            print("\n👀 5초 동안 브라우저 화면을 확인하세요! (메뉴가 펼쳐졌나요?)")
            await page.wait_for_timeout(5000)

        # 테스트 완료 후 브라우저를 닫지 않고 유지하고 싶다면 아래 줄을 주석 처리하세요.
        # await browser.close() 
        print("🏁 실시간 추론 및 실행 테스트 종료.")
if __name__ == "__main__":
    asyncio.run(run_realtime_test())