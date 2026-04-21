import asyncio
import json
import torch
import sys # 프로세스 유지를 위해 추가
from playwright.async_api import async_playwright
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_v2_model():
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
    return model, tokenizer

async def run_intelligent_loop(user_task):
    model, tokenizer = load_v2_model()
    
    # 1. Playwright 세션 시작 (with 문 없이 수동 관리)
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context(viewport={'width': 1280, 'height': 800})
    page = await context.new_page()
    
    print("🌐 네이버 지도로 이동합니다...")
    await page.goto("https://map.naver.com/v5/")
    await page.wait_for_timeout(5000)

    max_attempts = 3
    current_attempt = 0
    success = False

    while current_attempt < max_attempts and not success:
        current_attempt += 1
        print(f"\n🔄 [시도 {current_attempt}] 화면 분석 중...")

        visible_elements = await page.query_selector_all("button:visible, .bubble_keyword_text:visible, a:visible")
        visible_texts = [await el.inner_text() for el in visible_elements]
        
        all_elements = await page.query_selector_all("button, .bubble_keyword_text, a")
        html_context = ""
        for el in all_elements[:50]:
            html_context += await el.evaluate("el => el.outerHTML")

        prompt = (
            f"<|im_start|>system\nYou are a strategic web agent. Analyze the element and decide the next action.<|im_end|>\n"
            f"<|im_start|>user\nTask: {user_task}\nContext HTML: {html_context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=250, temperature=0.1)
        
        res_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            response = json.loads(res_text)
        except:
            print("⚠️ JSON 파싱 실패.")
            break
            
        print(f"🤖 AI 판단: {response['thought']}")
        
        target_text = response.get('target', '')
        is_actually_visible = any(target_text in vt for vt in visible_texts) if target_text else False

        if is_actually_visible:
            print(f"✨ 목표 발견! '{target_text}'를 직접 클릭합니다.")
            await page.click(f"text='{target_text}'")
            success = True
        elif current_attempt < max_attempts:
            print("🎯 메뉴 확장을 위해 '...' 버튼을 클릭합니다.")
            await page.click("button[aria-label*='키워드 더보기'], .btn_bubble_more")
            await page.wait_for_timeout(2000)
        else:
            # 2. 검색어 정제: user_task에서 "찾아줘" 등을 제외하고 target_text(피부과)만 추출
            search_query = target_text if target_text and target_text != "더보기 버튼" else user_task.replace("찾아줘", "").replace("클릭해", "").strip()
            
            print(f"\n🔍 [최종 전략] 검색창에 '{search_query}'를 입력합니다.")
            search_input = page.locator("input.input_search, #search-input").first
            
            if await search_input.count() > 0:
                await search_input.fill(search_query)
                await page.keyboard.press("Enter")
                print(f"✅ '{search_query}' 검색 완료.")
                success = True

    print("\n🏁 모든 프로세스가 완료되었습니다.")
    print("📢 브라우저 유지를 위해 스크립트 대기 중... (Ctrl+C를 누르면 종료됩니다)")
    
    # 3. 프로세스가 종료되지 않도록 무한 대기 (브라우저 유지 핵심)
    # inference_loop.py의 마지막 대기 부분 수정
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n👋 사용자 요청에 의해 브라우저를 닫습니다.")
        # 브라우저 종료 시 발생하는 통신 에러를 무시합니다.
        try:
            if browser:
                await browser.close()
            if p:
                await p.stop()
        except Exception:
            # 드라이버가 이미 종료된 경우 발생하는 에러를 숨깁니다.
            pass

if __name__ == "__main__":
    try:
        asyncio.run(run_intelligent_loop("피부과 찾아줘"))
    except KeyboardInterrupt:
        pass