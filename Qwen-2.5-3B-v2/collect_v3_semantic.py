## 좌표는 보조 용도정도로만 활용하고, '구조' 중심으로 학습하기 위한 raw-data 수집 스크립트.
## Qwen-2.5-3B의 collect.py를 기반으로, 요소의 '구조적 특징'을 중심으로 데이터를 수집하도록 수정했습니다.
## 이렇게 수집된 데이터는 모델이 '좌표'보다는 '구조적 특징'을 학습하는 데 도움을 줄 것입니다.   

import asyncio
from playwright.async_api import async_playwright
import json
import os

async def collect_semantic_data():
    os.makedirs("data", exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        # 다양한 해상도에서도 '구조'는 동일함을 학습하기 위해 표준 해상도 사용
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        await page.goto("https://map.naver.com/v5/")
        await page.wait_for_timeout(5000)

        async def get_element_info(el):
            # 익스텐션이 실시간으로 요소를 찾을 때 사용할 핵심 정보들
            return await el.evaluate("""el => {
                return {
                    "tag": el.tagName.toLowerCase(),
                    "id": el.id,
                    "class": el.className,
                    "text": el.innerText.trim().replace(/\\n/g, ' '),
                    "placeholder": el.placeholder || "",
                    "aria_label": el.getAttribute('aria-label') || "",
                    "role": el.getAttribute('role') || ""
                }
            }""")

        raw_data = []
        
        # 1. 초기 화면의 주요 조작 요소 수집
        targets = await page.query_selector_all("button, input, a")
        for el in targets:
            info = await get_element_info(el)
            if info["text"] or info["placeholder"] or info["aria_label"]:
                raw_data.append({
                    "context": "main_screen",
                    "info": info,
                    "html": (await el.evaluate("el => el.outerHTML"))[:300]
                })

        # 2. '더보기' 클릭 후 나타나는 요소 수집 (추론 루프용)
        try:
            await page.click(".btn_bubble_more")
            await page.wait_for_timeout(2000)
            more_targets = await page.query_selector_all(".bubble_keyword_text, button")
            for el in more_targets:
                info = await get_element_info(el)
                if info["text"]:
                    raw_data.append({
                        "context": "more_menu_layer",
                        "info": info,
                        "html": (await el.evaluate("el => el.outerHTML"))[:300]
                    })
        except: pass

        with open("data/raw_v3_semantic.json", "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 구조 중심 데이터 {len(raw_data)}개 수 Bristol 수집 완료!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(collect_semantic_data())