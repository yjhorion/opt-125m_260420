## Qwen-2.5-3B의 raw-data 수집 스크립트.

import asyncio
from playwright.async_api import async_playwright
import json
import os

async def collect_rich_data():
    # 데이터 폴더 생성
    os.makedirs("data", exist_ok=True)

    async with async_playwright() as p:
        # M4 성능을 믿고 브라우저를 띄워 확인하며 수집합니다.
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        print("🌐 네이버 지도로 접속 중...")
        await page.goto("https://map.naver.com/v5/")
        await page.wait_for_timeout(5000)

        # [핵심] '음식점' 카테고리를 클릭하여 검색 결과 리스트를 호출
        try:
            # 텍스트로 버튼을 찾거나 특정 클래스를 타겟팅합니다.
            await page.click("text=음식점")
            print("✅ 음식점 카테고리 클릭 완료. 리스트 로딩 대기...")
            await page.wait_for_timeout(4000)
        except Exception as e:
            print(f"⚠️ 카테고리 클릭 실패(이미 열려있을 수 있음): {e}")

        # 수집 대상: 버튼, 링크, 입력창, 그리고 검색 결과 아이템들
        selectors = "button, input, a, .search_item_cell, .link_search"
        elements = await page.query_selector_all(selectors)
        
        raw_data = []
        for index, el in enumerate(elements):
            text = (await el.inner_text()).strip().replace("\n", " ")
            rect = await el.bounding_box() # 비전 우회를 위한 좌표 데이터
            html = await el.evaluate("el => el.outerHTML")
            
            # 유효한 텍스트가 있거나 input 창인 경우만 수집
            if rect and (text or "input" in html):
                raw_data.append({
                    "element_id": index,
                    "text": text,
                    "rect": rect, # {x, y, width, height}
                    "html_snippet": html[:500] # Qwen은 문맥 파악이 좋으므로 정보를 조금 더 담음
                })

        with open("data/raw_v2.json", "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
            
        print(f"📊 수집 완료: 총 {len(raw_data)}개의 데이터가 'data/raw_v2.json'에 저장되었습니다.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(collect_rich_data())