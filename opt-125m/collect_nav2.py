## 네이버 길찾기 화면을 긁어와, 구성요소들을 가져오고 이를 raw_dataset.json파일로 저장하도록 하는 코드
# 네이버의 화면은 복잡한 구조를 가지고있어, 모든 요소를 가져오진 못하는 한계를 보임

import asyncio
from playwright.async_api import async_playwright
import json

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto("https://map.naver.com/v5/")
        await page.wait_for_timeout(3000)
        
        # 1. 상호작용 가능한 주요 요소들(버튼, 입력창)을 모두 찾습니다.
        # 가이드: 네이버 지도의 주요 메뉴들을 타겟팅합니다.
        targets = await page.query_selector_all("button, input, a.link_navbar")
        
        collected_data = []
        
        print(f"\n🔍 총 {len(targets)}개의 요소를 발견했습니다. 데이터를 추출합니다...")

        for index, el in enumerate(targets[:20]): # 너무 많을 수 있으니 상위 20개만
            # 요소의 속성들 추출
            html = await el.evaluate("el => el.outerHTML")
            text = await el.inner_text()
            
            # AI 학습용 데이터 구조 만들기
            data_point = {
                "element_id": index,
                "text": text.strip(),
                "html_snippet": html[:200] # 너무 길면 보기 힘드니 자름
            }
            collected_data.append(data_point)
            print(f"[{index}] 추출 완료: {text.strip()[:10]}...")

        # 2. 결과를 JSON 파일로 저장 (이게 우리의 첫 '데이터셋'입니다)
        with open("raw_dataset.json", "w", encoding="utf-8") as f:
            json.dump(collected_data, f, ensure_ascii=False, indent=2)
            
        print("\n✅ 'raw_dataset.json' 파일에 데이터가 저장되었습니다.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())