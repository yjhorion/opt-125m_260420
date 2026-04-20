## 네이버 길찾기 화면에서 '강남역 맛집'까지 직접 input검색창에 입력하는 자동화 (브라우저 자동검색)

import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        # 1. 브라우저 실행 (headless=False로 설정해야 우리 눈에 보입니다)
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        print("네이버 지도로 이동 중...")
        await page.goto("https://map.naver.com/v5/")
        
        # 페이지 로딩을 위해 잠시 대기 (네이버 지도는 무겁습니다)
        await page.wait_for_timeout(3000)
        
        # 2. 검색창 요소 찾기
        # 네이버 지도 검색창의 클래스나 ID는 바뀔 수 있지만, 현재 기준 input 태그를 찾습니다.
        search_input = await page.query_selector("input.input_search")
        
        if search_input:
            # 3. 해당 요소의 HTML 정보(Outer HTML) 가져오기
            html_content = await search_input.evaluate("el => el.outerHTML")
            print("\n[수집된 데이터 - 검색창 HTML]")
            print(html_content)
            
            # 4. 실제로 글자 입력해보기 (동작 확인)
            await search_input.fill("강남역 맛집")
            print("\n✅ 검색창에 '강남역 맛집'을 입력했습니다.")
        else:
            print("❌ 검색창을 찾지 못했습니다. 페이지 구조를 확인해야 합니다.")
            
        # 눈으로 확인하기 위해 5초간 대기 후 종료
        await page.wait_for_timeout(5000)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())