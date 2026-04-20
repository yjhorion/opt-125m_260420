## M4를 학습가능한상태로 최초 세팅

# 1. 메모리 할당 : M4 GPU 메모리 안에 1000x1000 크기의 거대한 숫자 표 (Tensor)를 생성. -> 총 100만개의 칸
# 2. 데이터 채우기 : 그 100만개의 tensor를 모두 1로 채움
# 3. 병렬 연산 : GPU에게 "이 100만개의 칸에 있는 숫자에 각각 2를 곱하도록" 명령
# 4. 결과확인 : 그중 첫번째 칸 "{y[0][0].item()}" 에 있는 숫자를 확인하니 2.0이 나와있는것.
# 5. 100만개의 연산을 CPU가 하나씩 처리하는것이 아닌 GPU가 처리했음

import torch
import time

def compare_speed():
    size = 50000  # 10,000 x 10,000 = 1억 개의 요소
    
    # 1. CPU 연산 속도 측정
    print("CPU 연산 시작...")
    start_cpu = time.time()
    x_cpu = torch.ones(size, size, device="cpu")
    y_cpu = x_cpu * 2
    end_cpu = time.time()
    cpu_duration = end_cpu - start_cpu
    print(f"📍 CPU 소요 시간: {cpu_duration:.4f}초")

    # 2. M4 GPU(MPS) 연산 속도 측정
    if torch.backends.mps.is_available():
        print("\nM4 GPU(MPS) 연산 시작...")
        mps_device = torch.device("mps")
        
        # GPU는 첫 실행 시 예열(Warm-up)이 필요하므로 한 번 실행
        _ = torch.ones(100, 100, device=mps_device)
        
        start_mps = time.time()
        x_mps = torch.ones(size, size, device=mps_device)
        y_mps = x_mps * 2
        
        # 중요: GPU 연산은 비동기이므로 완료될 때까지 기다려야 정확한 시간 측정 가능
        torch.mps.synchronize() 
        
        end_mps = time.time()
        mps_duration = end_mps - start_mps
        print(f"🚀 M4 GPU 소요 시간: {mps_duration:.4f}초")
        
        print(f"\n🔥 GPU가 CPU보다 약 {cpu_duration / mps_duration:.1f}배 빠릅니다!")

compare_speed()