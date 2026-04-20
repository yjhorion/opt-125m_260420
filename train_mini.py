## 학습용 데이터셋(train_dataset.jsonl)을 기반으로 학습을 진행하는 코드
# loss값이 점진적으로 적어지는것을 확인하기위해, 최초 100개의 소형 데이터셋으로 진행

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig

def train():
    # 1. 데이터셋 로드
    dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")

    # 2. 모델 및 토크나이저 설정
    model_id = "facebook/opt-125m" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 수정된 데이터 포맷팅 함수 (한 줄씩 처리하도록 변경)
    def formatting_prompts_func(example):
        # 한 개의 샘플(dict)이 들어오므로 바로 접근합니다.
        text = f"### Instruction: {example['instruction']}\n### Input: {example['input']}\n### Response: {example['output']}"
        return text

    # 4. LoRA 설정
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. 학습 파라미터
    training_args = TrainingArguments(
        output_dir="./m4_model_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=10,
        report_to="none",
    )

    # 6. 트레이너 실행
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(model_id),
        train_dataset=dataset,
        formatting_func=formatting_prompts_func, 
        peft_config=peft_config,
        args=training_args,
    )

    print(f"🚀 학습 장치: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    print("\n🏁 학습을 시작합니다. M4 GPU가 가동됩니다...")
    
    trainer.train()
    
    # 7. 모델 저장
    trainer.save_model("./m4_web_helper_model")
    print("\n✅ 학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()