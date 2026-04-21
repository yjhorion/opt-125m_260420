## 증강된 train_v2.jsonl 데이터를 활용하여, Qwen-2.5-3B 모델을 LoRA 방식으로 학습시키는 스크립트.
## FineTuning.

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig

def train():
    # 1. 환경 설정
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    dataset_path = "data/train_v2.jsonl"
    output_dir = "./qwen_web_helper_model"

    print(f"🚀 모델 로딩 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 모델 로드 (M4 최적화: dtype으로 변경 완료)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )

    # 3. 수정된 데이터 포맷팅 함수 (한 줄씩 문자열로 반환)
    def formatting_prompts_func(example):
        # 리스트가 아닌 단일 샘플(dict)이 들어오므로 바로 문자열을 생성하여 반환합니다.
        text = (
            f"<|im_start|>system\nYou are a web automation assistant that outputs JSON coordinates.<|im_end|>\n"
            f"<|im_start|>user\nTask: {example['instruction']}\nElement: {example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
        return text

    # 4. LoRA 설정
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 5. 안정적인 TrainingArguments 사용
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        learning_rate=1e-4,
        num_train_epochs=3,
        bf16=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False
    )

    # 6. 트레이너 초기화
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func, # 수정된 함수 사용
        args=training_args,
        # max_seq_length를 인자에서 제거하고 라이브러리 기본값에 맡깁니다.
    )

    print("🔥 모든 구조가 교정되었습니다. M4 GPU 가속을 시작합니다!")
    trainer.train()

    # 7. 저장
    trainer.save_model(output_dir)
    print(f"✅ 학습 완료! 모델이 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    train()