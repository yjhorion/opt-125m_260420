## Qwen-2.5-3B-v2 모델을 LoRA 방식으로 학습시키는 스크립트.
## Qwen-2.5-3B-v2의 collect_v3_semantic.py에서 생성된 데이터를 활용하여, 모델이 '구조적 특징'을 중심으로 학습할 수 있도록 학습 데이터를 생성합니다. 
## 이 스크립트는 각 요소에 대해 '현재 화면에서 보이는가?' 여부에 따라 다른 사고 과정과 행동 계획을 포함하는 JSON 출력을 생성합니다.

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig

def train_v2():
    # 1. 설정
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    dataset_path = "data/train_v3_semantic.jsonl"
    output_dir = "./qwen_v2_semantic_model"

    print(f"🚀 v2 모델 학습 시작: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 모델 로드 (M4 GPU 최적화)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, 
        device_map="auto"
    )

    # 3. 데이터 포맷팅 (지능형 추론 포맷)
    def formatting_prompts_func(example):
        text = (
            f"<|im_start|>system\nYou are a strategic web agent. Analyze the element and decide the next action based on the context.<|im_end|>\n"
            f"<|im_start|>user\nTask: {example['instruction']}\nContext HTML: {example['input']}<|im_end|>\n"
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

    # 5. 학습 인자
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, 
        learning_rate=1e-4,
        num_train_epochs=5, # 논리 구조 학습을 위해 Epoch를 조금 늘림
        bf16=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="epoch"
    )

    # 6. 트레이너 실행
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    print("🔥 M4 GPU가 새로운 논리 체계를 학습하고 있습니다...")
    trainer.train()

    # 7. 저장
    trainer.save_model(output_dir)
    print(f"✅ v2 모델 학습 완료! 저장 경로: {output_dir}")

if __name__ == "__main__":
    train_v2()