import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def main():
    print(">>> Bước 1: Load dataset JSONL...")
    dataset = load_dataset("json", data_files="dataemploy.jsonl", split="train", lines=True)

    print(">>> Bước 2: Tạo prompt/response template...")
    def create_prompt_response(example):
        prompt = f"Số nhân viên tại vị trí {example['position']} ở tháng {example['month']}/{example['year']} là bao nhiêu?"
        response = f"Vào tháng {example['month']}/{example['year']}, số nhân viên ở vị trí {example['position']} là {{so}} người"
        return {"prompt": prompt, "response": response}

    dataset = dataset.map(create_prompt_response)

    print(">>> Bước 3: Load model và tokenizer...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(">>> Bước 4: Format dữ liệu cho chat template...")
    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]},
             {"role": "assistant", "content": example["response"]}],
            tokenize=False
        )}
    dataset = dataset.map(format_chat)

    print(">>> Bước 5: Cấu hình LoRA + Training...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    training_args = TrainingArguments(
        output_dir="./phi3-finetuned",
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=1024,
    )

    trainer.train()
    trainer.model.save_pretrained("./phi3-finetuned")
    tokenizer.save_pretrained("./phi3-finetuned")
    print(">>> Huấn luyện xong, model đã lưu tại ./phi3-finetuned")

if __name__ == "__main__":
    main()
