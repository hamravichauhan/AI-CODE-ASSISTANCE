import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Fix: disable torch.compile (Inductor/Triton) on Windows
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# ✅ Fix fragmentation (helps prevent random OOM spikes)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Import unsloth FIRST
import unsloth
from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


def main():
    # ==========================================
    # 1. CONFIGURATION (RTX 4060 8GB)
    # ==========================================
    max_seq_length = 1024   # ✅ reduced from 2048 to prevent OOM
    dtype = None
    load_in_4bit = True

    # ==========================================
    # 2. LOAD MODEL
    # ==========================================
    print("⏳ Loading Qwen 2.5 Coder 1.5B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # ✅ reduced from 16 to save VRAM
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ==========================================
    # 3. LOAD DATA
    # ==========================================
    if not os.path.exists("MEGA_TRAIN.jsonl"):
        raise FileNotFoundError("MEGA_TRAIN.jsonl not found! Please verify the file name.")

    print("📂 Loading Dataset...")
    dataset = load_dataset("json", data_files="MEGA_TRAIN.jsonl", split="train")

    # Convert conversations -> single text
    def to_text(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True, num_proc=1)

    # ✅ OPTIONAL: filter super-long samples to avoid sudden VRAM spikes
    # You can adjust 6000 based on your dataset length
    def length_filter(example):
        return len(example["text"]) < 6000

    dataset = dataset.filter(length_filter)

    # Pre-tokenize (so TRL doesn't need dataset_text_field)
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names
    )

    # ==========================================
    # 4. TRAIN
    # ==========================================
    print("🚀 Starting Training on RTX 4060...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=20,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_steps=200,
            save_total_limit=2,
            report_to="none",
        ),
    )

    trainer.train()

    # ==========================================
    # 5. SAVE
    # ==========================================
    print("✅ Training Complete!")
    print("💾 Saving Model to 'my_local_js_model'...")
    model.save_pretrained("my_local_js_model")
    tokenizer.save_pretrained("my_local_js_model")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
