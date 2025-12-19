import os
import json
import torch
from pathlib import Path  # 新增：用于自动管理路径
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

# ================= 动态路径配置 (核心修改) =================
# 获取当前脚本所在目录 (D:\vlm_assembly\src)
CURRENT_DIR = Path(__file__).resolve().parent
# 获取项目根目录 (D:\vlm_assembly)
PROJECT_ROOT = CURRENT_DIR.parent

# 1. 缓存路径
os.environ["HF_HOME"] = str(PROJECT_ROOT / "cache" / "huggingface")

# 2. 模型与数据路径
# 假设模型还在 cache 里，或者您可以把它移到 models 文件夹
MODEL_ID = r"D:\vlm_assembly\cache\huggingface\hub\Qwen2.5-VL-7B-Instruct"

# 自动定位 JSON 文件
DATA_FILE = PROJECT_ROOT / "data" / "clean_assembly_sft_12_12focused.json"
# 自动定位图片根目录
IMAGE_ROOT = PROJECT_ROOT / "data"

# 3. 输出路径 (每次实验建议改一下版本号，如 v1, v2)
EXPERIMENT_NAME = "v2_focused_simple"
OUTPUT_DIR = PROJECT_ROOT / "output" / EXPERIMENT_NAME

print(f"项目根目录: {PROJECT_ROOT}")
print(f"数据文件: {DATA_FILE}")
print(f"输出目录: {OUTPUT_DIR}")


# ==========================================================

class AssemblyDataset(Dataset):
    def __init__(self, data_path, image_root, processor):
        self.processor = processor
        self.image_root = image_root  # 记住图片根目录
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # === 动态寻找图片路径 ===
        image_rel_path = item.get("image", None)
        if isinstance(image_rel_path, list):
            image_rel_path = image_rel_path[0]

        if not image_rel_path:
            messages = item.get("conversations") or item.get("messages")
            for msg in messages:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    for content in msg['content']:
                        if content.get('type') == 'image':
                            image_rel_path = content.get('image')
                            break
                if image_rel_path: break

        # === 加载图片 (路径拼接修复) ===
        if image_rel_path:
            # 组合绝对路径：D:\vlm_assembly\data + temp_frames\xxx.jpg
            full_image_path = self.image_root / image_rel_path

            if full_image_path.exists():
                image = Image.open(full_image_path).convert("RGB")
            else:
                # 尝试修复路径分隔符问题
                print(f"Warning: Image missing at {full_image_path}, trying adjustment...")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # === 提取文本 ===
        messages = item.get("conversations") or item.get("messages")
        user_content = ""
        assistant_content = ""

        for msg in messages:
            if msg['role'] == 'user':
                if isinstance(msg['content'], str):
                    user_content = msg['content']
                elif isinstance(msg['content'], list):
                    for c in msg['content']:
                        if c.get('type') == 'text':
                            user_content += c['text']
            elif msg['role'] == 'assistant':
                if isinstance(msg['content'], str):
                    assistant_content = msg['content']
                elif isinstance(msg['content'], list):
                    for c in msg['content']:
                        if c.get('type') == 'text':
                            assistant_content += c['text']

        # === Processor ===
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}]
            }
        ]

        text_input = self.processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=False,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"]
        }


@dataclass
class RobustQwenCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            if ids.dim() > 1: ids = ids.squeeze(0)
            input_ids.append(ids)
            labels.append(ids)

            mask = f["attention_mask"]
            if mask.dim() > 1: mask = mask.squeeze(0)
            attention_mask.append(mask)

            pv = f["pixel_values"]
            if pv.dim() > 1: pv = pv.squeeze(0)
            pixel_values.append(pv)

            thw = f["image_grid_thw"]
            if isinstance(thw, torch.Tensor):
                if thw.dim() > 2: thw = thw.squeeze(0)
            elif isinstance(thw, np.ndarray):
                thw = torch.from_numpy(thw)
                if thw.dim() > 2: thw = thw.squeeze(0)
            image_grid_thw.append(thw)

        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt"
        )

        labels_padded = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt"
        )["input_ids"]
        labels_padded[labels_padded == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels_padded

        batch["pixel_values"] = torch.cat(pixel_values, dim=0).to(dtype=torch.bfloat16)
        batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0).to(dtype=torch.long)

        return batch


def train():
    print("正在初始化 Processor 和 模型...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"Loading Dataset from {DATA_FILE}...")
    # 传入 image_root，确保 dataset 知道去哪里找图片
    train_dataset = AssemblyDataset(DATA_FILE, IMAGE_ROOT, processor)
    print(f"Dataset loaded. Total samples: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),  # 转为 string
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=str(OUTPUT_DIR / "runs"),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        # 优化 checkpoint 管理
        save_total_limit=2,  # 只保留最近的2个 checkpoint，节省硬盘
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=RobustQwenCollator(processor=processor),
    )

    print("开始训练...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print(f"训练完成！模型已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()