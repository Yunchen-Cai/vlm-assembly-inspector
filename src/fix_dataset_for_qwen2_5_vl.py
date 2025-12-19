# fix_json.py
import json
from pathlib import Path

input_file = "../data/assembly_sft_qwen2_5_vl.json"
output_file = "../data/clean_assembly_sft.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed_data = []
seen_paths = set()

for item in data:
    try:
        images = [c["image"] for c in item["messages"][0]["content"] if c["type"] == "image"]
        text = [c["text"] for c in item["messages"][0]["content"] if c["type"] == "text"][0]
        assistant = item["messages"][1]["content"].strip()
    except:
        continue  # 跳过格式错

    # 去重 + 跳空
    path_key = "_".join(images[:3])
    if not assistant or path_key in seen_paths:
        continue
    seen_paths.add(path_key)

    # 限 8 张图像
    images = images[:8]

    # 转 prompt 为英文（强烈推荐，提升模型输出）
    en_text = text.replace("你是一个装配质检AI。请观察这些连续视频帧，输出JSON格式：{\"action\": \"手部动作(e.g., grasp/place/tighten)\", \"tool\": \"工具(e.g., screwdriver/none)\", \"part\": \"零件(e.g., screw_A/chassis)\", \"is_correct\": true/false, \"suggestion\": \"如果错误，给出实时建议\"}。基于标准装配流程判断。",
                           "You are an assembly quality inspector. Observe these consecutive video frames and output in JSON format: {\"action\": \"hand action (e.g., grasp/place/tighten)\", \"tool\": \"tool (e.g., screwdriver/none)\", \"part\": \"part (e.g., screw_A/chassis)\", \"is_correct\": true/false, \"suggestion\": \"if wrong, give real-time suggestion\"}. Judge based on standard assembly process.")

    # 构建固定条
    fixed_item = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": en_text}]
            },
            {
                "role": "assistant",
                "content": assistant  # 保持中文
            }
        ]
    }
    fixed_data.append(fixed_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print(f"修复完成！{len(fixed_data)} 条，文件: {output_file}")