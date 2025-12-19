import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 您的全覆盖坐标 (基于参考图)
ROI_BOX = (257, 413, 1031, 576)  # (x, y, w, h)

# 2. 【关键】参考分辨率
# 您之前画框时那张图片的原始分辨率。
# 如果您不确定，通常工业相机或截图是 1920x1080。
# 如果您的参考图是其他尺寸，请务必修改这里！
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

# ===========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# 输入: 清洗过的简版 JSON
INPUT_JSON = DATA_DIR / "clean_assembly_sft_12_12simple.json"
# 输出: 聚焦后的新 JSON
OUTPUT_JSON = DATA_DIR / "clean_assembly_sft_12_12focused.json"

# 图片源目录
SRC_IMG_DIR = DATA_DIR
# 图片输出目录
DST_IMG_DIR = DATA_DIR / "temp_frames_focused"


# ===========================================

def process_dataset():
    print(f"正在读取数据: {INPUT_JSON}")
    if not INPUT_JSON.exists():
        print(f"❌ 错误: 找不到文件 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建输出目录
    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"裁剪后的图片将保存至: {DST_IMG_DIR}")
    print(f"目标参考分辨率: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
    print(f"裁剪区域 ROI: {ROI_BOX}")

    x, y, w, h = ROI_BOX
    processed_count = 0
    resized_count = 0  # 统计有多少图被缩放了
    error_count = 0

    new_data = []

    for item in tqdm(data, desc="自适应缩放裁剪中"):
        new_item = item.copy()
        messages = new_item.get("conversations") or new_item.get("messages")

        has_image = False
        if not messages: continue

        for msg in messages:
            if msg['role'] == 'user' and isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'image':
                        original_rel_path = content['image']
                        full_src_path = SRC_IMG_DIR / original_rel_path

                        if full_src_path.exists():
                            # 1. 读取原图
                            img = cv2.imread(str(full_src_path))

                            if img is not None:
                                cur_h, cur_w = img.shape[:2]

                                # === 核心逻辑：自适应缩放 ===
                                # 如果当前图片尺寸 不等于 参考尺寸，则强制缩放
                                if cur_w != REFERENCE_WIDTH or cur_h != REFERENCE_HEIGHT:
                                    # print(f"缩放图片: {original_rel_path} ({cur_w}x{cur_h} -> {REFERENCE_WIDTH}x{REFERENCE_HEIGHT})")
                                    img = cv2.resize(img, (REFERENCE_WIDTH, REFERENCE_HEIGHT),
                                                     interpolation=cv2.INTER_LINEAR)
                                    resized_count += 1

                                # 2. 现在坐标系对齐了，安全裁剪
                                # 增加边界保护，防止万一 ROI 画出界
                                safe_y = max(0, y)
                                safe_x = max(0, x)
                                safe_h = min(h, REFERENCE_HEIGHT - safe_y)
                                safe_w = min(w, REFERENCE_WIDTH - safe_x)

                                cropped_img = img[safe_y: safe_y + safe_h, safe_x: safe_x + safe_w]

                                # 3. 保存
                                if cropped_img.size > 0:
                                    file_name = Path(original_rel_path).name
                                    save_path = DST_IMG_DIR / file_name
                                    try:
                                        cv2.imwrite(str(save_path), cropped_img)

                                        # 4. 更新 JSON 路径
                                        content['image'] = f"temp_frames_focused/{file_name}"
                                        has_image = True
                                    except Exception as e:
                                        print(f"❌ 保存出错 {file_name}: {e}")
                                        error_count += 1
                                else:
                                    print(f"⚠️ 裁剪结果为空: {file_name}")
                                    error_count += 1
                            else:
                                print(f"❌ 图片损坏: {original_rel_path}")
                                error_count += 1
                        else:
                            # print(f"⚠️ 文件不存在: {original_rel_path}")
                            pass

        if has_image:
            new_data.append(new_item)
            processed_count += 1

    # 保存新的 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 30)
    print(f"✅ 全量处理完成！")
    print(f"共输出数据: {processed_count} 条")
    print(f"触发缩放修复: {resized_count} 条 (这些原本会导致黑图或报错)")
    print(f"失败: {error_count} 条")
    print(f"新数据集路径: {OUTPUT_JSON}")
    print("=" * 30)
    print("现在，所有图片都已统一对齐。请检查 temp_frames_focused 文件夹，确认图片清晰且包含手部。")


if __name__ == "__main__":
    process_dataset()