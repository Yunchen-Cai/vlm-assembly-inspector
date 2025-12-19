import json
import os

# 1. 配置
INPUT_FILE = "../data/clean_assembly_sft_12_12new.json"
OUTPUT_FILE = "../data/clean_assembly_sft_12_12flattened.json"  # 这是我们将要在训练中使用的文件


def preprocess():
    print(f"正在读取原始文件: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    print(f"原始数据量: {len(data)}")
    flattened_data = []
    success_count = 0
    fail_count = 0

    for i, item in enumerate(data):
        image_path = None

        # === 核心逻辑：从深层嵌套中挖掘图片路径 ===
        # 结构通常是 item["messages"][0]["content"] 里的某一个元素
        if "messages" in item:
            for message in item["messages"]:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "image":
                            image_path = content["image"]
                            break
                if image_path: break

        # 兼容逻辑：万一图片在最外层（虽然刚才证明了不在）
        elif "image" in item:
            image_path = item["image"]

        # === 格式标准化 ===
        if image_path:
            # 1. 确保 image 是列表格式
            if isinstance(image_path, str):
                image_list = [image_path]
            elif isinstance(image_path, list):
                image_list = image_path
            else:
                image_list = []

            # 2. 构造新的扁平对象
            new_item = {
                "image": image_list,  # 提取出来的图片，放在最外层
                "conversations": item["messages"]  # 原始对话
            }
            flattened_data.append(new_item)
            success_count += 1
        else:
            print(f"警告: 第 {i} 条数据找不到图片路径，跳过。")
            fail_count += 1

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flattened_data, f, indent=4, ensure_ascii=False)

    print("\n=== 处理完成 ===")
    print(f"成功提取并扁平化: {success_count} 条")
    print(f"失败/丢弃: {fail_count} 条")
    print(f"新文件已保存为: {OUTPUT_FILE}")
    print("现在，这个文件的结构非常简单，'image' 字段就在最外层，且一定是列表。")


if __name__ == "__main__":
    preprocess()