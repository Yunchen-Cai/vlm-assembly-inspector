print(">>> 脚本已启动，正在初始化...")  # 这一行用来检测脚本是否被执行

import json
import re
import os
import ast
from pathlib import Path

# ================= 配置区域 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# 输入文件 (您现在的 json)
INPUT_FILE = DATA_DIR / "clean_assembly_sft_12_12new.json"
# 输出文件 (清洗后的 json)
OUTPUT_FILE = DATA_DIR / "clean_assembly_sft_12_12simple.json"

# 新的 Prompt (去除 Description 要求)
NEW_SYSTEM_PROMPT = (
    "You are an assembly quality inspector. Observe this frame. "
    "Identify the hand action, tool, part, and phase. "
    "Output ONLY in this JSON format: "
    "{\"action\": \"...\", \"tool\": \"...\", \"part\": \"...\", \"phase\": \"...\"}."
)


# ===========================================

def robust_load_json(file_path):
    """
    尝试多种方式读取 JSON，容忍格式微小错误
    """
    print(f"正在读取文件: {file_path}")
    if not file_path.exists():
        print(f"❌ 错误: 找不到文件 {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 方法1: 标准 JSON 读取
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"⚠️ 标准 JSON 读取失败 (第 {e.lineno} 行): {e.msg}")
        print("🔄 尝试使用 Python AST 宽松模式读取...")

    # 方法2: Python AST 读取 (能处理末尾逗号、单引号等)
    try:
        # 预处理：把 JSON 的 null/true/false 替换为 Python 的 None/True/False
        content_py = content.replace("null", "None").replace("true", "True").replace("false", "False")
        data = ast.literal_eval(content_py)
        print("✅ AST 读取成功！已自动修复格式错误。")
        return data
    except Exception as e:
        print(f"❌ 宽松读取也失败了。错误信息: {e}")
        # 打印出错位置附近的文本帮助排查
        lines = content.splitlines()
        try:
            # 这里的 18 是基于您之前的报错 (line 19)
            err_line = 18
            print("\n--- 出错位置预览 ---")
            for i in range(max(0, err_line - 2), min(len(lines), err_line + 3)):
                prefix = ">> " if i == err_line else "   "
                print(f"{prefix}Line {i + 1}: {lines[i]}")
            print("--------------------")
        except:
            pass
        return []


def extract_json_part(text):
    """从混合文本中提取 JSON"""
    try:
        # 寻找 { ... }
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            # 简单清洗
            json_str = json_str.replace("'", '"')
            # 验证合法性
            json.loads(json_str)
            return json_str
    except:
        pass
    return None


def simplify_dataset():
    # 1. 读取
    data = robust_load_json(INPUT_FILE)
    if not data:
        print("❌ 数据加载失败，程序终止。")
        return

    print(f"原始数据量: {len(data)} 条")

    processed_data = []
    success_count = 0
    fail_count = 0

    for item in data:
        new_item = item.copy()

        msgs = new_item.get("conversations") or new_item.get("messages")
        if not msgs: continue

        valid_entry = True

        for msg in msgs:
            if msg['role'] == 'user':
                # 修改 Prompt
                if isinstance(msg['content'], list):
                    for content in msg['content']:
                        if content['type'] == 'text':
                            content['text'] = NEW_SYSTEM_PROMPT

            elif msg['role'] == 'assistant':
                # 清洗回答
                original_content = msg['content']

                # 如果已经是字典
                if isinstance(original_content, dict):
                    msg['content'] = json.dumps(original_content, ensure_ascii=False)
                # 如果是字符串
                elif isinstance(original_content, str):
                    clean_json = extract_json_part(original_content)
                    if clean_json:
                        msg['content'] = clean_json
                    else:
                        # 尝试直接用原文本（如果它本身就是JSON）
                        if original_content.strip().startswith("{") and original_content.strip().endswith("}"):
                            msg['content'] = original_content
                        else:
                            # 实在提不出来，这一条数据就废了
                            # print(f"⚠️ 丢弃脏数据: {original_content[:30]}...")
                            valid_entry = False

        if valid_entry:
            processed_data.append(new_item)
            success_count += 1
        else:
            fail_count += 1

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 30)
    print(f"✅ 清洗完成！")
    print(f"保留数据: {success_count} 条")
    print(f"清洗掉脏数据: {fail_count} 条")
    print(f"新文件路径: {OUTPUT_FILE}")
    print("=" * 30)


# ================= 关键入口 =================
# 请确保这几行代码在文件的最底部，且没有缩进
if __name__ == "__main__":
    simplify_dataset()