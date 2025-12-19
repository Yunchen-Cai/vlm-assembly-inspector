# video_to_lsf_dataset.py  —— 终极 Windows 版（已自动创建文件夹）
import os
import json
import whisper
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm

# ===== 新增：自动创建文件夹 =====
os.makedirs("../data/temp_frames12.6", exist_ok=True)  # 关键一行！自动创建 temp_frames12.6

model = whisper.load_model("large-v3")  # 已经下载好了，自动走缓存
data = []

video_dir = "../data/standard_videos"
output_file = "../data/assembly_sft.json"

for video_file in tqdm(os.listdir(video_dir)):
    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(video_dir, video_file)
    print(f"\n处理 {video_file}")

    # Whisper 转录（中文自动识别）
    result = model.transcribe(video_path, language="zh", word_timestamps=True)

    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()

    for seg in result['segments']:
        start_sec = seg['start']
        end_sec = seg['end']
        text = seg['text'].strip()

        # 计算起止帧
        start_f = max(0, int(start_sec * fps))
        end_f = min(len(vr) - 1, int(end_sec * fps))

        # 均匀抽最多16帧
        frame_indices = []
        if end_f - start_f >= 16:
            step = (end_f - start_f) // 15
            frame_indices = list(range(start_f, end_f, step))[:16]
        else:
            frame_indices = list(range(start_f, end_f + 1))

        # 读取并保存图片
        frames = vr.get_batch(frame_indices).asnumpy()
        image_paths = []
        for i, frame in enumerate(frames):
            img_path = f"data/temp_frames12.6/{video_file}_{start_f + i}.jpg"
            Image.fromarray(frame).save(img_path)
            image_paths.append(img_path)

        # 添加一条训练样本
        data.append({
            "messages": [
                {"role": "user", "content": [{"type": "video", "video": image_paths}]},
                {"role": "assistant", "content": text}
            ]
        })

# 保存最终数据集
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n数据集生成完成！共 {len(data)} 条样本，保存为 {output_file}")
print("temp_frames12.6 文件夹已生成，可在训练完后删除")