import cv2
import json
import os
from pathlib import Path

# ================= 配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
# 这里随便选一张典型的图片来定坐标
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data" / "temp_frames" / "202512121929_step1_frame_2.jpg"


# =======================================

def get_coordinates():
    image_path = str(SAMPLE_IMAGE_PATH)

    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        print("请修改代码中的 SAMPLE_IMAGE_PATH 为您的一张真实图片路径")
        return

    # 读取图片
    img = cv2.imread(image_path)
    # 为了方便画框，如果图太大，可以缩放一下显示（但这会影响坐标换算，建议原图显示）
    # 如果原图太大屏幕放不下，请告诉我，我加缩放逻辑。现在默认原图。

    print("\n" + "=" * 40)
    print("操作指南:")
    print("1. 弹窗出现后，按住鼠标左键【框选】您关注的手部操作区域。")
    print("2. 选好后，按【SPACE (空格键)】或【ENTER (回车键)】确认。")
    print("3. 如果想取消，按【c】键。")
    print("=" * 40 + "\n")

    # 调用 OpenCV 的 ROI 选择器
    roi = cv2.selectROI("Select Hand Operation Area", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("❌ 未选择区域或区域无效。")
        return

    print("\n" + "=" * 40)
    print(f"✅ 坐标提取成功！请记下这组数字：")
    print(f"X (左上角横坐标): {x}")
    print(f"Y (左上角纵坐标): {y}")
    print(f"W (宽度):        {w}")
    print(f"H (高度):        {h}")
    print("-" * 40)
    print(f"代码配置格式: ROI_BOX = ({x}, {y}, {w}, {h})")
    print("=" * 40)


if __name__ == "__main__":
    get_coordinates()