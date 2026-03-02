import os
import json
import dotenv
import ctypes  # 添加 ctypes 库用于获取系统缩放比例
from openai import OpenAI
from utils.agent_function_call import ComputerUse
from utils.take_screenshot import take_screenshot
from utils.chat_history import Messages

# dotenv.load_dotenv("/Users/zjlz/Qwen3-VL-cookbook/pythonProject6/.env")
dotenv.load_dotenv()

MODEL = "gui-plus"
# MODEL = "qwen3-vl-plus"
# MODEL = "qwen3-vl-flash"

# 获取Windows系统缩放比例的函数
def get_windows_scaling():
    """获取Windows系统的缩放比例"""
    try:
        # 获取系统DPI感知上下文
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()

        # 获取系统DPI
        hdc = user32.GetDC(0)
        logpixelsx = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELX
        user32.ReleaseDC(0, hdc)

        # 计算缩放比例 (96 DPI = 100%缩放)
        scale = logpixelsx / 96.0
        print(f"检测到系统缩放比例: {scale:.2f}x ({int(scale * 100)}%)")
        return scale
    except Exception as e:
        print(f"获取系统缩放比例失败: {e}，使用默认值1.0")
        return 1.0  # 默认返回100%缩放


# 图片编码

def get_qwen3_vl_action(messages, model_id):
    """
    使用 Qwen 模型执行 GUI 接地，以解释用户在屏幕截图上的查询。
    :param messages:
    :param model_id:
    :param min_pixels:
    :param max_pixels:
    :return:
    tuple(action)
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL")
    )

    # 初始化显示屏对象
    computerUse = ComputerUse(
        cfg={"display_width_px": 1000, "display_height_px": 1000}
    )

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    output_text = completion.choices[0].message.content
    print(output_text)
    action = json.loads(output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
    return output_text, action, computerUse


user_query = input("请输入你的需求")
message = Messages(user_query)
action_num = 1

# 获取系统缩放比例
system_scaling = get_windows_scaling()
# 设置实际屏幕分辨率
actual_screen_width = 1920
actual_screen_height = 1080

# 计算实际像素坐标（考虑缩放）
effective_width = actual_screen_width / system_scaling
effective_height = actual_screen_height / system_scaling
print(
    f"实际屏幕分辨率: {actual_screen_width}x{actual_screen_height}, 有效分辨率: {effective_width:.0f}x{effective_height:.0f}")

while True:
    image_path = take_screenshot(target_width=actual_screen_width, target_height=actual_screen_height)
    message.add_image_message(image_path=image_path)

    print(message.messages)
    output_text, action, computer_use = get_qwen3_vl_action(message.messages, MODEL)
    message.add_qwen_response(output_text)
    if action["arguments"]["action"] == "terminate":
        if action["arguments"]["status"] == "success":
            print(f"{user_query}成功🎆")
        else:
            print(f"{user_query}失败💔")
        break
    print(f"Qwen3 将要采取第{action_num}步行动:{action["arguments"]["action"]}")
    if action["arguments"]["action"] in ["left_click", "right_click", "middle_click", "double_click", "triple_click",
                                         "mouse_move", "left_click_drag"]:
        coordinate_relative = action['arguments']['coordinate']
        # 调整坐标转换，考虑系统缩放比例
        coordinate_absolute = [
            coordinate_relative[0] / 1000 * actual_screen_width,
            coordinate_relative[1] / 1000 * actual_screen_height
        ]
        action['arguments']['coordinate'] = coordinate_absolute
    print(action)
    computer_use.call(action['arguments'])
    action_num += 1
