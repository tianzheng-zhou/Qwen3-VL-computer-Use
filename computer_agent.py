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

MODEL = "qwen3.5-plus"
# MODEL = "qwen3.5-flash"
# MODEL = "qwen3-vl-plus"
# MODEL = "qwen3-vl-flash"


def get_client():
    """创建 OpenAI 客户端"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL")
    )


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
    :return:
    tuple(action)
    """
    client = get_client()

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


def summarize_history(summary_messages, model_id):
    """
    调用模型对历史操作进行摘要压缩。
    :param summary_messages: 用于摘要的消息列表
    :param model_id: 模型ID
    :return: 摘要文本
    """
    client = get_client()
    completion = client.chat.completions.create(
        model=model_id,
        messages=summary_messages,
    )
    summary_text = completion.choices[0].message.content
    print(f"=== 历史操作摘要 ===\n{summary_text}\n=== 摘要结束 ===")
    return summary_text


# 读取 prompt 文件夹下的所有文件
prompt_dir = "prompt"
if not os.path.exists(prompt_dir):
    os.makedirs(prompt_dir)

files = [f for f in os.listdir(prompt_dir) if os.path.isfile(os.path.join(prompt_dir, f))]

if not files:
    print(f"警告: {prompt_dir} 文件夹为空，请添加提示词文件。")
    user_query = input("请输入你的需求: ")
else:
    print(f"在 {prompt_dir} 文件夹中发现以下提示词文件:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    while True:
        try:
            choice = int(input("请选择要使用的提示词文件编号: "))
            if 1 <= choice <= len(files):
                selected_file = files[choice - 1]
                with open(os.path.join(prompt_dir, selected_file), "r", encoding="utf-8") as f:
                    user_query = f.read().strip()
                print(f"已选择文件: {selected_file}")
                print(f"读取到的需求: {user_query}")
                break
            else:
                print("无效的编号，请重新输入。")
        except ValueError:
            print("请输入有效的数字编号。")

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
    # 检查是否需要对历史操作进行摘要压缩
    if message.needs_summary():
        print(f"图片数量达到 {message.image_count}，触发历史摘要压缩...")
        summary_messages = message.get_summary_messages()
        summary_text = summarize_history(summary_messages, MODEL)
        message.compress_with_summary(summary_text)

    image_path = take_screenshot(target_width=actual_screen_width, target_height=actual_screen_height)
    message.add_image_message(image_path=image_path)

    print(message.messages)
    output_text, action, computer_use = get_qwen3_vl_action(message.messages, MODEL)
    
    # 修正模型可能输出 click 而非 left_click 的问题
    if action["arguments"]["action"] == "click":
        action["arguments"]["action"] = "left_click"
        
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
