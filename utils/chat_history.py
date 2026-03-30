import base64
from pathlib import Path
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from utils.agent_function_call import ComputerUse

computer_use = ComputerUse(cfg={"display_width_px": 1000, "display_height_px": 1000})

# 最大保留图片数量，超过此数量时触发历史摘要压缩
MAX_IMAGES_BEFORE_SUMMARY = 15

SYSTEM_PROMPT = '''
<role>
你是一个计算机操作助手，能够通过鼠标和键盘操控计算机来完成用户指定的任务。
你将通过观察屏幕截图来理解当前界面状态，并决定下一步操作。
</role>

<capabilities>
- 观察屏幕截图，理解当前界面状态和元素位置
- 通过鼠标点击、拖拽、滚动等操作与界面交互
- 通过键盘输入文本和执行快捷键
- 判断任务是否完成并报告状态
</capabilities>

<instructions>
- 每次收到屏幕截图后，分析当前界面状态，决定下一步操作
- 操作时要精确定位界面元素的中心位置，不要点击元素边缘
- 如果点击操作未生效，尝试微调坐标位置后重试
- 某些应用启动或操作需要加载时间，必要时使用 wait 动作等待
- 当你判断任务已经完成时，使用 terminate 动作报告成功或失败
- 如果收到之前操作的摘要信息，请基于摘要继续后续操作，不要重复已完成的步骤
</instructions>

<response_format>
你必须严格使用以下 XML 标签格式响应，不要返回裸 JSON：

<tool_call>
{"name": "computer_use", "arguments": {"action": "动作名称", ...其他参数}}
</tool_call>

在 tool_call 标签之前，你可以先用简短的文字说明你的思考过程和操作意图。
</response_format>
'''.strip()

SUMMARY_PROMPT = '''
<task>
请总结以下计算机操作的历史记录。你需要提取关键信息，包括：
- 用户的原始任务目标
- 已经完成了哪些操作步骤
- 当前的界面状态和进度
- 下一步可能需要做什么
</task>

<format>
请用简洁的中文总结，保留所有对后续操作有帮助的关键信息。
</format>
'''.strip()


class Messages:
    def __init__(self, user_query):
        system_message = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text=SYSTEM_PROMPT)]),
            ],
            functions=[computer_use.function],
            lang="zh",
        )
        system_message = system_message[0].model_dump()

        self.user_query = user_query
        self.image_count = 0

        self.messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]} for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<user_task>\n{user_query}\n</user_task>"},
                ],
            }
        ]

    def add_image_message(self, image_path):
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.png': 'png',
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.webp': 'webp'
        }.get(ext, 'png')

        with open(image_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')

        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime_type};base64,{base64_data}"
                    },
                },
                {"type": "text", "text": "<screenshot>当前完成操作后的屏幕截图</screenshot>"},
            ],
        })
        self.image_count += 1

    def add_qwen_response(self, qwen_response):
        self.messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": qwen_response},
            ],
        })

    def needs_summary(self):
        """判断是否需要进行历史摘要压缩"""
        return self.image_count >= MAX_IMAGES_BEFORE_SUMMARY

    def get_summary_messages(self):
        """构建用于生成摘要的消息列表，包含到目前为止的所有对话历史"""
        summary_msgs = []
        # 添加摘要专用系统提示
        summary_msgs.append({
            "role": "system",
            "content": [{"type": "text", "text": SUMMARY_PROMPT}],
        })
        # 复制除 system 消息外的所有历史记录给摘要模型
        for msg in self.messages[1:]:
            summary_msgs.append(msg)
        # 追加请求摘要的指令
        summary_msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": "请总结以上所有操作历史。"}],
        })
        return summary_msgs

    def compress_with_summary(self, summary_text):
        """用摘要文本替换旧的历史消息，只保留 system + 用户任务 + 摘要 + 最近一轮对话"""
        system_msg = self.messages[0]
        user_task_msg = self.messages[1]

        # 保留最近一轮（最后的 assistant 回复 + 最后的 user 截图）
        recent_messages = []
        for msg in reversed(self.messages[2:]):
            recent_messages.insert(0, msg)
            # 保留最近一组 user+assistant 对
            if len(recent_messages) >= 2:
                break

        # 重建消息列表
        self.messages = [
            system_msg,
            user_task_msg,
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<operation_summary>\n{summary_text}\n</operation_summary>"},
                ],
            },
        ] + recent_messages

        # 重置图片计数（只保留了最近的截图）
        self.image_count = sum(
            1 for msg in self.messages
            if msg["role"] == "user"
            and any(item.get("type") == "image_url" for item in msg.get("content", []))
        )
        print(f"历史已压缩，当前图片数量: {self.image_count}")


