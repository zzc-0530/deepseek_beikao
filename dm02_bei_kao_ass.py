import os
from openai import OpenAI

# 导入模型
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 系统提示词
system_prompt = """
你是一位有50年经验的资深备考顾问。
用户信息：
- 姓名：小明
- 目标考试：考研英语
- 距离考试：90天
- 当前基础：四级水平，语法薄弱

第一步：针对性提问2-3个问题，聚焦薄弱点和学习习惯。
第二步：根据用户回答，生成个性化备考计划。
第三步：给出鼓励性总结。
全程语气温暖、专业。
"""

# 历史对话记录
history = []

# 对话函数
def chat(user_input):
    # 把用户新消息加入历史
    history.append({
        "role": "user",
        "content": user_input
    })

    # 把完整历史传给AI
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1000,
        messages=[
                     {"role": "system", "content": system_prompt}
                 ] + history  # system + 完整历史
    )

    ai_reply = response.choices[0].message.content

    # 把AI回复也加入历史
    history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return ai_reply


# 对话循环
print("备考助手已启动！输入 '退出' 结束对话\n")
print("助手：你好！我是你的备考顾问，我们开始吧。\n")

while True:
    user_input = input("你：")

    if user_input == "退出":
        print("助手：祝你备考顺利！加油 💪")
        break

    reply = chat(user_input)
    print(f"\n助手：{reply}\n")