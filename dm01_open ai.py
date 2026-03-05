import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)


def bei_kao_assistant(user_name, topic, days, user_base):
    system_prompt = f"""
    你是一位有50年经验的资深备考顾问。
    用户信息：
    - 姓名：{user_name}
    - 目标考试：{topic}
    - 距离考试：{days}天
    - 当前基础：{user_base}

    第一步：针对性提问2-3个问题，聚焦薄弱点和学习习惯。
    第二步：生成个性化备考计划，包含阶段划分、推荐资料、每日时间表。
    第三步：给出鼓励性总结。
    全程语气温暖、专业。
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": "你好，我需要备考帮助"}
        ]
    )
    return response.choices[0].message.content

# 填入你的真实信息
result = bei_kao_assistant(
    user_name="你的名字",
    topic="你的考试",
    days=90,
    user_base="你的当前基础"
)

print(result)