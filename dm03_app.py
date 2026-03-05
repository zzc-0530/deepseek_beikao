import os
import streamlit as st
from openai import OpenAI

# 页面配置
st.set_page_config(
    page_title="备考助手",
    page_icon="📚",
    layout="centered"
)

# 标题
st.title("📚 AI备考助手")
st.caption("输入你的考试目标，AI为你定制专属备考计划")

# 侧边栏：用户信息输入
with st.sidebar:
    st.header("👤 你的信息")
    user_name = st.text_input("你的名字", value="同学")
    topic     = st.text_input("目标考试", value="AI应用面试")
    days      = st.number_input("距离考试天数", min_value=1, max_value=365, value=90)
    user_base = st.text_area("当前基础", value="了解基本概念，有RAG搭建经验")
    st.divider()
    st.caption("填写完信息后，直接在对话框里开始聊天 👇")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入框
user_input = st.chat_input("输入你的问题...")

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 构建system prompt
    system_prompt = f"""
    你是一位有50年经验的资深备考顾问。
    用户信息：
    - 姓名：{user_name}
    - 目标考试：{topic}
    - 距离考试：{days}天
    - 当前基础：{user_base}

    第一步：针对性提问2-3个问题，聚焦薄弱点和学习习惯。
    第二步：根据回答生成个性化备考计划，包含阶段划分、
            推荐资料、每日时间表。
    第三步：给出鼓励性总结。
    全程语气温暖、专业。
    """

    # 调用DeepSeek
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })