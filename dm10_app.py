import os
import streamlit as st
from openai import OpenAI
from dm06_deepseek_RAG import *
# import hashlib
# import json

# 调用DeepSeek
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# 只保留最近10条消息
MAX_HISTORY = 10



# # 用问题内容生成唯一key
# def get_cache_key(prompt, user_input):
#     content = prompt + user_input
#     return hashlib.md5(content.encode()).hexdigest()


# # 检查缓存
# @st.cache_data(ttl=3600)  # 缓存1小时
# def cached_response(cache_key, messages_json):
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=json.loads(messages_json)
#     )
#     return response.choices[0].message.content
#
#


# 模拟备考资料（真实场景可以读取PDF）
study_materials = [
    """
    LangChain核心概念：Chain是LangChain的基础单元，
    将多个组件串联起来完成复杂任务。常见的Chain包括
    LLMChain、RetrievalQAChain等。
    """,
    """
    LangGraph是LangChain的扩展，专门用于构建有状态的
    多步骤Agent工作流。它用图（Graph）的方式描述Agent
    的决策流程，支持循环和条件分支。
    """,
    """
    RAG面试常见问题：如何处理检索结果不相关的情况？
    可以使用重排序（Reranking）模型对检索结果进行
    二次筛选，提高准确率。
    """
]


# 页面配置
st.set_page_config(
    page_title="备考助手",
    page_icon="📚",
    layout="centered"
)

# 标题
st.title("📚 AI备考助手")
st.caption("输入你的考试目标，AI为你定制专属备考计划")



# 建立知识库（只需建一次）
@st.cache_resource
def get_knowledge_base():
    return build_knowledge_base(study_materials)

db = get_knowledge_base()



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
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


# 侧边栏显示
with st.sidebar:
    st.divider()
    st.caption(f"📊 本次会话Token消耗：约{st.session_state.total_tokens}")
    cost = st.session_state.total_tokens / 1000 * 0.001  # DeepSeek约¥0.001/千Token
    st.caption(f"💰 估算费用：约¥{cost:.4f}")


# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 初始化Token计数


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
    不确定的信息不要说，直接告诉用户"我不确定"
    推荐资料时只推荐你确定存在的
    每个推荐资料后标注：
    （来源：官方文档 / 业界通用做法）
    若无法确认来源，告知用户自行核实
    """
    # 在构建prompt后加一行
    system_prompt = " ".join(system_prompt.split())  # 压缩多余空格

    context = search_knowledge(db, user_input)

    # 把检索到的内容加入system prompt
    enhanced_prompt = system_prompt + f"""

        【参考资料】
        以下是与用户问题相关的备考资料，请优先基于这些内容回答：
        {context}
        """

    with st.chat_message("assistant"):
        # with st.spinner("思考中..."):
        placeholder = st.empty()
        full_reply = ""
        response = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=2000,
                temperature=0.2,
                stream=True,
                messages=[
                    {"role": "system", "content": enhanced_prompt}
                ] + st.session_state.messages[-MAX_HISTORY:]
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                placeholder.markdown(full_reply + "▌")
        placeholder.markdown(full_reply)
            # st.markdown(full_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_reply
    })
    # 在API调用时记录（流式模式需要最后统计）
    # 改用非流式先计数，或在流式结束后估算
    token_estimate = len(full_reply) * 2  # 粗略估算
    st.session_state.total_tokens += token_estimate


