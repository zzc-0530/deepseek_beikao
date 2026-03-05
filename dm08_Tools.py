import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ① 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_study_material",
            "description": "搜索备考资料和面试题，当用户需要学习资料时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如'LangGraph面试题'、'RAG优化方法'"
                    },
                    "material_type": {
                        "type": "string",
                        "enum": ["面试题", "学习资料", "实战项目"],
                        "description": "资料类型"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ② 模拟工具执行（真实场景可接入搜索API）
def execute_tool(tool_name, tool_args):
    if tool_name == "search_study_material":
        query = tool_args.get("query")
        material_type = tool_args.get("material_type", "学习资料")
        # 模拟返回搜索结果
        return f"""
        关于"{query}"的{material_type}：
        1. LangGraph官方文档：介绍状态图、节点、边的核心概念
        2. 常见面试题：LangGraph与LangChain的区别是什么？
        3. 实战项目：用LangGraph构建多步骤备考Agent
        """
    return "工具执行失败"

# ③ Agent主循环
def run_agent(user_input):
    messages = [
        {"role": "system", "content": "你是备考助手，有搜索工具可以使用。需要查资料时主动调用工具。"},
        {"role": "user",   "content": user_input}
    ]

    print(f"\n用户：{user_input}\n")

    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        choice = response.choices[0]
        print(choice)

        # 如果AI决定调用工具
        if choice.finish_reason == "tool_calls":
            tool_call = choice.message.tool_calls[0]
            tool_name = tool_call.function.name
            # json -> dict
            tool_args = json.loads(tool_call.function.arguments)

            print(f"🔧 Agent调用工具：{tool_name}")
            print(f"   参数：{tool_args}")

            # 执行工具
            tool_result = execute_tool(tool_name, tool_args)
            print(f"   结果：{tool_result[:50]}...\n")

            # 把工具结果加入对话
            messages.append(choice.message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # 如果AI决定直接回答
        elif choice.finish_reason == "stop":
            final_answer = choice.message.content
            print(f"助手：{final_answer}")
            return final_answer


run_agent("帮我找一些LangGraph的面试题")