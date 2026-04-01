import os
import json
import subprocess
from openai import OpenAI

# 初始化 OpenAI 客户端，通过环境变量支持兼容 OpenAI 接口的服务（如本地部署的 LLM）
# OPENAI_API_KEY: API 密钥
# OPENAI_BASE_URL: 自定义 API 网关（可选）
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

# 工具声明：以 OpenAI Function Calling 格式描述 Agent 可调用的工具列表
# 模型会根据上下文决定调用哪些工具，返回工具名称和 JSON 参数
tools = [
    {
        "type": "function",
        "function": {
            # 执行 shell 命令，使 Agent 具备系统操作能力
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # 读取指定路径的文件内容
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # 向指定路径写入文件内容，支持 Agent 生成和保存代码/文件
            "name": "write_file",
            "description": "Write to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


def execute_bash(command):
    """执行 shell 命令，返回标准输出和标准错误的合并结果。

    使用 subprocess.run + shell=True，支持管道、重定向等 shell 语法。
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def read_file(path):
    """读取指定路径的文件并以字符串形式返回内容。"""
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    """将内容写入指定路径的文件（覆盖模式），返回确认信息。"""
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"


# 工具名称到 Python 函数对象的映射，用于动态调用
functions = {"execute_bash": execute_bash, "read_file": read_file, "write_file": write_file}


"""
    运行 Agent 主循环：将用户输入发送给 LLM，处理工具调用，直到模型不再需要工具或达到最大迭代次数。
    Args:
        user_message: 用户的任务描述
        max_iterations: Agent 最多与 LLM 交互的轮数，防止无限循环
    Returns:
        模型最终回答文本，或 "Max iterations reached" 表示达到上限
"""
def run_agent(user_message, max_iterations=5):
    # 构造对话上下文：system prompt 定义 Agent 行为 + user 消息承载任务
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": user_message},
    ]
    # 迭代循环：每次调用 LLM 可能返回工具调用请求或直接回复
    for _ in range(max_iterations):
        # 请求 LLM 生成回复，附带可用工具列表
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,  # 包含历史对话，LLM 能感知之前的工具执行结果
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message)  # 将 LLM 的回复加入历史
        # 如果没有工具调用请求，说明 LLM 已经给出最终答案
        if not message.tool_calls:
            return message.content
        # 遍历所有工具调用
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {name}({args})")
            # 安全检查：确认工具名称在支持的映射中
            if name not in functions:
                result = f"Error: Unknown tool '{name}'"
            else:
                result = functions[name](**args)  # 解包参数调用实际函数
            # 将工具执行结果追加到消息历史，供 LLM 下一轮使用
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    return "Max iterations reached"


if __name__ == "__main__":
    import sys

    # 从命令行参数拼接任务描述，例如: python agent.py "读取 agent.py 并统计行数"
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent(task))
