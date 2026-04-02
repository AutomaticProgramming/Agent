import os
import json
import subprocess
from datetime import datetime
from openai import OpenAI

"""
    agent_2.py: 在 agent_1.py 基础上添加短期记忆和长期记忆系统

    短期记忆: 当前会话的 messages 列表，超过阈值自动 LLM 摘要压缩
    长期记忆: 跨会话持久化的 key-value 条目，存储在 ./memory/long_term.json
"""

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

# 短期记忆最大消息条数，超过后触发摘要压缩
SHORT_MEMORY_LIMIT = 10

# 长期记忆文件路径
LONG_MEMORY_PATH = "./memory/long_term.json"


# ======================== 工具定义 ========================

# OpenAI Function Calling 格式的工具描述列表
tools = [
    {
        "type": "function",
        "function": {
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
    {
        "type": "function",
        "function": {
            # 保存长期记忆：将 key-value 条目持久化到 long_term.json
            "name": "save_memory",
            "description": "Save a long-term memory entry (key-value pair) to persistent storage",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Topic or keyword, e.g. 'user_language_preference'"},
                    "value": {"type": "string", "description": "The actual memory content to remember"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # 搜索长期记忆：按关键词模糊匹配 key，返回匹配条目
            "name": "search_memory",
            "description": "Search long-term memory by keyword matching against keys",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search for in memory keys"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # 手动触发短期记忆摘要压缩（也可在 run_agent 中自动触发）
            "name": "summarize_memory",
            "description": "Summarize conversation history to save tokens when context is too long",
            "parameters": {
                "type": "object",
                "properties": {},  # 无需参数，由 Agent 判断需要压缩
            },
        },
    },
]


# ======================== 工具实现 ========================

def execute_bash(command):
    """执行 shell 命令，返回标准输出和标准错误的合并结果。"""
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


# ======================== 记忆管理 ========================

class MemoryManager:
    """负责长期记忆的持久化和短期记忆的摘要压缩。"""

    def __init__(self, path=LONG_MEMORY_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.memories = self._load()

    def _load(self):
        """从 JSON 文件加载长期记忆。文件不存在则返回空列表。"""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        """将内存条目写入 JSON 文件（覆盖写入）。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def save(self, key, value):
        """
        保存或更新一条长期记忆。
        如果 key 已存在则更新 value，否则追加新条目。
        """
        for entry in self.memories:
            if entry["key"] == key:
                # key 存在，更新 value 和时间戳
                entry["value"] = value
                entry["timestamp"] = datetime.now().isoformat()
                self._save()
                return f"Updated memory: {key}"
        # 新增条目
        entry = {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        self.memories.append(entry)
        self._save()
        return f"Saved memory: {key}"

    def search(self, query):
        """
        按关键词模糊匹配 key，返回匹配条目列表。
        query 为空时返回全部条目。
        """
        if not query:
            return self.memories
        matches = [e for e in self.memories if query.lower() in e["key"].lower()]
        return matches

    def get_all(self):
        """返回全部长期记忆条目（用于注入 system prompt）。"""
        return self.memories

    def summarize(self, messages, client, model="gpt-4o-mini"):
        """
        短期记忆摘要压缩：
        将 messages[1:]（跳过 system prompt）发给 LLM，要求生成一段简短摘要，
        然后用该摘要替换原始消息，释放 token 空间。

        Args:
            messages: 当前对话历史（将被修改）
            client: 当前 OpenAI client
            model: 用于摘要的模型

        Returns:
            压缩后的 messages 列表
        """
        # 提取需要摘要的消息（跳过 system prompt）
        to_summarize = [m for m in messages[1:] if m["role"] != "tool"]
        if len(to_summarize) <= 2:
            return messages  # 消息太少，无需压缩

        # 构造摘要请求
        summary_req = [
            {"role": "system", "content": "Summarize the following conversation concisely in 2-3 sentences. Keep key facts and decisions."},
            *to_summarize,
        ]
        resp = client.chat.completions.create(model=model, messages=summary_req)
        summary = resp.choices[0].message.content

        # 用摘要替换 system prompt 之后的所有消息（仅保留 system + 摘要 + 最新的一条用户消息防止断尾）
        system_msg = messages[0:1]  # system prompt
        last_user = [m for m in messages if m["role"] == "user"][-1:]  # 最新用户消息
        summary_msg = {"role": "assistant", "content": f"[Conversation summary] {summary}"}
        messages[:] = system_msg + [summary_msg] + last_user
        return messages


# 工具名称到 Python 函数的映射
functions = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
}

# 记忆管理器实例（全局唯一，跨轮次复用）
memory = MemoryManager()


# ======================== Agent 主循环 ========================

def _make_system_prompt(knowledge):
    """
    构造带有长期记忆注入和记忆保存指引的 system prompt。
    """
    mem_text = "\n".join(f"- {e['key']}: {e['value']}" for e in knowledge)
    mem_section = f"\nYou have the following long-term memory from past sessions:\n{mem_text}" if mem_text else ""

    return f"""You are a helpful assistant with long-term memory capability.{mem_section}

MEMORY RULES:
1. When you encounter user preferences, project decisions/constraints, facts they ask you to remember, or corrections to your mistakes, call save_memory(key, value) to record them.
2. Be selective — only save truly useful information.
3. Before starting work, call search_memory(query) if you think relevant past information might exist.
4. Keep responses concise."""


def run_agent(user_message, max_iterations=5):
    """
    带记忆系统的 Agent 主循环。

    短期记忆: messages 列表，超过 SHORT_MEMORY_LIMIT 条时自动摘要压缩
    长期记忆: MemoryManager 管理的持久化 key-value 存储

    Args:
        user_message: 用户的任务描述
        max_iterations: Agent 最多与 LLM 交互的轮数

    Returns:
        模型最终回答文本
    """
    # 加载长期记忆并构造 system prompt
    knowledge = memory.get_all()
    system_prompt = _make_system_prompt(knowledge)

    # 构造对话上下文（短期记忆起点）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for i in range(max_iterations):
        # 短期记忆检查：消息过多时自动摘要压缩
        if len(messages) > SHORT_MEMORY_LIMIT:
            print(f"[Memory] Summarizing {len(messages)} messages...")
            messages = memory.summarize(messages, client)

        # 请求 LLM
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message
        messages.append(message)

        # 没有工具调用 → LLM 给出最终答案
        if not message.tool_calls:
            return message.content

        # 处理工具调用
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {name}({args})")

            if name == "save_memory":
                # 保存长期记忆
                result = memory.save(args["key"], args["value"])
            elif name == "search_memory":
                # 搜索长期记忆
                matches = memory.search(args["query"])
                if matches:
                    result = json.dumps(matches, ensure_ascii=False)
                else:
                    result = "No matching memories found."
            elif name == "summarize_memory":
                # 手动触发摘要压缩
                messages = memory.summarize(messages, client)
                result = "Conversation history summarized."
            elif name in functions:
                # 原有工具调用
                result = functions[name](**args)
            else:
                result = f"Error: Unknown tool '{name}'"

            # 将工具执行结果追加到消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return "Max iterations reached"


if __name__ == "__main__":
    import sys

    # 从命令行参数拼接任务描述
    # 用法: python agent_3.py "读取 agent.py 内容，保存到 result.txt"
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello"
    print(run_agent(task))
