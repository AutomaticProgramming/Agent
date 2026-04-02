# Agent 记忆系统方案

基于 `agent_2.py` 实现短期记忆和长期记忆

## 1. 短期记忆（Short-term Memory）

- **本质**：当前会话的 `messages` 列表
- **增强**：当消息数超过阈值（如 10 条）时，调用 LLM 将前面所有消息压缩为一段摘要，替换原文，保留对话脉络

## 2. 长期记忆（Long-term Memory）

- **本质**：跨会话持久化的关键信息，存储为 `./memory/long_term.json`
- **格式**：`[{"key": "主题词", "value": "具体内容", "timestamp": "2026-04-01 10:30"}, ...]`
- **操作**：`save_memory`（写） + `search_memory`（按关键词匹配 key 读取）

## 3. 记忆判断机制（方案 A — System Prompt 指导）

在 system prompt 中写入判断规则，让 LLM 通过 Function Calling 主动调用 `save_memory`：

```
Save important information using save_memory when you encounter:
- User preferences and habits (e.g. "I like concise answers")
- Project decisions and constraints (e.g. "no vector DB")
- Facts the user explicitly asks you to remember
- Mistakes you made and the user corrected

Be selective — only save truly useful information.
```

Agent 在正常对话过程中，检测到上述信息时会**自动触发** `save_memory` 工具调用，与普通工具调用（bash/read/write）混在同一次 LLM 响应中，无需额外轮次。

## 4. 工具列表扩展

| 工具 | 用途 |
|------|------|
| `execute_bash` | 执行 shell 命令 |
| `read_file` / `write_file` | 文件读写 |
| `save_memory` | **新增**，保存键值对到 long_term.json |
| `search_memory` | **新增**，按关键词匹配 key 读取长期记忆 |
| `summarize_memory` | **新增**，用 LLM 压缩 messages 列表 |

## 5. `run_agent` 流程

```
启动
  → 加载 long_term.json 全部条目
  → 注入 system prompt（"你当前的记忆: ..."）

Agent 主循环（每轮用户输入）:
  → LLM 调用（含 tools 列表 + messages 历史）
  → 若有 tool_calls:
      - save_memory      → 追加写入 long_term.json
      - search_memory     → 从 long_term.json 过滤返回
      - summarize_memory  → 调用 LLM 生成摘要，压缩 messages
      - bash/read/write   → 原有逻辑
  → 若无 tool_calls → 返回最终答案
  → 检查 messages 长度 > 阈值 → 触发自动摘要压缩
```

## 6. 新增组件

### MemoryManager 类

封装以下能力：
- `load()` — 从 long_term.json 加载全部条目
- `save(key, value)` — 追加或更新条目
- `search(query)` — 按关键词匹配 key，返回子集
- `summarize(messages, client)` — 调用 LLM 生成摘要并压缩 messages
