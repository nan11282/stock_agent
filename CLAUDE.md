# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A股（Chinese A-share）投资助理。本地运行，两个独立进程共享同一份 SQLite + ChromaDB 数据：
- `main.py` — 交互式 CLI（Agent + ReAct 循环）
- `scheduler.py` — 每日 15:30 定时扫描持仓/自选/发现，通过 QQ SMTP 发邮件

技术栈：Python 3.12 · AKShare（行情/财务/分红数据源） · ChromaDB · SQLite FTS5 · jieba（中文分词）· Anthropic SDK · OpenAI SDK（DeepSeek 复用）

## Run / Dev loop

全部跑在 Docker 里。`docker-compose.yml` 已把项目目录热挂载进 `/app`，**改 `.py` 代码不需要 rebuild**，只要重启服务即可。

```bash
# 构建镜像（仅首次，或改了 Dockerfile/requirements）
docker compose build

# 启动两个服务（agent + scheduler 后台运行）
docker compose up -d

# 进入 agent 容器跑 CLI（方向键/历史已通过 `import readline` 启用）
docker compose exec -it agent python main.py
#  或者 Docker Desktop → agent 容器 → Exec → `python main.py`

# 每日扫描：立即手动触发一次
docker compose exec scheduler python scheduler.py --now

# 自检（重构后验证神经没断）
docker compose run --rm --no-deps agent python -c \
  "from adapters import Message, ToolCall, ToolResult, ClaudeAdapter, OpenAIAdapter; \
   from agent import Agent; import scheduler; print('import ok')"
```

### 环境变量

主机 PowerShell 设好后，`docker-compose.yml` 的 `${VAR}` 语法会把它们透传进容器：

- `DEEPSEEK_API_KEY_stock_agent` — `main.py` 默认用的 DeepSeek key（切回 Claude 则改用 `ANTHROPIC_API_KEY`）
- `MAIL_USER` / `MAIL_PASS` / `MAIL_TO` — scheduler 发报所需的 SMTP 凭证（`.env` 已有默认）

## Architecture（必读）

### 1. 中性消息格式 —— Adapter 两头都要转

`adapters.py` 定义了 **provider-agnostic** 的 `Message` / `ToolCall` / `ToolResult` dataclass。`Agent.history: list[Message]` 只存这个中性格式。每个 Adapter 内部负责：

- **发送时**：中性 `Message` → 目标 API 格式（Claude 的 content block / OpenAI 的 `tool_calls` 字段）
- **接收时**：API 响应 → 中性 `LLMResponse`（`text` + `tool_calls: list[ToolCall]`）

**不要**在 `agent.py` 里直接塞 Anthropic-flavored 的 `{"type": "tool_use", ...}` dict——这会让 OpenAI/DeepSeek 的 API 报 `unknown variant 'tool_use'`。新增 provider（Gemini 等）= 写一个新 Adapter 实现 `LLMAdapter.chat`，Agent 代码不动。

### 2. 记忆系统：SQLite + ChromaDB + RRF 融合

`memory.py` 里两个独立存储，通过 `doc_id` 关联：

- **`DecisionLog`（SQLite）** — 结构化：`decisions`（append-only，严禁 UPDATE）/ `positions` / `watchlist` / `retrospectives` / `scan_results` / `episodic_docs` + `episodic_fts`（FTS5 虚拟表）
- **`EpisodicMemory`（ChromaDB）** — 每轮对话结束由 `agent._save_conversation_insight` 调 LLM 总结后写入，embedding 用 Chroma 默认的 ONNX MiniLM

**检索路径**（`EpisodicMemory.retrieve`）：同一 query 同时走两路 —— ChromaDB 向量检索（语义）+ FTS5 全文检索（精确词、对股票代码/数字敏感），用 **RRF (K=60)** 融合后取 top_k。FTS5 查询前必须用 **jieba 分词**，token 之间用 ` OR ` 连接传给 `MATCH`（中文直接 MATCH 会零召回）。

### 3. 工具权限双轨制

`tools.py` 的 `READ_TOOLS` / `WRITE_TOOLS` 分离是**系统设计约束**，不是文档风格：

- **READ**（9 个）：Agent 可自主调用
- **WRITE**（7 个）：每个 tool description 里都写了【严格限制】—— 用户明确说操作词 + Agent 展示内容 + 用户回 "确认" 后才允许调用。**简化或删除这段 description 文字会直接破坏 Agent 的行为约束**

`system prompt`（`agent.py` 顶部的 `SYSTEM_PROMPT_TEMPLATE`）里【工具使用规则——不得违反】那段同样是 load-bearing。

### 4. 两进程共享 DB

`agent` 和 `scheduler` 两个容器同时打开 `/app/data/investment.db`。`DecisionLog._init_schema` 已打开 `PRAGMA journal_mode=WAL` —— 并发读没问题，但**不要**在代码里关闭 WAL 或把 DB 放到任何不支持 mmap 的文件系统上。

### 5. pysqlite3 shim

`memory.py` 开头的 `pysqlite3` 导入必须**保持在所有其他 import 之前**。Docker `python:3.12-slim` 自带的 sqlite3 不带 FTS5，要靠 `pysqlite3-binary` 替换。改动 memory.py 时别无意中把这段移到下面。

### 6. CLI 输入卫生

`main.py` 里 `input()` 前要 `import readline`（方向键/历史），读入后用 `_CTRL_CHARS` 正则清掉残留 ANSI 控制字符 —— 否则奇怪字节流进 ChromaDB 的 ONNX tokenizer 会炸 `TextInputSequence must be str`。

## 切换 LLM

只改 `main.py` 的这一行：

```python
# llm = ClaudeAdapter(model="claude-opus-4-5")
llm = OpenAIAdapter(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
)
```

`scheduler.py` 里写死用 `ClaudeAdapter`，改 LLM 需要单独改它一行。

## 不要做的事

- 不要往 `agent.py` 的 `history` 里塞 provider-specific 的 dict 格式；一律用 `Message` 构造
- 不要改 WRITE 工具 description 里的【严格限制】段落，那是 Agent 行为守门员
- 不要对 `decisions` 表做 UPDATE，所有修改走 `retrospectives` 追加
- 不要在 FTS5 `MATCH` 里直接塞未分词的中文 query
