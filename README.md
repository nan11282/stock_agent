# A股投资助理

AI 驱动的 A 股投资助手。CLI 对话 + Telegram 聊天机器人，支持行情分析、持仓管理、定时研报推送。

**核心功能：**
- **Telegram Bot**：手机上跟 Agent 聊天，股票分析/持仓管理/闲聊都行，事件驱动零 CPU 空转
- **PE 历史百分位**：基于近 10 年财务+行情数据，判断当前 PE 在历史中的分位（越低越便宜）
- **每日报告**：15:30 自动扫描持仓/自选/市场发现，LLM 深度分析（800+ 字研报）
- **CLI 对话**：ReAct Agent + 16 个工具（行情/财务/分红/持仓管理/记忆检索）
- **记忆系统**：ChromaDB 向量 + SQLite FTS5 全文检索，RRF 融合，跨设备共享上下文
- **后台管理**：本地 Web 管理长期洞察、持仓、自选、决策日志和扫描快照

**技术栈**：Python 3.12 · DeepSeek · Telegram Bot API · AKShare · SQLite + ChromaDB FTS5 · Docker · 标准库 Admin

---

## 环境准备

需要：
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）
- QQ 邮箱，开启 SMTP（QQ邮箱 → 设置 → 账户 → 开启 SMTP 服务 → 生成授权码）
- Telegram Bot Token（搜 [@BotFather](https://t.me/BotFather) → `/newbot` 创建）

> SMTP 用于发送每日邮件报告；Telegram Bot 用于手机聊天交互。

---

## 配置

### 第一步：克隆项目

```bash
git clone <repo-url>
cd stock_agent
```

### 第二步：配置环境变量

在项目根目录新建 `.env`：

```env
DEEPSEEK_API_KEY_stock_agent=sk-xxxxxxxxxxxxxxxx

MAIL_USER=你的QQ邮箱@qq.com
MAIL_PASS=你的SMTP授权码（16位）
MAIL_TO=接收报告的邮箱地址

TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11

ADMIN_TOKEN=一段足够长的后台口令
```

> `.env` 已加入 `.gitignore`，不会被提交。

### 第三步：构建并启动

```bash
docker compose build
docker compose up -d
```

默认启动两个服务：`agent` 和 `scheduler`（定时扫描 + Telegram Bot）。后台管理按需启动，避免未配置口令时暴露管理入口。

---

## 使用

### Telegram Bot（推荐日常使用）

手机上打开你的 bot，直接发消息：

```
> 分析一下招商银行 600036
> 我买了 5000 股工商银行 成本 5.2
> 查看我的持仓
> 最近市场有什么机会吗
```

消息发给 Agent（16 个工具全开），LLM 自己判断是调工具查数据还是闲聊。写入操作（记录决策、改持仓）会在 Agent 展示内容后问你"确认"。

支持 `/reset` 清空对话历史（数据库不动）。

### CLI 对话

```bash
docker compose exec -it agent python app/main.py
```

### 后台管理

```bash
docker compose --profile admin up -d admin
```

打开 [http://127.0.0.1:8787](http://127.0.0.1:8787)，用 `.env` 里的 `ADMIN_TOKEN` 登录。后台默认只映射到本机，可以直接查看和管理：

| 页面 | 用途 |
|------|------|
| 记忆 | 新增、编辑、删除长期洞察；同步维护 SQLite、FTS5 和 ChromaDB |
| 持仓 | 新增、更新、删除当前组合事实 |
| 自选 | 管理关注原因、估值/价格提醒阈值和人工复核备注 |
| 决策 | 追加或删除决策日志，保持历史判断可追溯 |
| 扫描 | 查看每日扫描快照 |

### 手动触发扫描

```bash
docker compose exec scheduler python app/scheduler.py --now
```

### 每日定时

每天 **15:30**（收盘后）自动扫描并发邮件报告，无需任何操作。

---

## 报告内容

每份研报覆盖：

| 维度 | 内容 |
|------|------|
| 估值 | PE、PB、PE 历史百分位、TTM 股息率 |
| 业务 | 公司做什么、行业地位、竞争优势 |
| 财务 | ROE 趋势、营收利润增长、负债率、现金流 |
| 分红 | 历史分红记录、股息稳定性 |
| 建议 | 买入/观望/卖出 + 合理价位区间 |

每次推荐 1-2 只当前最具价值的股票。

---

## 架构

```
stock_agent/
├── main.py           # CLI 入口
├── agent.py          # ReAct Agent 核心循环
├── admin.py          # 本地后台管理
├── tools.py          # 16 个工具（9 读 + 7 写）+ 腾讯行情
├── adapters.py       # LLM 适配层（Claude / OpenAI / DeepSeek）
├── memory.py         # SQLite + ChromaDB + RRF 混合检索
├── scheduler.py      # 定时扫描 + 深度分析（ReAct + 读工具）
├── telegram_bot.py   # Telegram 聊天机器人（事件驱动长轮询）
├── mailer.py         # HTML 邮件发送
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

行情数据源：腾讯 `qt.gtimg.cn`（实时）+ Tencent K-line（历史），Docker 内可用无需代理。

记忆系统：每个 Telegram 用户独立 Agent 实例（独立聊天历史），底层 ChromaDB + SQLite 共享（跨设备检索历史分析洞察）。

---

## 许可

MIT
