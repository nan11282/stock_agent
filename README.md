# A股投资助理

AI 驱动的 A 股投资助手。CLI 对话分析个股 + 定时/手机触发深度研报推送。

**核心功能：**
- CLI 对话：ReAct Agent + 16 个工具（行情/财务/分红/持仓管理/记忆检索）
- PE 历史百分位：基于近 10 年财务+行情数据，判断当前 PE 在历史中的分位
- 每日报告：15:30 自动扫描持仓/自选/市场发现，LLM 深度分析（800+ 字）
- 手机触发：发邮件即可随时随地触发扫描，回复详尽研报

**技术栈**：Python 3.12 · AKShare · DeepSeek · SQLite + ChromaDB FTS5 · Docker

---

## 环境准备

需要：
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）
- QQ 邮箱，开启 SMTP + IMAP（QQ邮箱 → 设置 → 账户 → 开启服务 → 生成授权码）

> SMTP 用于发报告，IMAP 用于接收手机触发指令。两者共用同一个授权码。

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
MAIL_PASS=你的SMTP/IMAP授权码（16位）
MAIL_TO=接收报告的邮箱地址
```

> `.env` 已加入 `.gitignore`，不会被提交。

### 第三步：构建并启动

```bash
docker compose build
docker compose up -d
```

两个服务：`agent`（对话）和 `scheduler`（定时扫描 + 手机触发器）。

---

## 使用

### CLI 对话

```bash
docker compose exec -it agent python main.py
```

示例对话：

```
> 分析一下招商银行 600036
> 把工商银行加入持仓，成本 5.2 买了 10000 股
> 复盘之前对中石化的分析
```

写入操作（记录决策、修改持仓等）需在 Agent 展示内容后回复 **"确认"** 才会执行。

### 手机触发扫描

**用 QQ 邮箱手机版给自己发邮件，主题写 `扫描`**，1-2 分钟内收到深度分析报告。

不限次数，一天想扫几次扫几次。

### 手动触发

```bash
docker compose exec scheduler python scheduler.py --now
```

### 每日定时

每天 **15:30**（收盘后）自动扫描并发报告，无需任何操作。

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
├── main.py          # CLI 入口
├── agent.py         # ReAct Agent 核心循环
├── tools.py         # 16 个工具（9 读 + 7 写）+ 腾讯行情
├── adapters.py      # LLM 适配层（Claude / OpenAI / DeepSeek）
├── memory.py        # SQLite + ChromaDB + RRF 混合检索
├── scheduler.py     # 定时扫描 + 深度分析 + 手机触发器
├── mailer.py        # HTML 邮件 + IMAP 监听
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

行情数据源：腾讯 `qt.gtimg.cn`（实时）+ Tencent K-line（历史），Docker 内可用无需代理。

---

## 切换 LLM

改 `main.py` 这一行：

```python
# Claude:
llm = ClaudeAdapter(model="claude-opus-4-5")

# DeepSeek（默认）：
llm = OpenAIAdapter(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY_stock_agent"),
)
```

---

## 许可

MIT
