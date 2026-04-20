# A股投资助理

AI 驱动的 A 股投资助手。通过对话分析个股行情、管理持仓与自选股，每日收盘后自动发送邮件报告。

**技术栈**：Python 3.12 · AKShare · DeepSeek / Claude · SQLite + ChromaDB · Docker

---

## 环境准备

**需要准备：**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）
- QQ 邮箱 SMTP 授权码（QQ邮箱 → 设置 → 账户 → 开启SMTP → 生成授权码）

---

## 配置

**第一步：克隆项目**

```bash
git clone <repo-url>
cd stock_agent
```

**第二步：配置环境变量**

项目通过 `.env` 文件把变量注入 Docker 容器，有两种方式二选一：

---

### 方式一：`.env` 文件（推荐，最简单）

在项目根目录新建 `.env`，填入以下内容（参考 `.env.example`）：

```env
DEEPSEEK_API_KEY_stock_agent=sk-xxxxxxxxxxxxxxxx

MAIL_USER=你的QQ邮箱@qq.com
MAIL_PASS=你的SMTP授权码（16位字母，不是QQ密码）
MAIL_TO=接收报告的邮箱地址
```

`docker compose up` 会自动读取同目录的 `.env`，无需额外操作。

> `.env` 已加入 `.gitignore`，不会被提交到 git。

---

### 方式二：系统环境变量（多项目共用 Key 时更方便）

在 **PowerShell** 里设置（当前会话有效）：

```powershell
$env:DEEPSEEK_API_KEY_stock_agent = "sk-xxxxxxxxxxxxxxxx"
$env:MAIL_USER = "你的QQ邮箱@qq.com"
$env:MAIL_PASS = "你的SMTP授权码"
$env:MAIL_TO   = "接收报告的邮箱地址"
```

若要**永久生效**（重启终端后仍有效），改用：

```powershell
[System.Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY_stock_agent", "sk-xxxxxxxxxxxxxxxx", "User")
[System.Environment]::SetEnvironmentVariable("MAIL_USER", "你的QQ邮箱@qq.com", "User")
[System.Environment]::SetEnvironmentVariable("MAIL_PASS", "你的SMTP授权码", "User")
[System.Environment]::SetEnvironmentVariable("MAIL_TO",   "接收报告的邮箱地址", "User")
```

设置后**重新打开 PowerShell**，`docker compose up` 会通过 `docker-compose.yml` 里的 `${VAR}` 语法自动将系统变量传入容器。

---

### 验证变量是否正确传入容器

```powershell
docker compose exec scheduler env | findstr DEEPSEEK
docker compose exec scheduler env | findstr MAIL
```

有输出即表示变量已生效。

---

## 启动

```bash
# 首次构建（仅需一次）
docker compose build

# 启动服务
docker compose up -d
```

---

## 使用

**进入对话界面：**

```bash
docker compose exec -it agent python main.py
```

然后直接用中文对话：

```
> 分析一下招商银行 600036
> 把贵州茅台 600519 加入自选股
> 查看我的持仓
```

> 写入操作（记录决策、加仓、删除等）需在 Agent 展示内容后回复**"确认"**才会执行。

**手动触发今日报告：**

```bash
docker compose stop agent
docker compose run --rm scheduler python scheduler.py --now
docker compose start agent
```

每天 15:30 收盘后 scheduler 会自动扫描并发送邮件，无需手动触发。
