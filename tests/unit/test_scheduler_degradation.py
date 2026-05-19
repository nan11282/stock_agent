import importlib
import sys
import types
from contextlib import contextmanager


@contextmanager
def _noop_timer(*args, **kwargs):
    yield


_FAKE_MODULE_NAMES = [
    "schedule",
    "memory",
    "mailer",
    "tools",
    "telegram_bot",
    "runtime",
    "metrics",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _FAKE_MODULE_NAMES}

schedule_module = types.ModuleType("schedule")
schedule_module.run_pending = lambda: None
sys.modules["schedule"] = schedule_module

memory_module = types.ModuleType("memory")
memory_module.MemoryManager = object
sys.modules["memory"] = memory_module

mailer_module = types.ModuleType("mailer")
mailer_module.send_report = lambda *args, **kwargs: None
sys.modules["mailer"] = mailer_module

tools_module = types.ModuleType("tools")
tools_module.READ_TOOLS = []
tools_module.fetch_tencent_quote = lambda code: {}
tools_module.ToolExecutor = lambda memory: object()
sys.modules["tools"] = tools_module

telegram_bot_module = types.ModuleType("telegram_bot")
telegram_bot_module.start_bot = lambda: None
sys.modules["telegram_bot"] = telegram_bot_module

runtime_module = types.ModuleType("runtime")
runtime_module.build_default_llm = lambda: None
sys.modules["runtime"] = runtime_module

metrics_module = types.ModuleType("metrics")
metrics_module.console_timer = _noop_timer
sys.modules["metrics"] = metrics_module

scheduler = importlib.import_module("scheduler")
DailyScanner = scheduler.DailyScanner
sys.modules.pop("scheduler", None)

for name, module in _ORIGINAL_MODULES.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


class AlwaysFailLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools, system):
        self.calls += 1
        raise TimeoutError("upstream timed out")


def test_deep_analysis_falls_back_after_three_llm_failures(monkeypatch):
    monkeypatch.setenv("LLM_ANALYSIS_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("LLM_ANALYSIS_RETRY_DELAY_SECONDS", "0")

    scanner = object.__new__(DailyScanner)
    scanner.memory = object()
    scanner.llm = AlwaysFailLLM()

    report = scanner._deep_analysis(
        pos_results=[{
            "stock_code": "600000",
            "stock_name": "浦发银行",
            "signal": "normal",
            "summary": "现价10.0",
        }],
        watch_results=[],
        disc_results=[],
    )

    assert scanner.llm.calls == 3
    assert "[降级报告]" in report
    assert "LLM 深度分析连续失败" in report
    assert "浦发银行(600000)" in report
    assert "TimeoutError: upstream timed out" in report


def test_invalid_llm_analysis_attempts_falls_back_to_three(monkeypatch):
    monkeypatch.setenv("LLM_ANALYSIS_MAX_ATTEMPTS", "bad")

    assert scheduler._llm_analysis_max_attempts() == 3
