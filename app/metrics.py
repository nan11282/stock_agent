"""
metrics.py — 轻量埋点

Tracer 由 Agent / ToolExecutor / Adapter 共享（通过 get_tracer()）。
每次 chat() 一个 turn，turn 结束时把指标 dump 一行到 traces/YYYYMMDD.jsonl。

环境变量：
  TRACE_ENABLED=1  开启埋点（默认关闭，main.py / 测试代码显式启用）
  TRACE_DIR        traces 输出目录，默认 ./traces

CLI：
  python app/metrics.py summary [traces/YYYYMMDD.jsonl]
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    name: str
    input: dict
    step: int = 0            # 第几个 ReAct 步
    result_chars: int = 0
    latency_ms: int = 0
    cached: bool = False
    error: str | None = None


@dataclass
class LLMCallRecord:
    step: int = 0
    prompt_chars: int = 0    # system + history 序列化后总字符
    history_msg_count: int = 0
    latency_ms: int = 0
    input_tokens: int | None = None    # 若 API 返回 usage
    output_tokens: int | None = None
    output_text_chars: int = 0
    tool_calls_emitted: int = 0


@dataclass
class TurnRecord:
    turn_id: str
    started_at: str
    user_input_chars: int = 0
    system_prompt_chars: int = 0
    history_msg_count_start: int = 0
    history_total_chars_start: int = 0
    history_msg_count_end: int = 0
    history_total_chars_end: int = 0
    react_steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    final_response_chars: int = 0
    memory_writes: int = 0
    total_latency_ms: int = 0
    error: str | None = None


# ─────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────

class Tracer:
    def __init__(self, trace_dir: str | None = None, enabled: bool = False):
        self.trace_dir = trace_dir or os.environ.get("TRACE_DIR", "./traces")
        self.enabled = enabled
        self._current: TurnRecord | None = None
        self._t0_turn: float = 0.0
        self._step_counter: int = 0
        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)

    # ── Turn 边界 ──────────────────────────────

    @contextlib.contextmanager
    def turn(self, user_input: str, history_snapshot: list):
        if not self.enabled:
            yield None
            return
        rec = TurnRecord(
            turn_id=datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
            started_at=datetime.now().isoformat(),
            user_input_chars=len(user_input or ""),
            history_msg_count_start=len(history_snapshot),
            history_total_chars_start=history_chars(history_snapshot),
        )
        self._current = rec
        self._step_counter = 0
        self._t0_turn = time.monotonic()
        try:
            yield rec
        except Exception as e:
            rec.error = repr(e)
            raise
        finally:
            rec.total_latency_ms = int((time.monotonic() - self._t0_turn) * 1000)
            self._dump(rec)
            self._current = None

    def set_system_prompt(self, system: str):
        if self._current:
            self._current.system_prompt_chars = len(system or "")

    def set_history_end(self, history: list, final_text: str):
        if self._current:
            self._current.history_msg_count_end = len(history)
            self._current.history_total_chars_end = history_chars(history)
            self._current.final_response_chars = len(final_text or "")

    def bump_memory_writes(self):
        if self._current:
            self._current.memory_writes += 1

    # ── LLM 调用 ───────────────────────────────

    @contextlib.contextmanager
    def llm_call(self, system: str, history: list):
        if not self.enabled or not self._current:
            yield _NullLLMRec()
            return
        self._step_counter += 1
        self._current.react_steps = self._step_counter
        rec = LLMCallRecord(
            step=self._step_counter,
            prompt_chars=len(system or "") + history_chars(history),
            history_msg_count=len(history),
        )
        t0 = time.monotonic()
        try:
            yield rec
        finally:
            rec.latency_ms = int((time.monotonic() - t0) * 1000)
            self._current.llm_calls.append(rec)

    # ── Tool 调用 ──────────────────────────────

    @contextlib.contextmanager
    def tool_call(self, name: str, input_: dict):
        if not self.enabled or not self._current:
            yield _NullToolRec()
            return
        rec = ToolCallRecord(
            name=name,
            input=dict(input_ or {}),
            step=self._step_counter,
        )
        t0 = time.monotonic()
        try:
            yield rec
        except Exception as e:
            rec.error = repr(e)
            raise
        finally:
            rec.latency_ms = int((time.monotonic() - t0) * 1000)
            self._current.tool_calls.append(rec)

    # ── 落盘 ───────────────────────────────────

    def _dump(self, rec: TurnRecord):
        path = os.path.join(
            self.trace_dir, datetime.now().strftime("%Y%m%d") + ".jsonl"
        )
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[tracer] dump failed: {e}")


# ─────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────

def history_chars(history: list) -> int:
    """估算 history 序列化到 LLM 消息后的字符数。
    覆盖 text / tool_calls / tool_results 三类。"""
    total = 0
    for m in history:
        if getattr(m, "text", None):
            total += len(m.text)
        for tc in getattr(m, "tool_calls", None) or []:
            total += len(getattr(tc, "name", "") or "")
            try:
                total += len(json.dumps(tc.input or {}, ensure_ascii=False))
            except (TypeError, ValueError):
                pass
        for r in getattr(m, "tool_results", None) or []:
            total += len(getattr(r, "content", "") or "")
    return total


class _NullToolRec:
    """禁用埋点时返回的占位对象，所有属性写入静默丢弃。"""
    def __setattr__(self, k, v): pass


class _NullLLMRec:
    def __setattr__(self, k, v): pass


# ─────────────────────────────────────────────
# 全局实例
# ─────────────────────────────────────────────

_default_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _default_tracer
    if _default_tracer is None:
        enabled = os.environ.get("TRACE_ENABLED", "").lower() in ("1", "true", "yes")
        _default_tracer = Tracer(enabled=enabled)
    return _default_tracer


def set_tracer(tracer: Tracer):
    global _default_tracer
    _default_tracer = tracer


# ─────────────────────────────────────────────
# Console Timing
# ─────────────────────────────────────────────

def timing_log_enabled() -> bool:
    return os.environ.get("TIMING_LOG_ENABLED", "1").lower() not in (
        "0", "false", "no", "off"
    )


@contextlib.contextmanager
def console_timer(stage: str, detail: str = ""):
    """Print elapsed time to Docker stdout for live bottleneck inspection."""
    if not timing_log_enabled():
        yield
        return

    label = f"{stage} {detail}".strip()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  [耗时] {label}: {elapsed_ms:.0f} ms", flush=True)


# ─────────────────────────────────────────────
# CLI: 汇总 jsonl
# ─────────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def cli_summary(path: str):
    if not os.path.exists(path):
        print(f"找不到文件: {path}")
        return
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print("空文件")
        return

    react_steps = [r["react_steps"] for r in records]
    latencies = [r["total_latency_ms"] for r in records]
    sys_prompt = [r["system_prompt_chars"] for r in records]
    hist_start = [r["history_total_chars_start"] for r in records]
    hist_end = [r["history_total_chars_end"] for r in records]
    final_resp = [r["final_response_chars"] for r in records]

    # 每轮的 prompt_chars 取最大值（最后一步通常最大）
    max_prompts = []
    total_tool_calls = 0
    cached_tool_calls = 0
    tool_freq: dict[str, int] = {}
    for r in records:
        if r["llm_calls"]:
            max_prompts.append(max(c["prompt_chars"] for c in r["llm_calls"]))
        for tc in r["tool_calls"]:
            total_tool_calls += 1
            if tc["cached"]:
                cached_tool_calls += 1
            tool_freq[tc["name"]] = tool_freq.get(tc["name"], 0) + 1

    def stats(name: str, vals: list[float], unit: str = ""):
        if not vals:
            print(f"  {name:24s} (空)")
            return
        print(
            f"  {name:24s} "
            f"min={min(vals):>7.0f}{unit}  "
            f"p50={_percentile(vals, 0.5):>7.0f}{unit}  "
            f"p95={_percentile(vals, 0.95):>7.0f}{unit}  "
            f"max={max(vals):>7.0f}{unit}"
        )

    print(f"\n=== Trace Summary [{path}] ===")
    print(f"总 turn 数: {len(records)}\n")
    stats("react_steps", react_steps)
    stats("total_latency_ms", latencies, "ms")
    stats("system_prompt_chars", sys_prompt)
    stats("history_chars_start", hist_start)
    stats("history_chars_end", hist_end)
    stats("max_prompt_chars/turn", max_prompts)
    stats("final_response_chars", final_resp)
    print()
    print(f"工具调用总次数: {total_tool_calls}（缓存命中 {cached_tool_calls}）")
    if tool_freq:
        print("工具频次（top）:")
        for name, n in sorted(tool_freq.items(), key=lambda x: -x[1])[:10]:
            print(f"  {name:30s} {n}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "summary":
        print("用法: python app/metrics.py summary [path]")
        sys.exit(1)
    if len(sys.argv) >= 3:
        path = sys.argv[2]
    else:
        # 默认拿 traces/ 里最新的一份
        trace_dir = os.environ.get("TRACE_DIR", "./traces")
        if not os.path.isdir(trace_dir):
            print(f"找不到目录: {trace_dir}")
            sys.exit(1)
        files = sorted(
            f for f in os.listdir(trace_dir) if f.endswith(".jsonl")
        )
        if not files:
            print("traces/ 下没有 .jsonl")
            sys.exit(1)
        path = os.path.join(trace_dir, files[-1])
    cli_summary(path)


if __name__ == "__main__":
    main()
