# Readiness–Telegram Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface per-strategy live-readiness status through two Telegram surfaces: the weekly multi-strategy summary (auto), and a new `/readiness` bot command (on-demand).

**Architecture:** Add one shared helper `short_summary(report) -> str` in `src/execution/live_readiness.py`. Both surfaces import it, ensuring one source of truth. Extend `_send_multi_strategy_telegram` in `src/run_learning_cycle.py` to call `build_report` per strategy and append the status icon. Add a new `_cmd_readiness` handler in `src/execution/telegram_bot.py` that renders multi (one line per strategy) or single (full `format_text` output) views.

**Tech Stack:** Python 3.12, stdlib, pytest, existing `requests` + Telegram Bot API wiring.

**Spec:** `/opt/cts/docs/superpowers/specs/2026-04-19-readiness-telegram-integration-design.md`

---

## File Structure

- Modify: `src/execution/live_readiness.py` — add `short_summary(report)` helper.
- Modify: `tests/test_live_readiness.py` — add unit tests for `short_summary`.
- Modify: `src/run_learning_cycle.py` — extend `_send_multi_strategy_telegram` (lines 482–516).
- Modify: `src/execution/telegram_bot.py` — add `_cmd_readiness` handler, wire into `handlers` dict, update `_cmd_help`.

No new files. No schema changes.

---

## Task 1: Add `short_summary` helper with tests

**Files:**
- Modify: `src/execution/live_readiness.py`
- Modify: `tests/test_live_readiness.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_live_readiness.py`:

```python
def test_short_summary_ready():
    assert lr.short_summary({"status": "READY"}) == "🟢 READY"


def test_short_summary_watch():
    assert lr.short_summary({"status": "WATCH"}) == "🟡 WATCH"


def test_short_summary_blocked():
    assert lr.short_summary({"status": "BLOCKED"}) == "🔴 BLOCKED"


def test_short_summary_insufficient_data():
    assert lr.short_summary({"status": "INSUFFICIENT_DATA"}) == "⚪ INSUFFICIENT_DATA"


def test_short_summary_unknown_status_has_no_icon():
    # Defensive: never raise on unexpected status
    assert lr.short_summary({"status": "NEW_TIER"}) == " NEW_TIER"
```

- [ ] **Step 2: Run and verify failure**

```bash
source /opt/cts/venv/bin/activate
python -m pytest tests/test_live_readiness.py -v -k short_summary
```
Expected: all 5 tests FAIL with `AttributeError: module 'src.execution.live_readiness' has no attribute 'short_summary'`.

- [ ] **Step 3: Implement `short_summary`**

Append to `src/execution/live_readiness.py` immediately after the `format_summary` function:

```python
def short_summary(report: dict) -> str:
    """One-line summary: status icon and label.

    Example: '🟡 WATCH'. Used by the weekly Telegram summary and the
    /readiness bot command.
    """
    return f"{_STATUS_ICON.get(report['status'], '')} {report['status']}"
```

- [ ] **Step 4: Run tests, verify pass**

```bash
source /opt/cts/venv/bin/activate
python -m pytest tests/test_live_readiness.py -v
```
Expected: all 36 tests PASS (31 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add src/execution/live_readiness.py tests/test_live_readiness.py
git commit -m "feat: add short_summary helper to live_readiness"
```

---

## Task 2: Extend weekly multi-strategy Telegram summary

**Files:**
- Modify: `src/run_learning_cycle.py` (lines 482–516)

No unit tests — the existing function has none and this is a small extension with a boundary (Telegram API) that would require mocking. Manual smoke test at the end.

- [ ] **Step 1: Replace `_send_multi_strategy_telegram`**

Read `/opt/cts/src/run_learning_cycle.py` lines 482–516 to confirm the current body, then replace the function body with:

```python
def _send_multi_strategy_telegram(config: dict, strategies: list):
    """Send a combined multi-strategy comparison via Telegram."""
    tg_cfg = config.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id or "YOUR" in bot_token:
        return

    try:
        import requests
        from src.learning.parameter_store import get_performance_trend
        from src.execution.live_readiness import build_report, short_summary

        lines = ["📊 *Multi-Strategy Weekly Summary*\n"]

        status_counts = {"READY": 0, "WATCH": 0, "BLOCKED": 0, "INSUFFICIENT_DATA": 0}

        for strategy in strategies:
            trend = get_performance_trend(strategy)

            try:
                report = build_report(strategy)
                icon_label = short_summary(report)
                status_counts[report["status"]] += 1
            except Exception:
                report = None
                icon_label = ""

            if trend:
                latest = trend[-1]
                wr = latest.get("win_rate_display", "?")
                suffix = f" {icon_label.split()[0]}" if icon_label else ""
                lines.append(f"*{strategy}*: WR {wr} (v{latest['version']}){suffix}")
            elif report is not None:
                # No trend yet but readiness was computed — still show the row
                lines.append(f"*{strategy}*: {icon_label}")

        lines.append(
            f"\nReadiness: {status_counts['READY']} READY | "
            f"{status_counts['WATCH']} WATCH | "
            f"{status_counts['BLOCKED']} BLOCKED | "
            f"{status_counts['INSUFFICIENT_DATA']} INSUFFICIENT_DATA"
        )
        lines.append("\n_Run parameter\\_store --trend for full history_")

        message = "\n".join(lines)
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("  Multi-strategy summary sent.")
    except Exception as e:
        print(f"  Multi-strategy summary failed: {e}")
```

Note: `icon_label.split()[0]` extracts just the emoji (`🟡`) from `"🟡 WATCH"` for the inline row suffix, while the footer line counts full status labels.

- [ ] **Step 2: Dry-run smoke test — verify the function renders without crashing**

Run from `/opt/cts`:

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.config import load_config
from src.run_learning_cycle import _send_multi_strategy_telegram
cfg = load_config()
# Force-disable Telegram so no real message goes out; we just want to exercise the code path
cfg['notifications']['telegram']['enabled'] = False
_send_multi_strategy_telegram(cfg, ['ma_crossover', 'dbw', 'candlestick'])
print('OK — function returned without raising')
"
```
Expected: prints `OK — function returned without raising` (the function early-returns because Telegram is disabled; we've at least proven the import graph works).

- [ ] **Step 3: Dry-run smoke test — exercise the actual rendering path**

Temporarily bypass the `enabled` check by pasting the function body into a Python REPL or adding a test flag. Instead of modifying config, construct the message manually:

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.learning.parameter_store import get_performance_trend
from src.execution.live_readiness import build_report, short_summary

strategies = ['ma_crossover', 'dbw', 'candlestick']
lines = ['Multi-Strategy Weekly Summary']
status_counts = {'READY': 0, 'WATCH': 0, 'BLOCKED': 0, 'INSUFFICIENT_DATA': 0}

for strategy in strategies:
    trend = get_performance_trend(strategy)
    try:
        report = build_report(strategy)
        icon_label = short_summary(report)
        status_counts[report['status']] += 1
    except Exception as e:
        report = None
        icon_label = ''
        print(f'  {strategy} build_report failed: {e}')
    if trend:
        latest = trend[-1]
        wr = latest.get('win_rate_display', '?')
        suffix = f' {icon_label.split()[0]}' if icon_label else ''
        lines.append(f'{strategy}: WR {wr} (v{latest[\"version\"]}){suffix}')
    elif report is not None:
        lines.append(f'{strategy}: {icon_label}')

lines.append(f'Readiness: {status_counts}')
print('\n'.join(lines))
"
```
Expected: prints a block ending with a line like `Readiness: {'READY': 0, 'WATCH': 2, 'BLOCKED': 1, 'INSUFFICIENT_DATA': 0}` with each strategy on its own line including a status emoji. No exceptions.

- [ ] **Step 4: Commit**

```bash
git add src/run_learning_cycle.py
git commit -m "feat: add readiness status to weekly multi-strategy Telegram summary"
```

---

## Task 3: Add `/readiness` command to Telegram bot

**Files:**
- Modify: `src/execution/telegram_bot.py` (add handler method, register in dict, update help)

No unit tests — existing handler methods have none. Manual verification by running the bot locally against the real Telegram API.

- [ ] **Step 1: Add `_cmd_readiness` method**

Insert the following method into the `TelegramBot` class, directly after the existing `_cmd_equity` method (which ends around line 294):

```python
    def _cmd_readiness(self, args: list) -> str:
        """Show live-trading readiness per strategy."""
        from src.execution.live_readiness import (
            KNOWN_STRATEGIES, build_report, format_text, short_summary,
        )

        # Single-strategy form
        if args:
            name = args[0].lower()
            if name not in KNOWN_STRATEGIES:
                known = ", ".join(KNOWN_STRATEGIES)
                return f"Unknown strategy: {name}\nKnown: {known}"
            try:
                report = build_report(name)
            except Exception as e:
                return f"Error evaluating {name}: {type(e).__name__}"
            return f"```\n{format_text(report)}\n```"

        # Multi-strategy form
        lines = ["📊 *Live-Readiness*\n"]
        status_counts = {"READY": 0, "WATCH": 0, "BLOCKED": 0, "INSUFFICIENT_DATA": 0}

        for strategy in KNOWN_STRATEGIES:
            try:
                report = build_report(strategy)
            except Exception:
                lines.append(f"*{strategy}*: ⚠️ error")
                continue
            lines.append(f"*{strategy}*: {short_summary(report)}")
            status_counts[report["status"]] += 1

        lines.append(
            f"\n_{status_counts['READY']} READY | "
            f"{status_counts['WATCH']} WATCH | "
            f"{status_counts['BLOCKED']} BLOCKED | "
            f"{status_counts['INSUFFICIENT_DATA']} INSUFFICIENT_DATA_"
        )
        return "\n".join(lines)
```

- [ ] **Step 2: Register the handler**

Modify the `handlers` dict inside `_handle_command` (around line 112). Current code:

```python
        handlers = {
            "/positions": self._cmd_positions,
            "/pos": self._cmd_positions,
            "/equity": self._cmd_equity,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }
```

Replace with:

```python
        handlers = {
            "/positions": self._cmd_positions,
            "/pos": self._cmd_positions,
            "/equity": self._cmd_equity,
            "/readiness": self._cmd_readiness,
            "/ready": self._cmd_readiness,
            "/help": self._cmd_help,
            "/start": self._cmd_help,
        }
```

- [ ] **Step 3: Update `_cmd_help`**

Modify `_cmd_help` at line 154. Current return string:

```python
        return (
            "🤖 *CTS Trading Bot*\n\n"
            "Available commands:\n"
            "`/positions` — all open positions\n"
            "`/positions paper` — paper positions only\n"
            "`/positions live` — live positions only\n"
            "`/positions <strategy>` — filter by strategy\n"
            "`/equity` — equity summary per strategy\n"
            "`/help` — this message"
        )
```

Replace with:

```python
        return (
            "🤖 *CTS Trading Bot*\n\n"
            "Available commands:\n"
            "`/positions` — all open positions\n"
            "`/positions paper` — paper positions only\n"
            "`/positions live` — live positions only\n"
            "`/positions <strategy>` — filter by strategy\n"
            "`/equity` — equity summary per strategy\n"
            "`/readiness` — per-strategy paper→live readiness\n"
            "`/readiness <strategy>` — full gate detail for one strategy\n"
            "`/help` — this message"
        )
```

- [ ] **Step 4: Smoke test the handler — multi form**

Run from `/opt/cts` (does not connect to Telegram API; directly invokes the method):

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.execution.telegram_bot import TelegramBot
from src.config import load_config
cfg = load_config()
# Patch token so __init__ doesn't sys.exit
cfg['notifications']['telegram']['bot_token'] = 'fake:token'
cfg['notifications']['telegram']['chat_id'] = '0'
bot = TelegramBot(config=cfg)
print(bot._cmd_readiness([]))
"
```
Expected: prints a message starting with `📊 *Live-Readiness*` and containing one line per strategy, ending with a summary line. No exceptions.

- [ ] **Step 5: Smoke test the handler — single-strategy form**

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.execution.telegram_bot import TelegramBot
from src.config import load_config
cfg = load_config()
cfg['notifications']['telegram']['bot_token'] = 'fake:token'
cfg['notifications']['telegram']['chat_id'] = '0'
bot = TelegramBot(config=cfg)
print(bot._cmd_readiness(['dbw']))
"
```
Expected: prints a triple-backtick-wrapped block containing `Strategy: dbw` and multiple gate rows (the `format_text` output).

- [ ] **Step 6: Smoke test — unknown strategy**

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.execution.telegram_bot import TelegramBot
from src.config import load_config
cfg = load_config()
cfg['notifications']['telegram']['bot_token'] = 'fake:token'
cfg['notifications']['telegram']['chat_id'] = '0'
bot = TelegramBot(config=cfg)
print(bot._cmd_readiness(['nonexistent']))
"
```
Expected: prints `Unknown strategy: nonexistent\nKnown: darvas, dbw, candlestick, macd, ma_crossover, mean_reversion, sr_breakout, cvd, bb_squeeze, rsi_vwap`.

- [ ] **Step 7: Verify `/help` text**

```bash
source /opt/cts/venv/bin/activate
python -c "
from src.execution.telegram_bot import TelegramBot
from src.config import load_config
cfg = load_config()
cfg['notifications']['telegram']['bot_token'] = 'fake:token'
cfg['notifications']['telegram']['chat_id'] = '0'
bot = TelegramBot(config=cfg)
print(bot._cmd_help([]))
"
```
Expected: output contains the new `/readiness` lines.

- [ ] **Step 8: Run full test suite to confirm nothing broke**

```bash
source /opt/cts/venv/bin/activate
python -m pytest tests/ -v 2>&1 | tail -10
```
Expected: all live_readiness tests still pass (36 total); no new failures introduced elsewhere.

- [ ] **Step 9: Commit**

```bash
git add src/execution/telegram_bot.py
git commit -m "feat: add /readiness command to Telegram bot"
```

---

## Task 4: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Telegram Bot architecture section**

Find the `### Telegram Bot` subsection (currently around line 114):

```markdown
### Telegram Bot

`src/execution/telegram_bot.py` is a long-polling Telegram bot that listens for commands from the configured `chat_id`. Supported commands: `/positions`, `/positions paper`, `/positions live`, `/positions <strategy>`, `/equity`, `/help`. Run as a systemd service on the VPS.
```

Replace with:

```markdown
### Telegram Bot

`src/execution/telegram_bot.py` is a long-polling Telegram bot that listens for commands from the configured `chat_id`. Supported commands: `/positions`, `/positions paper`, `/positions live`, `/positions <strategy>`, `/equity`, `/readiness`, `/readiness <strategy>`, `/help`. Run as a systemd service on the VPS.
```

- [ ] **Step 2: Add a note to the Live-Readiness Check section**

Find the `### Live-Readiness Check` subsection (currently around line 110). After the existing paragraph, append:

```markdown

The weekly learning-cycle Telegram summary (`_send_multi_strategy_telegram` in `src/run_learning_cycle.py`) includes a per-strategy status icon and a readiness footer. The Telegram bot's `/readiness` and `/readiness <strategy>` commands expose the same data on demand.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document /readiness bot command and weekly summary integration"
```

---

## Self-Review Notes

**Spec coverage:**
- Spec §1 (`short_summary` helper) → Task 1.
- Spec §2 (extend `_send_multi_strategy_telegram`) → Task 2, including the no-trend-but-has-readiness fallback row (Task 2 Step 1 has the `elif report is not None` branch) and the readiness footer.
- Spec §3 (`/readiness` and `/ready`, zero and single-arg forms, unknown-strategy response) → Task 3 Steps 1–6.
- Spec §3 `/help` update → Task 3 Step 3.
- Spec error-handling rows: weekly omits icon on failure (Task 2, `icon_label = ""`); bot multi shows `⚠️ error` (Task 3); bot single returns `Error evaluating ...` (Task 3).
- CLAUDE.md update not in spec but consistent with the earlier live-readiness feature pattern — Task 4.

**Placeholder scan:** Every code block is full executable code. No TBDs.

**Type consistency:** `short_summary(report) -> str` is called identically in Task 2 and Task 3. Import paths match. `KNOWN_STRATEGIES`, `build_report`, `format_text` are all pre-existing names confirmed in `src/execution/live_readiness.py`.
