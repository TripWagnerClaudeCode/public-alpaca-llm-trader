# alpaca-llm-trader

A daily swing-trading bot for US stocks and ETFs. Claude agents form market views. A portfolio optimizer turns those views into position sizes, and plain code enforces every risk limit. Orders go to [Alpaca](https://alpaca.markets), on a **paper** account by default.

> **Not financial advice.** This is an experiment. It has no track record. Run it on paper, and if you ever trade real money with it, only use money you can afford to lose.

```
market data ──► agent team ──────────────────────────────► optimizer ──► risk gate ──► Alpaca
(prices, news,   Regime · Technical · Fundamentals · News     (skfolio,     (plain code,   (paper by
 fundamentals,   → Bull vs Bear → Portfolio Manager           Black-        kill switch)   default)
 SEC filings)    → views: expected return + confidence        Litterman)
                                    │
                       SQLite journal + daily Markdown report
```

**The LLMs never place orders or choose position sizes.** They output views only. The optimizer sizes the positions, and deterministic code checks every limit before an order is sent.

## Two strategies (profiles)

| Profile | Config | What it trades |
|---|---|---|
| **A: large-cap** (`--profile default`) | `config/settings.yaml` | ~25 liquid large caps and sector ETFs, daily rebalance toward the optimizer's targets |
| **B: small-cap** (`--profile smallcap`) | `config/settings_smallcap.yaml` | A weekly screened small-cap momentum/catalyst book with hard per-position stops. See [docs/PROFILE_B_SMALLCAP.md](docs/PROFILE_B_SMALLCAP.md) |

Each profile uses its own Alpaca keys, journal, reports and kill switch.

Each profile also has a **frozen control** (`control`, `smallcap_control`): the same strategy with the original prompts and all upgrades off, run on a simulated book. Comparing the live profile with its control shows whether the upgrades actually help.

## Features

All of these can be switched off under `features:` in the settings.

- **Memory:** outcomes are scored, and the agents get a short block of lessons and scorecards. A weekly Curator keeps the lessons. You can edit `memory/<profile>/lessons.md` yourself. Your edits and a `## Pinned by you` section always win over the bot.
- **Research:** up to 3 cited deep dives per run (web search plus SEC filings) before the Portfolio Manager decides.
- **Reviewers:** a second PM and a Risk Reviewer can cut or drop trades. They can never add or enlarge one.
- **Intraday watch:** every 30 minutes the bot refreshes orders and stops. An Event Review can vote to exit a held name; it needs two agents to agree. Intraday buys need 2 approvals and still go through the optimizer and the caps.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- An Alpaca **paper** account (free). Profile B needs a second paper account.
- For the agents, one of:
  - **Claude Code CLI** (`agents.backend: claude-cli`, the default). It runs headless `claude -p` on your Claude subscription. No API key is needed, but runs count toward your plan's usage limits.
  - **Anthropic API** (`agents.backend: api` plus `ANTHROPIC_API_KEY`). You pay per token. `agents.max_daily_output_tokens` caps the spend per run.

## Setup

```bash
git clone https://github.com/<owner>/public-alpaca-llm-trader.git alpaca-llm-trader
cd alpaca-llm-trader
uv sync
cp .env.example .env      # then fill it in (see below)
```

In `.env`:

| Variable | Needed for |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Profile A (paper keys) |
| `ALPACA_B_API_KEY`, `ALPACA_B_SECRET_KEY` | Profile B (a second paper account) |
| `SEC_USER_AGENT` | SEC filing lookups. The SEC requires `Your Name your@email`. Without it, research runs without filings. |
| `ANTHROPIC_API_KEY` | Only with `agents.backend: api` |
| `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO` | Optional email summary after each run |
| `ALLOW_LIVE_TRADING` | Leave it at `no` (see below) |

## Run it

```bash
# Full pipeline with no broker and no orders (simulated $100k account)
uv run python -m bot.run_daily --phase all --mode dry-run

# Quant baseline only, with no LLM calls
uv run python -m bot.run_daily --phase all --mode dry-run --no-agents

# Paper trading: analyze after the close, execute after the next open, reconcile at the close
uv run python -m bot.run_daily --phase analyze   --mode paper
uv run python -m bot.run_daily --phase execute   --mode paper    # --approve to confirm each order batch
uv run python -m bot.run_daily --phase reconcile --mode paper
uv run python -m bot.run_daily --phase watch     --mode paper    # intraday, every 30 min

# Profile B: refresh the screened universe weekly, then the same phases
uv run python scripts/screen.py --profile smallcap
uv run python -m bot.run_daily --profile smallcap --phase analyze --mode paper
```

Each run writes a Markdown report under `reports/` and logs everything to `data/journal.db`: inputs, every agent's output, views, targets, orders, fills and token cost.

### Scheduling

- **Windows:** `.\scripts\register_tasks.ps1 -Mode paper` (add `-Profile smallcap` for profile B). It registers weekday Task Scheduler jobs, with times converted from US Eastern.
  - analyze 17:15
  - execute 09:45
  - reconcile 16:15
  - watch every 30 min
- **Linux/macOS:** run the same commands from cron or systemd timers at those Eastern times.

## Safety rails

These are enforced in code (`config/settings.yaml`, under `risk:`):

| Limit | Default |
|---|---|
| Max weight per position | 15% |
| Max invested (gross) | 95% |
| Max orders per run | 20 (the rest roll to later days) |
| Max daily turnover | 40% |
| Daily loss vs previous close | −3% → **kill switch** |
| Drawdown from peak equity | −15% → **kill switch** |
| Stale data, broker trading block, closed market | no trades that run |
| Margin, shorting | never |

**Kill switch:** create a file named `KILL_SWITCH` in the repo root, or set `BOT_KILL_SWITCH=yes`, and nothing trades. The bot writes this file itself when a loss limit trips. Delete it to resume.

**Live trading is off by default.** It requires `execution.mode: live` (or `--mode live`) **and** `ALLOW_LIVE_TRADING=yes`. Live runs also ask you to type `yes` before submitting, unless `execution.live_auto_execute: true`. Don't enable it until you have weeks of paper results you trust.

## Judging it honestly

- `uv run python scripts/backtest_quant.py` runs a walk-forward backtest of SPY vs equal weight vs HRP (the no-view fallback). The universe was chosen recently, so expect survivorship bias.
- `uv run python scripts/eval_kronos.py` checks the optional Kronos forecaster (in `vendor/Kronos`, MIT). It is off by default because it showed no skill on this universe.
- `uv run python scripts/pnl_report.py --profile smallcap` shows daily paper P&L next to a "realistic" P&L (minus modeled spread and market impact) and the profile's benchmark.
- **The agents can only be judged by forward paper trading.** LLMs already know past prices, so a backtest of them would be meaningless. Paper trade for 6–8 weeks and compare against SPY and the `--no-agents` baseline, net of LLM cost.

## Layout

```
config/                 settings per profile (+ frozen controls), radar sources
src/bot/data/           market data + technical features
src/bot/agents/         prompts, schemas, LLM wrapper (CLI or API), team orchestration
src/bot/research/       SEC EDGAR helper for the Researcher
src/bot/memory/         outcome scoring, scorecards, lessons, Curator
src/bot/portfolio/      skfolio optimizer, order planner
src/bot/risk/gate.py    hard limits + kill switch
src/bot/broker/         Broker protocol, Alpaca, dry-run simulator
src/bot/journal/        SQLite journal + Markdown reports
src/bot/run_daily.py    entry point
scripts/                screener, backtest, evals, P&L report, Task Scheduler setup
vendor/Kronos/          optional forecaster (MIT, pinned commit in vendor/KRONOS_COMMIT)
```

## Tests

```bash
uv run --all-groups python -m pytest -q
```

The tests never reach the network, the Claude CLI, the Anthropic API or a broker.

## License

MIT, see [LICENSE](LICENSE). `vendor/Kronos` keeps its own MIT license.
