# Profile B: Small-Cap Momentum & Catalyst Bot (Alpaca paper account 2)

*Spec written 2026-11-03.*

Profile B reuses bot A's pipeline (agents → views → skfolio → risk gate → Alpaca) with different
rules, aimed at **high returns from small caps**: a concentrated momentum/catalyst book that rotates
weekly, with hard per-position stops and honest (cost-adjusted) fills.

## Non-negotiables (from the plan)
1. **Profile A behavior does not change.** `config/settings.yaml` is untouched. Every new config field
   has a default that reproduces bot A exactly. All existing tests keep passing.
2. Profile B uses its **own** Alpaca keys, journal DB, reports dir, logs, kill-switch file and scheduled tasks.
3. Every screened candidate is logged, traded or not.
4. Fringe/news text never drives orders directly. The radar runs in **shadow mode**: features are
   journaled, never passed to agents.
5. Paper P&L and **realistic** P&L (modeled spread + impact) are reported side by side.
6. LLMs output views only. Deterministic code sizes and enforces every limit.

## Profile wiring
- `uv run python -m bot.run_daily --profile smallcap --phase analyze --mode dry-run`
  `--profile default` (the default) = `config/settings.yaml`; `--profile smallcap` = `config/settings_smallcap.yaml`.
  `--config <path>` still works and overrides `--profile`.
- New settings fields (defaults = bot A behavior):
  - `profile: default`
  - `broker.api_key_env: ALPACA_API_KEY`, `broker.secret_key_env: ALPACA_SECRET_KEY`
    (B: `ALPACA_B_API_KEY` / `ALPACA_B_SECRET_KEY` in `.env`)
  - `universe.source: static | screener` (default static). With `screener`: `universe.screener_dir`,
    `universe.max_age_days` (8). A missing or stale universe **blocks the run** (fail closed).
  - `agents.prompt_dir: prompts` (B: `prompts_smallcap`)
  - `execution.order_style: notional | auto` (default notional). `auto` = fractional notional market
    order when the asset is fractionable, else whole-share **marketable limit** order
    (buy limit = ask × (1 + `limit_buffer_pct`), sell limit = bid × (1 − buffer); fallback to last close
    × (1 ± 1%) if no quote), TIF day. Skip if qty rounds to 0.
  - `execution.record_quotes: false` (B: true), stage 2.
  - `risk.stop_loss_pct: null` (B: 0.15), `risk.stop_cooldown_days: 0` (B: 10), `risk.max_adv_participation: null` (B: 0.02),
    `risk.max_spread_pct: null` (B: 0.015, stage 2).
  - `radar.enabled: false` (B: true, shadow), stage 2.
- Paths for B: `data/journal_smallcap.db`, `reports/smallcap/`, `data/logs/smallcap/`,
  kill switch `KILL_SWITCH_SMALLCAP`. Bar cache dir can be shared.

## Universe: weekly screener (`scripts/screen.py`, Sunday 20:00 ET)
1. Alpaca active + tradable US equities on NYSE/NASDAQ/AMEX/ARCA/BATS. Exclude OTC, ETFs/funds/trusts,
   warrants, units, rights, preferreds (symbols containing `.` or `/`; names matching warrant/unit/right/ETF/fund/trust).
2. Filters: price ≥ $3; 20-day average dollar volume ≥ $5M; market cap $300M–$5B; ≥ 126 trading days of history.
   Use bulk price downloads, and fetch market cap only for symbols that pass price/volume. Missing fundamentals fail
   **soft** (skip the field or drop the symbol with a logged reason), never crash.
3. Score = weighted sum of cross-sectional z-scores (winsorized at ±3):
   - 0.40 momentum 12-1 (return from t−252 to t−21; use 6-1 when history < 252 days)
   - 0.20 3-month return (t−63 to t)
   - 0.20 volume surge: 5-day ADV ÷ 50-day ADV (log)
   - 0.20 proximity to 52-week high (close ÷ 252-day high)
   - 0.00 radar chatter (`mention_z`) — weight stays 0 until it passes evaluation.
4. Output the top 20 to `data/universe/smallcap/YYYY-MM-DD.json` as
   `{"as_of": "...", "generated_at": "...", "symbols": [...], "candidates": [{symbol, rank, score, features...}], "filters": {...}}`.
5. Log **every** candidate (rank, score, features, passed/failed filter + reason) to the `screen_candidates` table in
   the profile's journal.
6. Allowlist at analyze = screener top 20 ∪ currently held symbols. Held names that drop out get target 0 and are sold.

Interface (owned by the screener): `bot.screen.universe.load_universe(dir: Path, as_of: date, max_age_days: int) -> dict | None`
returns the parsed JSON (with `symbols`) for the newest file ≤ `max_age_days` old as of `as_of`, else `None`.

## Agent team (same structure, small-cap prompts in `src/bot/agents/prompts_smallcap/`)
- Horizon 10 trading days. `max_abs_view_return: 0.20`. Confidence capped at 0.8 (unchanged code).
- Style: momentum + catalysts. Typical views run from ±3% to ±12%. Beyond ±15% needs a hard, dated catalyst.
- Red flags that are bearish by default: dilution (ATM programs, S-1/S-3 shelf, convertibles, registered directs),
  reverse splits, going-concern language, promotional/paid-pump headlines, short-report allegations.
- Bullish catalysts: earnings beat plus raised guidance, contract/partnership wins with dollar figures, FDA/regulatory
  approvals, index inclusion, insider buying, breakouts to new highs on volume surges.
- Fundamentals focus: revenue growth acceleration, gross margin trend, cash runway (cash ÷ burn), debt load, share-count growth.
  P/E is often meaningless here.
- PM prompt keeps the same `{horizon} {max_view} {min_cash} {universe}` placeholders and output schema.

## Portfolio (skfolio)
`method: mean_risk`, `lookback_days: 252` (symbols with < 90% of the window are dropped by existing code, so newer listings
need about 126 days, which the screener enforces), `tau: 0.10`, `delta: 1.0` (more aggressive tilts), `l2_coef: 0.0001`,
`max_weight: 0.12`, `min_cash_weight: 0.05`. The result is about 8–12 concentrated positions.

## Risk rules for B (deterministic, in code)
| Rule | B | A |
|---|---|---|
| Max weight per position | 12% | 15% |
| Max gross | 95% | 95% |
| Max daily turnover | 50% | 40% |
| Max orders per run | 20 | 20 |
| Min order notional | $25 | $5 |
| Daily loss → kill switch | −6% | −3% |
| Drawdown from peak → kill switch | −25% | −15% |
| **Per-position stop** | unrealized P&L ≤ −15% vs avg entry → forced to target 0 at the next execute; re-entry blocked for 10 trading days | none |
| **Liquidity cap** | order notional ≤ 2% of 20-day ADV, else shrink the order to fit | none |
| **Spread guard** (stage 2) | skip *buys* when the quoted spread > 1.5% of mid | none |
| Margin, shorting | never | never |

Stops and cooldowns are journaled (`stops` table: symbol, triggered_at, entry, price, pnl_pct, cooldown_until).

## Honest fills (stage 2)
At execute, record bid/ask/mid/spread/quote time/feed for each order. At reconcile, compute for each fill: slippage vs mid,
and modeled cost = ½ spread + impact, where impact = σ_daily × √(order_shares ÷ ADV_shares) (square-root law, coefficient 1.0).
The report and email show **paper P&L** and **realistic P&L** side by side. Strategy decisions use the realistic number.

## News radar (shadow, `src/bot/radar/`)
Collector every 30 min → `data/radar.db` (`url, source, symbol, text, ts`, dedupe by URL and text hash).
Sources: HN Algolia, GDELT DOC API, Google News RSS (mainstream baseline), Substack RSS (configurable feeds),
Bluesky search (optional, needs `BSKY_HANDLE`/`BSKY_APP_PASSWORD`), YouTube transcripts (optional, off by default).
Every source fails soft. Ticker matching uses cashtags plus company names for the current universe (A + B).
Daily features frozen at analyze time (only posts with ts ≤ freeze time): `mention_z` (vs 30-day baseline), `n_sources`,
`fringe_only` (3+ independent non-mainstream sources and no Google News hit), and an optional ≤ 200-char summary.
Post text is untrusted: strip it and cap its length before any LLM sees it.

## Schedule (Task Scheduler, weekdays unless noted)
| Task | ET |
|---|---|
| TradingBotB-Screen | Sunday 20:00 |
| TradingBotB-Analyze | 17:30 |
| TradingBotB-Execute | 10:00 (after the opening auction volatility) |
| TradingBotB-Reconcile | 16:20 |
| TradingBot-Radar | every 30 min (all days) |

## Hypotheses (freeze before starting the clock)
- H1: Screener top-ranked beat bottom-ranked over 10 days (rank IC > 0, t > 2).
- H2: `mention_z` predicts 10-day returns in small caps but not large caps.
- H3: `fringe_only` stories precede larger moves or higher volatility.
- Success bar: ______ · Evaluation date: ______
