"""Email summaries after each run (SMTP; Gmail by default).

Configure in .env:
  SMTP_USER=you@gmail.com
  SMTP_PASSWORD=<16-char Gmail app password>   (myaccount.google.com/apppasswords; needs 2-Step Verification)
  EMAIL_TO=you@gmail.com                         (optional, defaults to SMTP_USER)
  SMTP_HOST=smtp.gmail.com  SMTP_PORT=465        (optional)

Email is best-effort: a failed send is logged and never affects trading.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from bot.config import secret
from bot.journal.store import Journal

log = logging.getLogger(__name__)


def email_configured() -> bool:
    return bool(secret("SMTP_USER", required=False) and secret("SMTP_PASSWORD", required=False))


def send_email(subject: str, body: str) -> bool:
    if not email_configured():
        log.info("Email not configured (SMTP_USER/SMTP_PASSWORD missing); skipping summary")
        return False
    user = secret("SMTP_USER")
    to = secret("EMAIL_TO", required=False) or user
    host = secret("SMTP_HOST", required=False) or "smtp.gmail.com"
    port = int(secret("SMTP_PORT", required=False) or 465)
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as smtp:
            smtp.login(user, secret("SMTP_PASSWORD"))
            smtp.send_message(msg)
        log.info("Summary emailed to %s", to)
        return True
    except Exception as e:
        log.warning("Email send failed: %s", e)
        return False


def summarize_runs(journal: Journal, after_run_id: int, mode: str, phase: str,
                   kill_switch_on: bool, realistic_pnl: bool = False) -> tuple[str, str] | None:
    """Build (subject, body) for all journal runs created after `after_run_id`.
    realistic_pnl (profiles with execution.record_quotes): add paper vs realistic P&L lines."""
    runs = journal.query("SELECT * FROM runs WHERE id > ? ORDER BY id", (after_run_id,))
    eq = journal.query("SELECT * FROM equity ORDER BY date DESC LIMIT 2")
    if runs.empty and phase != "reconcile":
        return None
    worst = "ok"
    for st in runs["status"] if not runs.empty else []:
        if st == "error":
            worst = "error"
        elif st == "blocked" and worst != "error":
            worst = "blocked"
    tag = {"ok": "OK", "blocked": "BLOCKED", "error": "ERROR"}[worst]
    if kill_switch_on:
        tag = "KILL SWITCH ON"

    L = []
    if kill_switch_on:
        L += ["*** KILL SWITCH IS ACTIVE: the bot will not trade until the KILL_SWITCH file is deleted. ***", ""]
    for _, r in runs.iterrows():
        L.append(f"[{r['phase']}] run {r['id']} status={r['status']} mode={r['mode']}")
        if r["notes"]:
            L.append(f"  notes: {r['notes']}")
        if r["est_cost_usd"]:
            L.append(f"  AI cost: ${r['est_cost_usd']:.2f}")
        if r["phase"] == "analyze" and r["status"] == "ok":
            tw = journal.query("SELECT symbol, weight FROM target_weights WHERE run_id=? AND weight>0 "
                               "ORDER BY weight DESC", (int(r["id"]),))
            if not tw.empty:
                L.append(f"  targets ({tw['weight'].sum():.0%} invested): " +
                         ", ".join(f"{s} {w:.1%}" for s, w in zip(tw["symbol"], tw["weight"])))
            vw = journal.query("SELECT symbol, expected_return, confidence, rationale FROM views WHERE run_id=? "
                               "ORDER BY expected_return DESC", (int(r["id"]),))
            for _, v in vw.iterrows():
                L.append(f"    view {v['symbol']:<5} {v['expected_return']:+.2%} conf {v['confidence']:.0%}: {v['rationale']}")
        od = journal.query("SELECT symbol, side, notional, status, error FROM orders WHERE run_id=?", (int(r["id"]),))
        for _, o in od.iterrows():
            err = f"  ({o['error'][:120]})" if o["error"] else ""
            L.append(f"    {o['side'].upper():<4} {o['symbol']:<5} ${o['notional']:>10,.2f}  {o['status']}{err}")
        L.append("")
    if not eq.empty:
        e0 = eq.iloc[0]
        line = f"Equity {e0['date']}: ${e0['equity']:,.2f} (cash ${e0['cash']:,.2f})"
        if len(eq) > 1 and eq.iloc[1]["equity"]:
            line += f", {e0['equity'] / eq.iloc[1]['equity'] - 1:+.2%} vs {eq.iloc[1]['date']}"
        L.append(line)
    if realistic_pnl:
        from bot.journal.store import format_pnl

        p = journal.pnl_summary()
        if p is not None:
            L += format_pnl(p)
    subject = f"[trading-bot {mode}] {phase} {tag}"
    return subject, "\n".join(L).strip() + "\n\nDetails: reports\\ and data\\logs\\ in the trading-bot folder.\n"
