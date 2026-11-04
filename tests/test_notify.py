import datetime as dt

import pandas as pd

from bot import notify
from bot.journal.store import Journal
from bot.models import OrderPlan, SubmittedOrder, View


def _journal_with_runs(tmp_path):
    j = Journal(tmp_path / "j.db")
    a = j.start_run("analyze", "paper")
    j.views(a, [View(symbol="MA", expected_return=0.018, confidence=0.55, rationale="beat and raise")])
    j.target_weights(a, pd.Series({"MA": 0.15, "V": 0.0}))
    j.finish_run(a, "ok", est_cost_usd=0.56)
    e = j.start_run("execute", "paper")
    plan = OrderPlan(symbol="MA", side="buy", notional=6000, current_weight=0, target_weight=0.06)
    j.order(e, plan, SubmittedOrder(symbol="MA", side="buy", notional=6000, broker_order_id="x1", status="accepted"))
    j.finish_run(e, "blocked", "market closed")
    j.record_equity(dt.date(2026, 11, 2), 100_500, 94_000, None)
    j.record_equity(dt.date(2026, 10, 30), 100_000, 100_000, None)
    return j


def test_summary_contents(tmp_path):
    j = _journal_with_runs(tmp_path)
    subject, body = notify.summarize_runs(j, 0, "paper", "all", kill_switch_on=False)
    assert subject == "[trading-bot paper] all BLOCKED"
    assert "MA 15.0%" in body and "V 0.0%" not in body
    assert "view MA" in body and "+1.80%" in body
    assert "BUY  MA" in body and "$  6,000.00" in body
    assert "market closed" in body
    assert "+0.50% vs 2026-10-30" in body
    assert "AI cost: $0.56" in body


def test_summary_kill_switch_flag(tmp_path):
    j = _journal_with_runs(tmp_path)
    subject, body = notify.summarize_runs(j, 0, "paper", "execute", kill_switch_on=True)
    assert "KILL SWITCH ON" in subject and "KILL SWITCH IS ACTIVE" in body


def test_only_new_runs(tmp_path):
    j = _journal_with_runs(tmp_path)
    _, body = notify.summarize_runs(j, 1, "paper", "execute", kill_switch_on=False)
    assert "[analyze]" not in body and "[execute]" in body


def test_send_skips_without_credentials(monkeypatch):
    monkeypatch.setattr(notify, "secret", lambda name, required=True: "")
    assert notify.send_email("s", "b") is False


def test_send_uses_smtp_and_never_raises(monkeypatch):
    env = {"SMTP_USER": "me@gmail.com", "SMTP_PASSWORD": "app-pw", "EMAIL_TO": "", "SMTP_HOST": "", "SMTP_PORT": ""}
    monkeypatch.setattr(notify, "secret", lambda name, required=True: env.get(name, ""))
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, context=None, timeout=None):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, u, p):
            sent["login"] = (u, p)

        def send_message(self, msg):
            sent["to"], sent["subject"] = msg["To"], msg["Subject"]

    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", FakeSMTP)
    assert notify.send_email("hello", "body") is True
    assert sent == {"host": "smtp.gmail.com", "port": 465, "login": ("me@gmail.com", "app-pw"),
                    "to": "me@gmail.com", "subject": "hello"}

    class BrokenSMTP(FakeSMTP):
        def login(self, u, p):
            raise OSError("auth failed")

    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", BrokenSMTP)
    assert notify.send_email("hello", "body") is False
