"""Thin wrapper over Claude: structured output, token budget, cost tracking.

Two backends (agents.backend in settings):
- "claude-cli" (default): each call runs headless `claude -p` on the Claude subscription the
  local `claude` CLI is logged in to. No API key, no $ cost recorded; the per-run output-token
  budget still applies, and calls count against the plan's usage limits. Ported from insider-bot.
- "api": the Anthropic SDK with ANTHROPIC_API_KEY (pay per token).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

PROMPT_DIR = Path(__file__).parent / "prompts"
CLI_TIMEOUT_S = 600
RESEARCH_TOOLS = "WebSearch,WebFetch"   # the only tools a research call ever gets (read-only web)


def find_claude_cli(explicit: str | None = None) -> str:
    """Path to the claude CLI: explicit setting, then PATH, then ~/.local/bin (Task Scheduler's PATH can be thin)."""
    for cand in (explicit, shutil.which("claude"),
                 str(Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude"))):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("claude CLI not found; install Claude Code or set agents.cli_path")

# $ per 1M tokens (input, output). Cached on 2026-10-06; verify against current pricing.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class BudgetExceeded(RuntimeError):
    pass


class AgentRefused(RuntimeError):
    pass


class UsageLimited(RuntimeError):
    """The Claude subscription refused the call for usage/rate limits; lower-priority steps should skip."""


_LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached", re.IGNORECASE)


def prompt_path(prompt_dir: str = "prompts") -> Path:
    """agents.prompt_dir: relative to src/bot/agents/ unless absolute."""
    p = Path(prompt_dir)
    return p if p.is_absolute() else Path(__file__).parent / p


def load_prompt(name: str, prompt_dir: str = "prompts") -> str:
    return (prompt_path(prompt_dir) / f"{name}.md").read_text(encoding="utf-8").strip()


def prompt_version(prompt_dir: str = "prompts") -> str:
    files = sorted(prompt_path(prompt_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no prompt files in {prompt_path(prompt_dir)}")
    h = hashlib.sha1()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


class LLM:
    def __init__(self, max_output_tokens: int, client: anthropic.Anthropic | None = None,
                 *, cli_path: str | None = None):
        """cli_path set (and no test client) -> subscription backend via `claude -p`."""
        self._cli_path = cli_path if client is None else None
        self._cli_cmd = [cli_path] if self._cli_path else []   # argv prefix; tests point it at a fake script
        self.client = client or (None if cli_path else anthropic.Anthropic(max_retries=4))
        self.max_output_tokens = max_output_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._lock = threading.Lock()

    def _account(self, model: str, usage) -> None:
        pin, pout = PRICES.get(model, (4.00, 20.00))
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        with self._lock:
            self.input_tokens += usage.input_tokens + cache_read + cache_write
            self.output_tokens += usage.output_tokens
            self.cost_usd += (usage.input_tokens * pin + cache_write * pin * 1.25 + cache_read * pin * 0.1
                              + usage.output_tokens * pout) / 1e6

    def structured(self, *, model: str, effort: str, system: str, user: str, schema: type[T],
                   max_tokens: int = 16000) -> T:
        if self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(f"daily output-token budget {self.max_output_tokens} exhausted")
        if self._cli_path:
            return self._structured_cli(model=model, effort=effort, system=system, user=user, schema=schema)
        resp = self.client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            # Stable system prompt first and cached; volatile market data goes in the user turn.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
            output_format=schema,
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self._account(model, resp.usage)
        if resp.stop_reason == "refusal":
            raise AgentRefused(f"{schema.__name__}: model declined ({getattr(resp, 'stop_details', None)})")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"{schema.__name__}: output truncated at max_tokens={max_tokens}")
        parsed = resp.parsed_output
        if parsed is None:
            raise RuntimeError(f"{schema.__name__}: no parsed output (stop_reason={resp.stop_reason})")
        return parsed

    def _structured_cli(self, *, model: str, effort: str, system: str, user: str,
                        schema: type[T]) -> T:
        """One headless `claude -p` call: no tools, no user settings/hooks/CLAUDE.md, no saved
        session, empty temp cwd. ANTHROPIC_* env is dropped so it bills the subscription, never
        an API key. Market data goes in on stdin. Failures raise, and the analyze run fails closed."""
        cmd = [self._cli_path, "-p", "--output-format", "json",
               "--json-schema", json.dumps(schema.model_json_schema()),
               "--model", model, "--effort", effort, "--system-prompt", system,
               "--tools", "", "--setting-sources", "", "--no-session-persistence"]
        env = self._cli_env()
        with tempfile.TemporaryDirectory(prefix="tradingbot-claude-") as cwd:
            try:
                p = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                                   timeout=CLI_TIMEOUT_S, cwd=cwd, env=env)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise RuntimeError(f"{schema.__name__}: claude CLI failed ({type(e).__name__}: {e})") from e
        try:
            d = json.loads(p.stdout)
        except ValueError:
            tail = (p.stderr or p.stdout or "").strip()[-300:]
            if _LIMIT_RE.search(tail):
                raise UsageLimited(f"{schema.__name__}: {tail}") from None
            raise RuntimeError(f"{schema.__name__}: claude CLI exit {p.returncode}: {tail}") from None
        if not isinstance(d, dict):
            raise RuntimeError(f"{schema.__name__}: claude CLI returned non-object JSON")
        return self._from_result(d, schema, p.stderr or "")

    @staticmethod
    def _cli_env() -> dict:
        return {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}

    def _from_result(self, d: dict, schema: type[T], stderr: str) -> T:
        """Account tokens and validate one CLI result object (shared by structured and research calls)."""
        u = d.get("usage") or {}
        out_tok = int(u.get("output_tokens") or 0)
        in_tok = sum(int(u.get(k) or 0) for k in
                     ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
        if d.get("stop_reason") == "refusal":
            raise AgentRefused(f"{schema.__name__}: model declined")
        if d.get("is_error") or d.get("subtype") != "success":
            if _LIMIT_RE.search(str(d.get("result")) + stderr):
                raise UsageLimited(f"{schema.__name__}: {str(d.get('result'))[:300]}")
            raise RuntimeError(f"{schema.__name__}: claude CLI error ({d.get('subtype')}): "
                               f"{str(d.get('result'))[:300]}")
        out = d.get("structured_output")
        if out is None:
            raise RuntimeError(f"{schema.__name__}: claude CLI returned no structured output")
        try:
            return schema.model_validate(out)
        except ValueError as e:
            raise RuntimeError(f"{schema.__name__}: invalid structured output ({e})") from e

    def research(self, *, model: str, effort: str, system: str, user: str, schema: type[T],
                 max_turns: int = 12, timeout_s: int = 300) -> tuple[T, list[dict]]:
        """One headless `claude -p` call that may use ONLY WebSearch and WebFetch (read-only web; no
        shell, files, MCP or settings). Streams events so every search query and fetched URL is returned
        for the journal. Fails closed: timeout, garbage, a missing result or an error result raise."""
        if self.output_tokens >= self.max_output_tokens:
            raise BudgetExceeded(f"daily output-token budget {self.max_output_tokens} exhausted")
        if not self._cli_cmd:
            raise RuntimeError("research calls need the claude-cli backend")
        cmd = [*self._cli_cmd, "-p", "--output-format", "stream-json", "--verbose",
               "--json-schema", json.dumps(schema.model_json_schema()),
               "--model", model, "--effort", effort, "--system-prompt", system,
               "--tools", RESEARCH_TOOLS, "--allowedTools", "WebSearch", "WebFetch",
               "--max-turns", str(max_turns), "--setting-sources", "", "--no-session-persistence"]
        touched: list[dict] = []
        result: dict | None = None
        lines: queue.Queue = queue.Queue()
        with tempfile.TemporaryDirectory(prefix="tradingbot-research-") as cwd:
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", cwd=cwd, env=self._cli_env())
            except OSError as e:
                raise RuntimeError(f"{schema.__name__}: claude CLI failed ({e})") from e
            errbuf: list[str] = []

            def pump_out():
                for line in p.stdout:
                    lines.put(line)
                lines.put(None)

            def feed():   # in a thread: a CLI that never reads stdin must not block the deadline
                try:
                    p.stdin.write(user)
                    p.stdin.close()
                except (OSError, ValueError):
                    pass

            deadline = time.monotonic() + timeout_s
            threading.Thread(target=pump_out, daemon=True).start()
            threading.Thread(target=lambda: errbuf.append(p.stderr.read()), daemon=True).start()
            threading.Thread(target=feed, daemon=True).start()
            try:
                while True:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise RuntimeError(f"{schema.__name__}: research call timed out after {timeout_s}s")
                    try:
                        line = lines.get(timeout=min(left, 1.0))
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(d, dict):
                        continue
                    if d.get("type") == "assistant":
                        for c in (d.get("message") or {}).get("content") or []:
                            if isinstance(c, dict) and c.get("type") == "tool_use":
                                inp = c.get("input") or {}
                                if c.get("name") == "WebFetch":
                                    touched.append({"tool": "WebFetch", "url": str(inp.get("url", ""))})
                                elif c.get("name") == "WebSearch":
                                    touched.append({"tool": "WebSearch", "query": str(inp.get("query", ""))})
                    elif d.get("type") == "result":
                        result = d
            finally:
                if p.poll() is None:
                    p.kill()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        stderr = "".join(errbuf)
        if result is None:
            if _LIMIT_RE.search(stderr):
                raise UsageLimited(f"{schema.__name__}: {stderr.strip()[-300:]}")
            raise RuntimeError(f"{schema.__name__}: research call returned no result "
                               f"(exit {p.returncode}): {stderr.strip()[-300:]}")
        return self._from_result(result, schema, stderr), touched
