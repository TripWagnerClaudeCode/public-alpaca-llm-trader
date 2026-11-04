"""LLM.research: tool-enabled headless claude call (WebSearch/WebFetch only), every source journaled."""
import json
import sys
import textwrap
import time

import pytest
from pydantic import BaseModel

from bot.agents.llm import LLM, RESEARCH_TOOLS, UsageLimited


class Out(BaseModel):
    answer: str


def _fake(tmp_path, body: str):
    p = tmp_path / "fake_claude.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    llm = LLM(100_000, cli_path="unused")
    llm._cli_cmd = [sys.executable, str(p)]
    return llm


STREAM = """
import json, sys
sys.stdin.read()
open(sys.argv[0] + ".argv", "w").write(json.dumps(sys.argv[1:]))
def emit(d): print(json.dumps(d), flush=True)
emit({"type": "system", "subtype": "init"})
print("not json at all", flush=True)
emit({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "WebSearch", "input": {"query": "acme 8-K"}},
    {"type": "tool_use", "name": "WebFetch", "input": {"url": "https://www.sec.gov/x.htm", "prompt": "p"}}]}})
emit({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "StructuredOutput", "input": {"answer": "yes"}}]}})
emit({"type": "result", "subtype": "success", "is_error": False, "structured_output": {"answer": "yes"},
      "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 42}})
"""


def _call(llm, **kw):
    return llm.research(model="claude-sonnet-5-5", effort="medium", system="s", user="u", schema=Out, **kw)


def test_research_returns_output_and_touched(tmp_path):
    llm = _fake(tmp_path, STREAM)
    out, touched = _call(llm)
    assert out.answer == "yes"
    assert touched == [{"tool": "WebSearch", "query": "acme 8-K"},
                       {"tool": "WebFetch", "url": "https://www.sec.gov/x.htm"}]
    assert llm.output_tokens == 42 and llm.input_tokens == 15


def test_research_cmd_has_only_web_tools(tmp_path):
    llm = _fake(tmp_path, STREAM)
    _call(llm, max_turns=7)
    argv = json.loads((tmp_path / "fake_claude.py.argv").read_text())
    assert argv[argv.index("--tools") + 1] == RESEARCH_TOOLS == "WebSearch,WebFetch"
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in argv
    assert not any("Bash" in a or "Write" in a or "Edit" in a for a in argv)


def test_research_timeout_and_garbage(tmp_path):
    slow = _fake(tmp_path, "import time, sys\nsys.stdin.read()\nprint('junk', flush=True)\ntime.sleep(30)\n")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        _call(slow, timeout_s=2)
    assert time.time() - t0 < 10
    noresult = _fake(tmp_path, "import sys\nsys.stdin.read()\nprint('{\"type\": \"system\"}')\n")
    with pytest.raises(RuntimeError, match="no result"):
        _call(noresult)


def test_research_usage_limit(tmp_path):
    llm = _fake(tmp_path, """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True,
                          "result": "Claude usage limit reached", "usage": {}}))
    """)
    with pytest.raises(UsageLimited):
        _call(llm)


def test_research_error_result_raises(tmp_path):
    llm = _fake(tmp_path, """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "", "usage": {}}))
    """)
    with pytest.raises(RuntimeError, match="error_max_turns"):
        _call(llm)


def test_research_refuses_real_cli_in_tests():
    llm = LLM(100_000, cli_path="C:/somewhere/claude.exe")
    with pytest.raises(RuntimeError, match="disabled in tests"):
        _call(llm)
