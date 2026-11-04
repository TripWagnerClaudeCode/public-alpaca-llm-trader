"""News/chatter radar (shadow mode).

Collector (every 30 min) -> data/radar.db -> daily per-symbol features frozen at analyze time.
Features are journaled only; they never reach the agents or orders (spec: PROFILE_B_SMALLCAP.md).
No LLM is called anywhere in this package (numeric features only); see features.summary_hook.
"""
