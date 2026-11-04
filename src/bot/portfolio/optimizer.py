"""Turn views into target weights with skfolio.

Prior: equilibrium returns (market-neutral starting point) + shrunk covariance.
Views: Black-Litterman absolute views, one per symbol, with Idzorek confidences.
Optimizer: MeanRisk maximizing utility, long-only, weight caps, cash buffer.
Fallback/baseline: Hierarchical Risk Parity (ignores views).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from skfolio import RiskMeasure
from skfolio.moments import EquilibriumMu, ShrunkCovariance
from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction
from skfolio.prior import BlackLitterman, EmpiricalPrior

from bot.config import PortfolioCfg
from bot.models import View

log = logging.getLogger(__name__)


def horizon_to_daily(r: float, horizon_days: int) -> float:
    return (1.0 + r) ** (1.0 / horizon_days) - 1.0


def views_to_skfolio(views: list[View], columns: list[str], horizon_days: int) -> tuple[list[str], list[float]]:
    """Validated conversion to skfolio view strings. Unknown symbols / zero-confidence views are dropped."""
    cols = set(columns)
    strings, confs = [], []
    seen = set()
    for v in views:
        if v.symbol not in cols or v.symbol in seen or v.confidence <= 0:
            continue
        seen.add(v.symbol)
        daily = horizon_to_daily(v.expected_return, horizon_days)
        strings.append(f"{v.symbol} == {daily:.8f}")
        # Idzorek confidences must lie strictly inside (0, 1)
        confs.append(float(np.clip(v.confidence, 0.01, 0.99)))
    return strings, confs


def _risk_measure(name: str) -> RiskMeasure:
    return {"variance": RiskMeasure.VARIANCE, "cvar": RiskMeasure.CVAR}[name]


def hrp_weights(returns: pd.DataFrame, cfg: PortfolioCfg) -> pd.Series:
    # HRP must allocate 100% before cash scaling; with a tiny universe the cap may be infeasible.
    # Loosen it just enough here. If that pushes a weight over the risk limit, the risk gate blocks.
    max_w = max(cfg.max_weight, 1.0 / returns.shape[1] + 1e-6)
    model = HierarchicalRiskParity(risk_measure=_risk_measure(cfg.risk_measure), max_weights=max_w)
    model.fit(returns)
    w = pd.Series(model.weights_, index=returns.columns)
    return w * (1.0 - cfg.min_cash_weight)


def bl_weights(returns: pd.DataFrame, views: list[View], cfg: PortfolioCfg, horizon_days: int,
               delta: float = 2.5) -> pd.Series:
    """`delta` is the equilibrium risk aversion. skfolio's utility is mu - lambda*variance while
    equilibrium mu = delta*Sigma*w, so lambda = delta/2 makes the no-view optimum the neutral
    (equal-weight) portfolio. Views then tilt away from that neutral starting point."""
    strings, confs = views_to_skfolio(views, list(returns.columns), horizon_days)
    if not strings:
        log.info("No usable views; falling back to HRP")
        return hrp_weights(returns, cfg)
    prior = BlackLitterman(
        views=strings,
        view_confidences=confs,
        tau=cfg.tau,
        prior_estimator=EmpiricalPrior(
            mu_estimator=EquilibriumMu(risk_aversion=delta),
            covariance_estimator=ShrunkCovariance(),
        ),
    )
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=_risk_measure(cfg.risk_measure),
        risk_aversion=delta / 2.0,
        prior_estimator=prior,
        min_weights=0.0,
        max_weights=cfg.max_weight,
        budget=None,
        min_budget=0.0,
        max_budget=1.0 - cfg.min_cash_weight,
        l2_coef=cfg.l2_coef,  # diversification penalty; stops one view from spilling into all correlated names
    )
    try:
        model.fit(returns)
    except Exception as e:
        log.warning("Black-Litterman optimization failed (%s); falling back to HRP", e)
        return hrp_weights(returns, cfg)
    return pd.Series(model.weights_, index=returns.columns).clip(lower=0.0)


def target_weights(closes: pd.DataFrame, views: list[View], cfg: PortfolioCfg, horizon_days: int,
                   exclude: tuple[str, ...] = ()) -> pd.Series:
    """Main entry point. `closes` = daily close prices (dates x symbols)."""
    px = closes.drop(columns=[c for c in exclude if c in closes.columns]).tail(cfg.lookback_days + 1)
    px = px.dropna(axis=1, thresh=int(len(px) * 0.9)).ffill().dropna()
    returns = px.pct_change().dropna()
    if cfg.method == "hrp":
        w = hrp_weights(returns, cfg)
    else:
        w = bl_weights(returns, views, cfg, horizon_days, delta=cfg.delta)
    w = w.where(w > 1e-4, 0.0)  # drop dust
    return w.round(6)
