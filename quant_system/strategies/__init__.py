"""Strategy package: regime-routed alpha and capital-preservation modules."""

from __future__ import annotations

from quant_system.strategies.adaptive_momentum import AdaptiveMomentumStrategy
from quant_system.strategies.base import (
    BaseStrategy,
    PositionSnapshot,
    Signal,
    StrategyContext,
)
from quant_system.strategies.risk_preservation import (
    PreservationAction,
    RiskPreservationStrategy,
)
from quant_system.strategies.stat_arb import PairsStatArbStrategy

__all__: list[str] = [
    "BaseStrategy",
    "Signal",
    "StrategyContext",
    "PositionSnapshot",
    "PairsStatArbStrategy",
    "AdaptiveMomentumStrategy",
    "RiskPreservationStrategy",
    "PreservationAction",
]
