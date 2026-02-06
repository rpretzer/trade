"""
Regime-scenario backtests
Synthetic 20-business-day price slices that mimic three distinct market
regimes are fed through the full backtest engine (realistic execution,
transaction costs, risk management).

Scenarios
---------
COVID crash   – 4 stable days, 6-day flash crash, 10-day recovery.
                Model stays bullish → crash-segment trades are losers.
                Assertions focus on circuit-breaker / drawdown behaviour.
2021 bull     – Steady TICKER_LONG outperformance; model is correct.
                Assertions verify positive returns and no drawdown breach.
2022 bear     – TICKER_LONG underperforms; model is wrong (bullish).
                Assertions verify losses are realised and stay bounded.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest_strategy import backtest_strategy
from constants import TICKER_LONG, TICKER_SHORT


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _price_frame(long_prices, short_prices, start, freq="B"):
    """Build a price DataFrame with the columns backtest_strategy expects."""
    long_arr  = np.asarray(long_prices,  dtype=float)
    short_arr = np.asarray(short_prices, dtype=float)
    return pd.DataFrame(
        {
            TICKER_LONG:        long_arr,
            TICKER_SHORT:       short_arr,
            "Price_Difference": long_arr - short_arr,
        },
        index=pd.date_range(start=start, periods=len(long_arr), freq=freq),
    )


# ---------------------------------------------------------------------------
# Synthetic price data — 20 business-day windows
# ---------------------------------------------------------------------------

# --- COVID crash (Feb–Mar 2020) -----------------------------------------
# 4 stable → 6-day crash (−35 %) → 10-day partial recovery.
# Model stayed bullish throughout, so every crash-segment trade is a loser.
_COVID_LONG = (
    [320.0] * 4
    + [290, 260, 230, 200, 180, 165]       # crash
    + [155, 150, 148, 152, 158, 165, 172, 180, 188, 195]  # recovery
)
_COVID_SHORT = (
    [190.0] * 4
    + [172, 153, 135, 118, 105, 96]        # crash
    + [90, 87, 85, 88, 92, 97, 102, 108, 113, 118]        # recovery
)
_COVID_PREDS = np.array([0.0] + [0.8] * 19)   # bullish throughout

# --- 2021 bull ----------------------------------------------------------
# TICKER_LONG outperforms → Price_Difference trends upward.
# Model correctly predicts the positive trend.
_BULL_LONG  = np.linspace(120, 170, 20)   # +50 over the window
_BULL_SHORT = np.linspace(260, 255, 20)   # slight drift down
_BULL_PREDS = np.array([0.0] + [0.8] * 19)

# --- 2022 bear ----------------------------------------------------------
# TICKER_LONG underperforms → Price_Difference trends downward.
# Model is bullish (wrong), so every trade is a loser.
_BEAR_LONG  = np.linspace(175, 110, 20)   # −65 over the window
_BEAR_SHORT = np.linspace(310, 295, 20)   # modest decline
_BEAR_PREDS = np.array([0.0] + [0.8] * 19)


# ---------------------------------------------------------------------------
# COVID-crash regime
# ---------------------------------------------------------------------------


class TestCOVIDCrash:
    """Circuit-breaker and drawdown guards must activate during a flash crash."""

    @pytest.fixture
    def results(self):
        np.random.seed(0)
        price_data = _price_frame(_COVID_LONG, _COVID_SHORT, start="2020-02-17")
        return backtest_strategy(
            _COVID_PREDS,
            price_data["Price_Difference"].values,
            price_data,
            threshold=0.5,
            initial_capital=10_000,
            max_drawdown=0.01,  # tight 1 % limit — crash losses exceed it fast
        )

    def test_drawdown_guard_fires(self, results):
        """The 1 % drawdown circuit breaker must trip during the crash."""
        assert results["drawdown_limit_reached"], (
            "Drawdown guard did not fire — crash losses were smaller than expected"
        )

    def test_max_drawdown_exceeds_threshold(self, results):
        """Recorded max drawdown must be at least −1 %."""
        assert results["max_drawdown_pct"] < -1.0

    def test_equity_never_negative(self, results):
        """Capital must remain ≥ 0 at every point in the equity curve."""
        assert min(results["equity_curve"]) >= 0

    def test_losses_stop_after_pause(self, results):
        """Equity must be flat from the drawdown-pause point onward."""
        equity = results["equity_curve"]
        min_eq  = min(equity)
        min_idx = equity.index(min_eq)
        assert all(
            e == pytest.approx(min_eq, abs=0.01) for e in equity[min_idx:]
        ), "Equity changed after trading was paused by drawdown guard"


# ---------------------------------------------------------------------------
# 2021 bull-market regime
# ---------------------------------------------------------------------------


class TestBullMarket:
    """Correct predictions in a sustained uptrend must produce positive returns."""

    @pytest.fixture
    def results(self):
        np.random.seed(0)
        price_data = _price_frame(_BULL_LONG, _BULL_SHORT, start="2021-03-01")
        return backtest_strategy(
            _BULL_PREDS,
            price_data["Price_Difference"].values,
            price_data,
            threshold=0.5,
            initial_capital=10_000,
            max_drawdown=0.05,
        )

    def test_positive_return(self, results):
        """Net return must be positive when the model is consistently right."""
        assert results["total_return_pct"] > 0, (
            f"Expected positive return, got {results['total_return_pct']:.2f} %"
        )

    def test_trades_executed(self, results):
        """At least one trade must have been filled during the window."""
        assert results["num_trades"] > 0

    def test_no_drawdown_breach(self, results):
        """A profitable bull run must not trigger the drawdown guard."""
        assert not results["drawdown_limit_reached"]


# ---------------------------------------------------------------------------
# 2022 bear-market regime
# ---------------------------------------------------------------------------


class TestBearMarket:
    """Consistently wrong predictions in a downtrend must produce losses."""

    @pytest.fixture
    def results(self):
        np.random.seed(0)
        price_data = _price_frame(_BEAR_LONG, _BEAR_SHORT, start="2022-01-03")
        return backtest_strategy(
            _BEAR_PREDS,
            price_data["Price_Difference"].values,
            price_data,
            threshold=0.5,
            initial_capital=10_000,
            max_drawdown=0.05,
        )

    def test_negative_return(self, results):
        """Persistent mispredictions in a bear market must produce a loss."""
        assert results["total_return_pct"] < 0, (
            f"Expected negative return, got {results['total_return_pct']:.2f} %"
        )

    def test_trades_executed(self, results):
        """At least one trade must have been filled (not all rejected)."""
        assert results["num_trades"] > 0

    def test_drawdown_bounded(self, results):
        """With a 5 % drawdown cap the max loss must stay within a safe margin."""
        # Allow a small buffer for the one-iteration delay before the guard fires
        assert results["max_drawdown_pct"] >= -7.0, (
            f"Drawdown {results['max_drawdown_pct']:.2f} % is unexpectedly large"
        )
