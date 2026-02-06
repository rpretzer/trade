"""
Unit tests for the multi-model ensemble prediction layer.

Covers:
  - ensemble_predict() weighted-average maths
  - _xgb_predict() ModelLoadError when xgboost is missing
  - The try/except fallback that preserves the LSTM prediction when XGBoost fails
"""

import pytest
import numpy as np
import unittest.mock as mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_trading import ensemble_predict, ENSEMBLE_WEIGHTS, _xgb_predict


# ---------------------------------------------------------------------------
# ensemble_predict — pure-function tests
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    """Weighted-average combiner correctness."""

    def test_default_weights_favour_lstm(self):
        """Default (0.6 / 0.4): LSTM=1, XGB=0 → 0.6."""
        assert ensemble_predict(1.0, 0.0) == pytest.approx(0.6)

    def test_default_weights_favour_xgb(self):
        """Default (0.6 / 0.4): LSTM=0, XGB=1 → 0.4."""
        assert ensemble_predict(0.0, 1.0) == pytest.approx(0.4)

    def test_equal_weights_gives_mean(self):
        """Equal weights → simple average."""
        assert ensemble_predict(1.0, 0.0, weights=(1.0, 1.0)) == pytest.approx(0.5)

    def test_lstm_only_weight(self):
        """XGBoost weight = 0 → LSTM prediction unchanged."""
        assert ensemble_predict(0.75, 999.0, weights=(1.0, 0.0)) == pytest.approx(0.75)

    def test_xgb_only_weight(self):
        """LSTM weight = 0 → XGBoost prediction unchanged."""
        assert ensemble_predict(999.0, -0.3, weights=(0.0, 1.0)) == pytest.approx(-0.3)

    def test_negative_predictions_blend_correctly(self):
        """Both negative: (0.6 * -0.8 + 0.4 * -0.4) / 1.0 = -0.64."""
        assert ensemble_predict(-0.8, -0.4) == pytest.approx(-0.64)

    def test_unnormalised_weights_are_normalised(self):
        """Weights don't need to sum to 1: (2*10 + 3*20) / 5 = 16.0."""
        assert ensemble_predict(10.0, 20.0, weights=(2.0, 3.0)) == pytest.approx(16.0)

    def test_identical_predictions_give_same_value(self):
        """When both models agree, any weight split returns that value."""
        assert ensemble_predict(0.42, 0.42, weights=(0.7, 0.3)) == pytest.approx(0.42)

    def test_default_weights_constant_matches_docstring(self):
        """ENSEMBLE_WEIGHTS is (0.6, 0.4) as documented."""
        assert ENSEMBLE_WEIGHTS == (0.6, 0.4)


# ---------------------------------------------------------------------------
# _xgb_predict — import-failure path
# ---------------------------------------------------------------------------

class TestXgbPredict:
    """Verify _xgb_predict raises cleanly when xgboost is unavailable."""

    def test_raises_model_load_error_when_xgboost_missing(self):
        """Blocking the xgboost import produces a ModelLoadError."""
        from exceptions import ModelLoadError

        # Force `from xgboost import XGBRegressor` to fail regardless of
        # whether xgboost is actually installed in this env.
        with mock.patch.dict(sys.modules, {'xgboost': None}):
            with pytest.raises(ModelLoadError, match="XGBoost not available"):
                _xgb_predict(np.array([[1.0, 2.0]]), ['a', 'b'])


# ---------------------------------------------------------------------------
# Ensemble fallback — mirrors the try/except in get_latest_signal
# ---------------------------------------------------------------------------

class TestEnsembleFallback:
    """XGBoost failure must never clobber the LSTM prediction."""

    def test_prediction_unchanged_when_xgb_raises(self, monkeypatch):
        """Replays the 7-line ensemble block: exception → LSTM value survives."""
        import live_trading

        monkeypatch.setenv('ENSEMBLE_MODE', '1')

        lstm_pred = 0.8
        prediction = lstm_pred  # starting value, same as in get_latest_signal

        with mock.patch.object(live_trading, '_xgb_predict',
                               side_effect=RuntimeError("model file missing")):
            # Exact try/except structure from get_latest_signal
            try:
                xgb_pred = live_trading._xgb_predict(None, None)
                prediction = live_trading.ensemble_predict(prediction, xgb_pred)
            except Exception:
                pass  # production code logs a warning here

        assert prediction == pytest.approx(lstm_pred)

    def test_prediction_blended_when_xgb_succeeds(self, monkeypatch):
        """Happy path: _xgb_predict returns a value → prediction is blended."""
        import live_trading

        monkeypatch.setenv('ENSEMBLE_MODE', '1')

        lstm_pred = 0.8
        xgb_pred_value = 0.2
        prediction = lstm_pred

        with mock.patch.object(live_trading, '_xgb_predict',
                               return_value=xgb_pred_value):
            try:
                xgb_pred = live_trading._xgb_predict(None, None)
                prediction = live_trading.ensemble_predict(prediction, xgb_pred)
            except Exception:
                pass

        # 0.6 * 0.8 + 0.4 * 0.2 = 0.48 + 0.08 = 0.56
        assert prediction == pytest.approx(0.56)

    def test_ensemble_mode_unset_means_no_blend(self, monkeypatch):
        """When ENSEMBLE_MODE is absent the blend branch is never entered."""
        monkeypatch.delenv('ENSEMBLE_MODE', raising=False)

        lstm_pred = 0.8
        prediction = lstm_pred

        # Gate check — identical to the `if` in get_latest_signal
        if os.environ.get('ENSEMBLE_MODE'):
            prediction = -999.0  # sentinel: should never execute

        assert prediction == pytest.approx(lstm_pred)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
