"""
Unit tests for alert_dispatch module
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alert_dispatch import dispatch_alert


class TestAlertDispatch:
    """Verify alert routing logic."""

    def test_high_alert_prints_to_stderr_when_no_webhook(self, capsys, monkeypatch):
        """HIGH severity alerts fall back to stderr when ALERT_WEBHOOK_URL is unset."""
        monkeypatch.delenv('ALERT_WEBHOOK_URL', raising=False)

        dispatch_alert({'severity': 'HIGH', 'message': 'position limit breached'})

        captured = capsys.readouterr()
        assert 'HIGH' in captured.err
        assert 'position limit breached' in captured.err
        assert 'ALERT_WEBHOOK_URL' in captured.err

    def test_critical_alert_prints_to_stderr_when_no_webhook(self, capsys, monkeypatch):
        """CRITICAL severity alerts fall back to stderr when ALERT_WEBHOOK_URL is unset."""
        monkeypatch.delenv('ALERT_WEBHOOK_URL', raising=False)

        dispatch_alert({'severity': 'CRITICAL', 'message': 'circuit breaker fired'})

        captured = capsys.readouterr()
        assert 'CRITICAL' in captured.err
        assert 'circuit breaker fired' in captured.err

    def test_low_severity_is_silent_when_no_webhook(self, capsys, monkeypatch):
        """LOW / MEDIUM alerts produce no stderr noise when webhook is unset."""
        monkeypatch.delenv('ALERT_WEBHOOK_URL', raising=False)

        dispatch_alert({'severity': 'LOW', 'message': 'routine check'})
        dispatch_alert({'severity': 'MEDIUM', 'message': 'minor anomaly'})

        captured = capsys.readouterr()
        assert captured.err == ''

    def test_missing_severity_defaults_to_low(self, capsys, monkeypatch):
        """An alert with no severity key is treated as LOW — no stderr output."""
        monkeypatch.delenv('ALERT_WEBHOOK_URL', raising=False)

        dispatch_alert({'message': 'no severity field'})

        captured = capsys.readouterr()
        assert captured.err == ''

    def test_webhook_post_does_not_crash_on_bad_url(self, monkeypatch):
        """dispatch_alert handles an unreachable webhook URL gracefully."""
        monkeypatch.setenv('ALERT_WEBHOOK_URL', 'http://192.0.2.1:9999/nope')

        # Should not raise — failure is logged, not propagated
        dispatch_alert({'severity': 'HIGH', 'message': 'test'})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
