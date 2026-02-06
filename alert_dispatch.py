"""
Alert Dispatch Module
Routes HIGH / CRITICAL alerts to configured webhook sinks.

Configuration
-------------
Set the environment variable ALERT_WEBHOOK_URL to a URL that accepts a
POST with a JSON body.  Works out of the box with Slack incoming webhooks,
Discord webhooks, PagerDuty Events API v2, or any generic HTTP endpoint.

If the variable is not set, HIGH / CRITICAL alerts fall back to a loud
message on stderr so the operator is not silently unaware.
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime

logger = logging.getLogger(__name__)


def dispatch_alert(alert: dict):
    """
    Route an alert to all configured sinks.

    The file sink (logs/alerts.log) is handled upstream by
    ``logging_config.log_drift_alert``.  This function is responsible for
    the webhook sink and the stderr fallback.

    Args:
        alert: Dict with at minimum ``severity`` and ``message`` keys.
    """
    severity = alert.get('severity', 'LOW').upper()
    webhook_url = os.environ.get('ALERT_WEBHOOK_URL', '').strip()

    if webhook_url:
        _post_to_webhook(webhook_url, alert)
    elif severity in ('HIGH', 'CRITICAL'):
        _stderr_fallback(alert)


def _post_to_webhook(url: str, alert: dict):
    """POST the alert payload as JSON to *url*.  Detects Slack format automatically."""
    payload = {'timestamp': datetime.now().isoformat(), **alert}

    # When the target is a Slack incoming webhook, wrap in Slack message format
    if 'hooks.slack.com' in url:
        text = f"[{alert.get('severity', 'INFO')}] {alert.get('message', '')}"
        payload = {
            'text': text,
            'attachments': [{'text': json.dumps(alert, indent=2)}],
        }

    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info("Alert dispatched to webhook (HTTP %d)", resp.status)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        # Log but do NOT crash — alert delivery failure must never kill the
        # trading loop.
        logger.error("Failed to POST alert to %s: %s", url, e)


def _stderr_fallback(alert: dict):
    """Print a loud warning to stderr when no webhook is configured."""
    timestamp = datetime.now().isoformat()
    print(
        f"\n*** [{timestamp}] ALERT ({alert.get('severity', '???')}) — "
        f"no ALERT_WEBHOOK_URL configured ***\n"
        f"    {alert.get('message', 'No message')}\n"
        f"    Set env var ALERT_WEBHOOK_URL to route alerts to Slack / PagerDuty / etc.\n",
        file=sys.stderr,
        flush=True,
    )
