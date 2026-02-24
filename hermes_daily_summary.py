#!/usr/bin/env python3
"""Hermes Daily Cost Summary — sends cost report via iMessage if threshold exceeded.

Designed to run as a systemd oneshot service triggered by a timer at 11pm CT.
Idempotent: checks Redis guard key before sending.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hermes-daily-summary")

# Config
COST_THRESHOLD = float(os.environ.get("HERMES_COST_SUMMARY_THRESHOLD", "1.0"))
REDIS_HOST = os.environ.get("COPPERMIND_REDIS_HOST", "100.64.219.124")
REDIS_PORT = 6379
SSH_HOST = os.environ.get("HERMES_SSH_HOST", "ben-mac")
SEND_SCRIPT = os.environ.get("HERMES_SEND_SCRIPT", "/Users/benfinklea/bin/send_imessage.py")
RECIPIENT = os.environ.get("HERMES_CHAT_IDENTIFIER", "coppermind@volacci.com")

# Known models for cost breakdown
KNOWN_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]
MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "Haiku",
    "claude-sonnet-4-6": "Sonnet",
    "claude-opus-4-6": "Opus",
}


def get_redis():
    import redis
    password = Path("~/.redis-password").expanduser().read_text().strip()
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT,
        password=password, decode_responses=True,
        socket_timeout=5, socket_connect_timeout=5,
    )


def send_imessage(text: str) -> bool:
    """Send iMessage via SSH to Mac."""
    payload = json.dumps({"recipient": RECIPIENT, "text": text})
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=3",
                SSH_HOST,
                "python3", SEND_SCRIPT,
            ],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"iMessage sent: {text[:80]}...")
            return True
        stderr = result.stderr.decode("utf-8", errors="replace")
        logger.error(f"iMessage send failed: {stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("SSH timed out (Mac may be asleep)")
        return False
    except Exception as e:
        logger.error(f"Failed to send iMessage: {e}")
        return False


def main():
    today = date.today().isoformat()
    r = get_redis()

    # Idempotency guard
    guard_key = f"hermes:summary_sent:{today}"
    if r.get(guard_key):
        logger.info(f"Summary already sent for {today}")
        return

    # Get total cost
    total_cost = float(r.get(f"hermes:cost:{today}") or 0)

    # Check threshold
    if total_cost < COST_THRESHOLD:
        logger.info(f"Cost ${total_cost:.2f} < threshold ${COST_THRESHOLD:.2f} — skipping")
        return

    # Get message count
    msg_count = int(r.get(f"hermes:msg_count:{today}") or 0)

    # Get per-model breakdown
    breakdown_lines = []
    for model_id in KNOWN_MODELS:
        model_cost = float(r.get(f"hermes:cost:{today}:{model_id}") or 0)
        if model_cost > 0:
            short_name = MODEL_SHORT.get(model_id, model_id)
            breakdown_lines.append(f"- {short_name}: ${model_cost:.2f}")

    # Format message
    lines = [
        "Coppermind> Daily summary",
        f"Messages: {msg_count}",
        f"Cost: ${total_cost:.2f}",
    ]
    lines.extend(breakdown_lines)
    message = "\n".join(lines)

    # Send
    if send_imessage(message):
        # Set idempotency guard (48h TTL)
        r.set(guard_key, "1", ex=172800)
        logger.info("Daily summary sent successfully")
    else:
        logger.error("Failed to send daily summary")
        sys.exit(1)


if __name__ == "__main__":
    main()
