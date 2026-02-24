"""Confirmation Service — reusable voice + dialog confirmation via SSH to Mac.

Presents a TTS readback via `say` and a macOS dialog with Send/Cancel buttons.
Timeout = don't send (safe by default, like CarPlay).

Usage:
    from confirmation import confirm_action
    result = confirm_action("Send email to Janis about meeting notes?")
    # Returns: "approved", "rejected", "timeout", or "error"
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# SSH target — same as evaluator.py
CONFIRMATION_SSH_HOST = os.environ.get("REMINDER_SSH_HOST", "ben-mac")


def _escape_applescript(s: str) -> str:
    """Escape a string for AppleScript double-quoted string context."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def confirm_action(summary: str, details: str = "", timeout: int = 30) -> str:
    """Present a confirmation dialog on Mac and wait for user response.

    Uses SSH to Mac → `say` (TTS readback) + `display dialog` (Send/Cancel).
    The AppleScript is piped via heredoc to avoid nested shell quoting issues.

    Args:
        summary: Short description read aloud and shown in dialog title.
        details: Longer text shown in the dialog body (e.g., email preview).
        timeout: Seconds before auto-dismiss (treated as rejection).

    Returns:
        "approved"  — user clicked Send
        "rejected"  — user clicked Cancel
        "timeout"   — dialog auto-dismissed (Mac asleep, user away, etc.)
        "error"     — SSH or other failure
    """
    # Escape summary for say command (single-quote shell context)
    safe_summary_say = summary.replace("'", "'\\''")

    # Escape for AppleScript double-quote context
    safe_summary_as = _escape_applescript(summary)
    safe_details_as = _escape_applescript(details)

    if details:
        dialog_text = f"{safe_summary_as}\\n\\n{safe_details_as}"
    else:
        dialog_text = safe_summary_as

    # Build bash script using a heredoc for the AppleScript portion.
    # The quoted heredoc (<<'APPLESCRIPT_EOF') prevents bash from
    # interpreting anything inside, so only AppleScript escaping is needed.
    # This eliminates the fragile multi-layer quoting of osascript -e '...'.
    bash_script = (
        f"say '{safe_summary_say}' &\n"
        f"CONSOLE_UID=$(id -u \"$(stat -f '%Su' /dev/console)\")\n"
        f"launchctl asuser $CONSOLE_UID osascript <<'APPLESCRIPT_EOF'\n"
        f'display dialog "{dialog_text}" '
        f'buttons {{"Cancel", "Send"}} '
        f'default button "Send" '
        f"giving up after {timeout}\n"
        f"APPLESCRIPT_EOF\n"
    )

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=5",
                "-o", "ServerAliveInterval=2",
                "-o", "ServerAliveCountMax=1",
                CONFIRMATION_SSH_HOST,
                "bash", "-s",
            ],
            input=bash_script.encode("utf-8"),
            capture_output=True,
            timeout=timeout + 15,  # SSH overhead + dialog timeout + buffer
        )

        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip()

        if result.returncode != 0:
            # User clicked Cancel → osascript exits with code 1
            # "User canceled" in stderr is the standard AppleScript cancel message
            if "User canceled" in stderr or result.returncode == 1:
                logger.info(f"confirmation_rejected: {summary[:50]}")
                return "rejected"
            logger.warning(f"confirmation_error: rc={result.returncode} stderr={stderr}")
            return "error"

        # Parse osascript output: "button returned:Send, gave up:false"
        if "gave up:true" in stdout:
            logger.info(f"confirmation_timeout: {summary[:50]}")
            return "timeout"

        if "button returned:Send" in stdout:
            logger.info(f"confirmation_approved: {summary[:50]}")
            return "approved"

        # Unexpected output — treat as rejection for safety
        logger.warning(f"confirmation_unexpected: stdout={stdout}")
        return "rejected"

    except subprocess.TimeoutExpired:
        logger.warning(f"confirmation_ssh_timeout: Mac may be asleep")
        return "timeout"
    except Exception as e:
        logger.warning(f"confirmation_error: {e}")
        return "error"
