"""Hermes — conversational messaging engine for Coppermind.

Handles incoming messages from iMessage (and future channels),
assembles memory context, calls Claude, and returns replies.
"""

import logging
import os
import json
import re
from datetime import datetime, date
from typing import Optional, Tuple

import anthropic
import redis

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}
DEFAULT_MODEL = "haiku"

# ── Budget config ────────────────────────────────────────────

DAILY_BUDGET_USD = float(os.environ.get("HERMES_DAILY_BUDGET", "5.0"))

# ── Proactive messaging config ───────────────────────────────

PROACTIVE_SSH_HOST = os.environ.get("HERMES_SSH_HOST", "ben-mac")
PROACTIVE_SEND_SCRIPT = os.environ.get("HERMES_SEND_SCRIPT", "/Users/benfinklea/bin/send_imessage.py")
REDIS_HOST = os.environ.get("COPPERMIND_REDIS_HOST", "100.64.219.124")
REDIS_PORT = 6379

# Approximate cost per 1K tokens (input/output) for budget tracking
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
}

# ── Redis client (lazy init) ────────────────────────────────

_redis_client = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        from pathlib import Path
        password = Path("~/.redis-password").expanduser().read_text().strip()
        _redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            password=password, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5,
        )
    return _redis_client


# ── Anthropic client (lazy init) ─────────────────────────────

_anthropic_client = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        from pathlib import Path
        api_key = Path("~/.anthropic-api-key").expanduser().read_text().strip()
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── Public API ───────────────────────────────────────────────


def process_message(content: str, channel: str, model_override: Optional[str], mem, images: list = None) -> dict:
    """Main entry point: process an incoming message and return a reply.

    Returns dict with: reply, model_used, cost_usd, intent
    """
    # 1. Parse model override from message prefix
    model_key, content, model_explicit = _select_model(content, model_override)

    # Auto-upgrade to Sonnet for image messages (only when using default model)
    if images and model_key == "haiku" and not model_explicit:
        model_key = "sonnet"
        logger.info("Auto-upgraded to Sonnet for image message")

    # Handle bare model command with no message text
    if not content and not images:
        _store_turn(mem, "user", f"/{model_key}")
        reply = f"Switched to {model_key}. Send your next message."
        _store_turn(mem, "assistant", reply)
        return {
            "reply": reply,
            "model_used": model_key,
            "cost_usd": 0.0,
            "intent": "model_switch",
        }

    model_id = MODELS[model_key]

    # 2. Detect slash commands (handled without Claude)
    intent, intent_response = _detect_intent(content, mem)
    if intent_response is not None:
        # Store both turns
        _store_turn(mem, "user", content)
        _store_turn(mem, "assistant", intent_response)
        return {
            "reply": intent_response,
            "model_used": "none",
            "cost_usd": 0.0,
            "intent": intent,
        }

    # 3. Check budget
    budget_ok, budget_msg = _check_budget()
    if not budget_ok:
        return {
            "reply": budget_msg,
            "model_used": "none",
            "cost_usd": 0.0,
            "intent": "budget_exceeded",
        }

    # 4. Assemble context (memories + recent history)
    context_str = _assemble_context(content, mem)
    history = _get_recent_history(mem)

    # 5. Build system prompt
    system_prompt = _build_system_prompt(context_str, channel)

    # 6. Call Claude
    reply, cost_usd = _call_claude(content, system_prompt, model_id, history, images=images)

    # 7. Record cost
    _record_cost(cost_usd, model_id)

    # 8. Store both turns (without base64 image data)
    if images:
        n = len(images)
        store_content = f"[Sent {n} image{'s' if n > 1 else ''}, no longer visible] {content}" if content else f"[Sent {n} image{'s' if n > 1 else ''}, no longer visible]"
    else:
        store_content = content
    _store_turn(mem, "user", store_content)
    _store_turn(mem, "assistant", reply)

    return {
        "reply": reply,
        "model_used": model_key,
        "cost_usd": cost_usd,
        "intent": None,
    }


# ── Internal functions ───────────────────────────────────────


def _select_model(content: str, model_override: Optional[str] = None) -> Tuple[str, str, bool]:
    """Parse /haiku, /sonnet, /opus prefix from message. Returns (model_key, cleaned_content, was_explicit)."""
    if model_override and model_override in MODELS:
        return model_override, content, True

    lower = content.lower()
    for key in MODELS:
        prefix = f"/{key}"
        if lower == prefix or lower.startswith(prefix + " "):
            cleaned = content[len(prefix):].strip()
            return key, cleaned, True

    return DEFAULT_MODEL, content, False


def _detect_intent(content: str, mem) -> Tuple[Optional[str], Optional[str]]:
    """Detect slash commands that bypass Claude. Returns (intent, response) or (None, None)."""
    stripped = content.strip()

    # /remember <text>
    if stripped.lower().startswith("/remember "):
        text = stripped[10:].strip()
        if not text:
            return "remember", "What would you like me to remember?"
        try:
            learning_id = mem.learning.postgres_store.store_knowledge(
                agent_name=mem.agent_name,
                category="fact",
                content=text,
                source="hermes_imessage",
                confidence=0.9,
                learning_intensity="routine",
                source_type="explicit_user",
                source_confidence=1.0,
                memory_type="spark",
                metadata={
                    "spark": True,
                    "capture_source": "hermes_imessage",
                    "original_transcript": text,
                    "evaluated": False,
                },
                skip_quality_gates=True,
            )
            return "remember", f"Remembered: {text[:100]}{'...' if len(text) > 100 else ''}"
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return "remember", f"Failed to store: {e}"

    # /recall <query>
    if stripped.lower().startswith("/recall "):
        query = stripped[8:].strip()
        if not query:
            return "recall", "What should I search for?"
        try:
            results = mem.learning.postgres_store.search_knowledge(
                agent_name=mem.agent_name,
                query=query,
                limit=5,
            )
            if not results:
                return "recall", f"No memories found for: {query}"
            lines = []
            for r in results:
                content_preview = r.get("content", "")[:120]
                sim = r.get("similarity", 0)
                lines.append(f"• {content_preview} ({sim:.0%})")
            return "recall", "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return "recall", f"Search failed: {e}"

    # /tasks
    if stripped.lower() == "/tasks":
        try:
            r = _get_redis()
            # Check for Coppermind tasks
            tasks_raw = r.lrange("coppermind:tasks", 0, -1)
            if not tasks_raw:
                return "tasks", "No tasks found."
            import json
            lines = []
            for t in tasks_raw:
                task = json.loads(t) if isinstance(t, str) else t
                status = task.get("status", "pending")
                title = task.get("title", "untitled")
                emoji = "✅" if status == "done" else "⏳"
                lines.append(f"{emoji} {title}")
            return "tasks", "\n".join(lines)
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return "tasks", f"Failed to list tasks: {e}"

    # /task <title>
    if stripped.lower().startswith("/task "):
        title = stripped[6:].strip()
        if not title:
            return "task", "What's the task?"
        try:
            import json
            r = _get_redis()
            task = json.dumps({"title": title, "status": "pending", "created": datetime.utcnow().isoformat()})
            r.rpush("coppermind:tasks", task)
            return "task", f"Task added: {title}"
        except Exception as e:
            logger.error(f"Failed to add task: {e}")
            return "task", f"Failed to add task: {e}"

    # /status
    if stripped.lower() == "/status":
        try:
            r = _get_redis()
            today = date.today().isoformat()
            spent = float(r.get(f"hermes:cost:{today}") or 0)
            remaining = DAILY_BUDGET_USD - spent
            return "status", (
                f"Budget: ${spent:.3f} / ${DAILY_BUDGET_USD:.2f} today\n"
                f"Remaining: ${remaining:.3f}\n"
                f"Default model: {DEFAULT_MODEL}"
            )
        except Exception as e:
            return "status", f"Status check failed: {e}"

    return None, None


def _assemble_context(content: str, mem) -> str:
    """Build context string from relevant memories + recent conversation."""
    sections = []

    # Semantic search for relevant memories
    try:
        results = mem.learning.postgres_store.search_knowledge(
            agent_name=mem.agent_name,
            query=content,
            limit=10,
            threshold=0.3,
        )
        if results:
            memory_lines = []
            for r in results:
                mem_content = r.get("content", "")[:300]
                category = r.get("category", "fact")
                sim = r.get("similarity", 0)
                memory_lines.append(f"[{category}] {mem_content} (relevance: {sim:.0%})")
                # Record access for spaced repetition
                try:
                    mem.learning.postgres_store.record_access(str(r["id"]), "retrieval")
                except Exception:
                    pass
            sections.append("## Relevant Memories\n" + "\n".join(memory_lines))
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")

    return "\n\n".join(sections) if sections else ""


def _get_recent_history(mem, limit: int = 20) -> list:
    """Get recent Hermes conversation turns as Claude message format."""
    try:
        conn = mem.learning.postgres_store._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT sender, content, msg_timestamp
                FROM conversation_context
                WHERE agent_name = %s AND project = 'hermes'
                ORDER BY msg_timestamp DESC
                LIMIT %s
            """, (mem.agent_name, limit))
            rows = cur.fetchall()

        # Reverse to chronological order
        rows = list(reversed(rows))

        messages = []
        for sender, content, ts in rows:
            role = "user" if sender == "user" else "assistant"
            messages.append({"role": role, "content": content})

        return messages
    except Exception as e:
        logger.warning(f"Failed to get history: {e}")
        return []


def _build_system_prompt(context: str, channel: str) -> str:
    """Build the system prompt for Claude."""
    base = (
        "You are Ben's personal AI assistant. You communicate via iMessage. "
        "You have access to Ben's personal memory system (Coppermind) which stores facts, "
        "preferences, patterns, and decisions he's shared over time.\n\n"
        "Guidelines:\n"
        "- Be concise — this is texting, not email. Short replies are better.\n"
        "- Be warm and helpful, like a knowledgeable friend.\n"
        "- Use the provided memories to give personalized, contextual answers.\n"
        "- If you don't know something and it's not in the memories, say so.\n"
        "- You can reference things from earlier in the conversation naturally.\n"
        "- Messages in history prefixed with [Sent N image(s), no longer visible] mean the user previously sent images "
        "that are no longer visible. If asked about prior images, note you can only see images "
        "in the current message.\n"
        "- No emojis unless Ben uses them first.\n"
        "- For technical topics, be precise but not verbose.\n"
        f"- Current date: {date.today().isoformat()}\n"
        f"- Channel: {channel}\n"
    )

    if context:
        base += f"\n{context}\n"

    base += (
        "\nAvailable commands Ben can use:\n"
        "/remember <text> — store a memory\n"
        "/recall <query> — search memories\n"
        "/tasks — list tasks\n"
        "/task <title> — add a task\n"
        "/status — check budget and model info\n"
        "/sonnet or /opus — use a more powerful model for this message\n"
    )

    return base


def _call_claude(message: str, system: str, model: str, history: list, images: list = None) -> Tuple[str, float]:
    """Call Claude API and return (reply_text, cost_usd)."""
    client = _get_anthropic()

    messages = history.copy()

    # Build content blocks for the user message
    if images:
        content_blocks = []
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"],
                },
            })
        # Add text block (default prompt if no text with image)
        text = message.strip() if message else "What's in this image?"
        content_blocks.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content_blocks})
    else:
        messages.append({"role": "user", "content": message})

    # Longer timeout for vision requests
    timeout = 120.0 if images else 60.0

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
            timeout=timeout,
        )

        reply = response.content[0].text

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        costs = MODEL_COSTS.get(model, MODEL_COSTS["claude-haiku-4-5-20251001"])
        cost_usd = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000

        logger.info(
            f"Claude {model}: {input_tokens}in/{output_tokens}out "
            f"= ${cost_usd:.4f}"
        )

        return reply, cost_usd

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"Sorry, I hit an error: {str(e)[:200]}", 0.0


def _store_turn(mem, sender: str, content: str):
    """Store a conversation turn in conversation_context with project='hermes'."""
    try:
        conn = mem.learning.postgres_store._get_conn()
        from lib.postgres_store import compute_embedding
        embedding = compute_embedding(content)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_context
                (agent_name, sender, content, msg_timestamp, project, embedding)
                VALUES (%s, %s, %s, %s, 'hermes', %s)
            """, (mem.agent_name, sender, content, datetime.utcnow(), embedding))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to store turn: {e}")


def _check_budget() -> Tuple[bool, str]:
    """Check if daily budget is exceeded. Returns (ok, message)."""
    try:
        r = _get_redis()
        today = date.today().isoformat()
        spent = float(r.get(f"hermes:cost:{today}") or 0)
        if spent >= DAILY_BUDGET_USD:
            return False, (
                f"Daily budget reached (${spent:.2f} / ${DAILY_BUDGET_USD:.2f}). "
                "Try again tomorrow, or ask Ben to increase the limit."
            )
        return True, ""
    except Exception as e:
        logger.warning(f"Budget check failed (allowing): {e}")
        return True, ""


def _record_cost(cost_usd: float, model: str):
    """Record cost in Redis with daily TTL."""
    try:
        r = _get_redis()
        today = date.today().isoformat()

        # Total daily cost
        key = f"hermes:cost:{today}"
        r.incrbyfloat(key, cost_usd)
        r.expire(key, 172800)  # 48h TTL

        # Per-model breakdown
        model_key = f"hermes:cost:{today}:{model}"
        r.incrbyfloat(model_key, cost_usd)
        r.expire(model_key, 172800)
    except Exception as e:
        logger.warning(f"Failed to record cost: {e}")


def send_imessage_proactive(recipient: str, text: str) -> bool:
    """Send an iMessage proactively by SSH-ing to ben.local and piping to send_imessage.py."""
    import subprocess

    payload = json.dumps({"recipient": recipient, "text": text})

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=5",
                "-o", "ServerAliveInterval=5",
                "-o", "ServerAliveCountMax=2",
                PROACTIVE_SSH_HOST,
                "python3", PROACTIVE_SEND_SCRIPT,
            ],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info(f"Proactive iMessage sent to {recipient}: {text[:80]}...")
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning(f"Proactive iMessage failed: {stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Proactive iMessage SSH timed out (Mac may be asleep)")
        return False
    except Exception as e:
        logger.warning(f"Proactive iMessage failed: {e}")
        return False
