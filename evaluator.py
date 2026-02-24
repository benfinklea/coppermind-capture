"""Spark Evaluator — LLM classification + Apple Reminders routing.

Store-first, evaluate-after pattern. Classifies sparks via Ollama (gemma3:12b)
with Haiku fallback, lightly cleans voice artifacts, and routes actionable
items to Apple Reminders via SSH to ben.local.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Coppermind path — set by capture.py before import
COPPERMIND_PATH = os.environ.get("COPPERMIND_PATH", "/workspace/coppermind")

# SSH target for Apple Reminders
REMINDER_SSH_HOST = os.environ.get("REMINDER_SSH_HOST", "ben-mac")
REMINDER_SCRIPT = os.environ.get("REMINDER_SCRIPT",
                                  "/Users/benfinklea/bin/create_reminder.py")

# Timezone for prompt context
USER_TIMEZONE = os.environ.get("USER_TIMEZONE", "America/Chicago")


# ── Pydantic model for LLM response validation ──────────────

VALID_CATEGORIES = {"fact", "pattern", "preference", "decision", "gotcha"}
VALID_INTENSITIES = {"routine", "urgent", "error_recovery", "breakthrough", "critical_failure"}
VALID_ACTION_TYPES = {"send_email", "send_text", "send_slack", None}

# SSH target + script for email sending
EMAIL_SCRIPT = os.environ.get("EMAIL_SCRIPT",
                               "/Users/benfinklea/bin/send_email.py")
VALID_CONTEXTS = {
    "grocery_store", "at_computer", "errands", "home",
    "driving", "anywhere", "work", None
}

# Map LLM context values to Apple Reminders list names
CONTEXT_TO_LIST = {
    "grocery_store": "Grocery Store",
    "at_computer": "At Computer",
    "errands": "Errands",
    "home": "Home",
    "driving": "While Driving",
    "anywhere": "Anywhere",
    "work": "Work",
    None: "Anywhere",
}


VALID_DUE_DAYS = {
    "today", "tomorrow", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", None
}

DAY_NAME_TO_WEEKDAY = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class SparkClassification(BaseModel):
    """Validated LLM classification of a spark."""
    category: str = "fact"
    intensity: str = "routine"
    confidence: float = 0.5
    is_actionable: bool = False
    task_title: Optional[str] = None
    context: Optional[str] = None
    due_day: Optional[str] = None      # "today", "tomorrow", "monday", etc.
    due_time: Optional[str] = None     # "HH:MM" 24h format, or "morning"/"afternoon"/"evening"
    cleaned_content: str = ""
    reasoning: str = ""

    # Keep due_date for backward compat if LLM still returns it
    due_date: Optional[str] = None

    # Enrichment: explicit lookup requests
    needs_enrichment: bool = False
    enrichment_queries: Optional[list[str]] = None

    # Action detection: send email/text/Slack
    action_type: Optional[str] = None       # "send_email" | "send_text" | "send_slack"
    action_recipient: Optional[str] = None  # "Janis", "Mom", "the team"
    action_message: Optional[str] = None    # "I'll be 5 minutes late"

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in VALID_CATEGORIES:
            return "fact"
        return v

    @field_validator("intensity")
    @classmethod
    def validate_intensity(cls, v):
        if v not in VALID_INTENSITIES:
            return "routine"
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @field_validator("context")
    @classmethod
    def validate_context(cls, v):
        if v not in VALID_CONTEXTS:
            return None
        return v

    @field_validator("due_day")
    @classmethod
    def validate_due_day(cls, v):
        if v is not None:
            v = str(v).lower().strip()
        if v not in VALID_DUE_DAYS:
            return None
        return v

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        if v is not None:
            v = str(v).lower().strip()
        if v not in VALID_ACTION_TYPES:
            return None
        return v


def resolve_due_date(due_day: Optional[str], due_time: Optional[str]) -> Optional[str]:
    """Compute an ISO datetime from LLM's due_day + due_time.

    Python handles the date math — no LLM arithmetic needed.
    Returns ISO string with timezone, or None.
    """
    if not due_day:
        return None

    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(USER_TIMEZONE)
    except Exception:
        tz = timezone(timedelta(hours=-6))

    now = datetime.now(tz)
    due_day = due_day.lower().strip()

    # Resolve the date
    if due_day == "today":
        target_date = now.date()
    elif due_day == "tomorrow":
        target_date = (now + timedelta(days=1)).date()
    elif due_day in DAY_NAME_TO_WEEKDAY:
        target_weekday = DAY_NAME_TO_WEEKDAY[due_day]
        days_ahead = (target_weekday - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # "Monday" on a Monday means next Monday
        target_date = (now + timedelta(days=days_ahead)).date()
    else:
        return None

    # Resolve the time
    hour, minute = 9, 0  # default: 9:00 AM
    if due_time:
        due_time = due_time.lower().strip()
        if due_time == "morning":
            hour, minute = 9, 0
        elif due_time == "afternoon":
            hour, minute = 13, 0
        elif due_time == "evening":
            hour, minute = 18, 0
        elif due_time == "noon":
            hour, minute = 12, 0
        elif due_time == "night":
            hour, minute = 20, 0
        else:
            # Try parsing "HH:MM" or "H:MM"
            import re as _re
            m = _re.match(r'^(\d{1,2}):(\d{2})$', due_time)
            if m:
                hour, minute = int(m.group(1)), int(m.group(2))
            else:
                # Try "4 PM", "4PM", "16:00" etc
                m = _re.match(r'^(\d{1,2})\s*(am|pm)?$', due_time, _re.IGNORECASE)
                if m:
                    hour = int(m.group(1))
                    if m.group(2) and m.group(2).lower() == 'pm' and hour < 12:
                        hour += 12
                    elif m.group(2) and m.group(2).lower() == 'am' and hour == 12:
                        hour = 0

    dt = datetime(target_date.year, target_date.month, target_date.day,
                  hour, minute, 0, tzinfo=tz)

    # Format with colon in offset for Python 3.9 compat
    offset = dt.strftime('%z')
    if offset and len(offset) == 5:  # +0600 → +06:00
        offset = offset[:3] + ':' + offset[3:]
    return dt.strftime(f'%Y-%m-%dT%H:%M:%S{offset}')


# ── LLM Prompt ───────────────────────────────────────────────

def _build_prompt(content: str) -> str:
    """Build classification prompt with current datetime context."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(USER_TIMEZONE)
    except Exception:
        tz = timezone(timedelta(hours=-6))  # CST fallback

    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    day_name = now.strftime("%A")

    # Build a mini calendar so the LLM doesn't have to do date arithmetic
    calendar_lines = []
    for i in range(14):
        d = now + timedelta(days=i)
        label = ""
        if i == 0:
            label = " (TODAY)"
        elif i == 1:
            label = " (tomorrow)"
        calendar_lines.append(f"  {d.strftime('%a %b %d, %Y')}{label}")
    calendar = "\n".join(calendar_lines)

    return f"""You are a personal knowledge classifier. Classify this voice-captured thought and lightly clean it.

Current: {date_str} ({day_name}, {USER_TIMEZONE})
Calendar:
{calendar}

RULES:
- Remove filler words (um, uh, like, you know, so, basically).
- Fix grammar and punctuation.
- KEEP all location, time, emotional, and contextual details. Do NOT remove or summarize context.
- Do NOT add information not present in the original.
- If the thought is something the person needs to DO (buy, call, fix, remind, schedule, pick up, etc.), set is_actionable=true and provide a clean task_title.
- For time references, extract due_day and due_time as described below. Do NOT compute dates yourself.

CATEGORIES (pick one):
- fact: A piece of information, observation, or note
- pattern: A recognized pattern, analogy, or insight about how things work
- preference: A personal preference or opinion
- decision: A decision that was made
- gotcha: A warning, lesson learned, or thing to watch out for

INTENSITY (pick one):
- routine: Everyday thought, note, or task
- urgent: Time-sensitive or safety-related
- breakthrough: Major realization or important insight

CONTEXT (pick one, for actionable items):
- grocery_store: Shopping/groceries
- at_computer: Needs a computer
- errands: Out-and-about tasks
- home: At home tasks
- driving: While driving
- anywhere: No location constraint
- work: Work-related tasks

DUE_DAY (for actionable items with a time reference):
- "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
- Use the day name exactly as spoken. "Monday morning" → due_day: "monday"
- null if no day/date is mentioned

DUE_TIME (for actionable items with a time reference):
- "HH:MM" in 24h format if a specific time is mentioned: "at 4 PM" → "16:00"
- "morning", "afternoon", "evening", "noon", "night" if general
- null if no time is mentioned

ENRICHMENT DETECTION:
- If the person EXPLICITLY asks to LOOK UP, FIND, or SEARCH for information
  (e.g., "look up the phone number", "find their address", "search for the email"),
  set needs_enrichment=true and list what to search for in enrichment_queries.
- Only for EXPLICIT requests. "Call the plumber" is NOT enrichment.
  "Call the plumber — look up their number" IS enrichment.

ACTION DETECTION:
- If the person wants to SEND a message (email, text, Slack), detect it.
- action_type: "send_email" / "send_text" / "send_slack" / null
- action_recipient: The name of who to send to (e.g., "Janis", "Mom", "the team")
- action_message: What they want to communicate (the message content)
- "Email Janis about the meeting" → action_type: "send_email", action_recipient: "Janis", action_message: "about the meeting"
- "Text Mom I'm on my way" → action_type: "send_text", action_recipient: "Mom", action_message: "I'm on my way"
- Action sparks are ALSO actionable (is_actionable=true, get a task_title).
- If no sending action is detected, leave all action fields null.

Respond with ONLY a JSON object, no other text:

{{
  "category": "fact|pattern|preference|decision|gotcha",
  "intensity": "routine|urgent|breakthrough",
  "confidence": 0.0-1.0,
  "is_actionable": true/false,
  "task_title": "Clean title for reminder" or null,
  "context": "grocery_store|at_computer|errands|home|driving|anywhere|work" or null,
  "due_day": "today|tomorrow|monday|tuesday|..." or null,
  "due_time": "HH:MM" or "morning|afternoon|evening" or null,
  "cleaned_content": "Lightly cleaned text with all context preserved",
  "reasoning": "One sentence explanation",
  "needs_enrichment": false,
  "enrichment_queries": null,
  "action_type": null,
  "action_recipient": null,
  "action_message": null
}}

EXAMPLES:

Input: "um I need to buy eggs and milk at the store"
Output: {{"category":"fact","intensity":"routine","confidence":0.9,"is_actionable":true,"task_title":"Buy eggs and milk","context":"grocery_store","due_day":null,"due_time":null,"cleaned_content":"I need to buy eggs and milk at the store.","reasoning":"Actionable shopping task with location context."}}

Input: "remind me to turn off the grill at 4 PM"
Output: {{"category":"fact","intensity":"urgent","confidence":0.95,"is_actionable":true,"task_title":"Turn off the grill","context":"home","due_day":"today","due_time":"16:00","cleaned_content":"Remind me to turn off the grill at 4 PM.","reasoning":"Time-sensitive safety reminder."}}

Input: "call the plumber on Monday morning"
Output: {{"category":"fact","intensity":"routine","confidence":0.9,"is_actionable":true,"task_title":"Call the plumber","context":"anywhere","due_day":"monday","due_time":"morning","cleaned_content":"Call the plumber on Monday morning.","reasoning":"Actionable task scheduled for Monday morning."}}

Input: "the API rate limiting problem is exactly like highway on-ramps with metered lights"
Output: {{"category":"pattern","intensity":"breakthrough","confidence":0.85,"is_actionable":false,"task_title":null,"context":null,"due_day":null,"due_time":null,"cleaned_content":"The API rate limiting problem is exactly like highway on-ramps with metered lights.","reasoning":"Analogy connecting API design to traffic engineering."}}

Input: "I talked to Sarah today and she said the project deadline moved to March"
Output: {{"category":"fact","intensity":"routine","confidence":0.8,"is_actionable":false,"task_title":null,"context":null,"due_day":null,"due_time":null,"cleaned_content":"I talked to Sarah today and she said the project deadline moved to March.","reasoning":"Factual update about project timeline from conversation.","needs_enrichment":false,"enrichment_queries":null}}

Input: "call AG Workers Monday morning to add the minivan please look up the AG Workers phone number"
Output: {{"category":"fact","intensity":"routine","confidence":0.9,"is_actionable":true,"task_title":"Call AG Workers to add the minivan","context":"anywhere","due_day":"monday","due_time":"morning","cleaned_content":"Call AG Workers Monday morning to add the minivan. Look up the AG Workers phone number.","reasoning":"Actionable task with explicit lookup request for phone number.","needs_enrichment":true,"enrichment_queries":["AG Workers phone number"],"action_type":null,"action_recipient":null,"action_message":null}}

Input: "email Janis that I'm going to be 5 minutes late to our call on Monday"
Output: {{"category":"fact","intensity":"routine","confidence":0.9,"is_actionable":true,"task_title":"Email Janis about being late to Monday call","context":"anywhere","due_day":null,"due_time":null,"cleaned_content":"Email Janis that I'm going to be 5 minutes late to our call on Monday.","reasoning":"Action to send an email about being late.","needs_enrichment":false,"enrichment_queries":null,"action_type":"send_email","action_recipient":"Janis","action_message":"I'm going to be 5 minutes late to our call on Monday"}}

Input: "text Mom I'm on my way home"
Output: {{"category":"fact","intensity":"routine","confidence":0.9,"is_actionable":true,"task_title":"Text Mom about being on the way home","context":"anywhere","due_day":null,"due_time":null,"cleaned_content":"Text Mom that I'm on my way home.","reasoning":"Action to send a text message.","needs_enrichment":false,"enrichment_queries":null,"action_type":"send_text","action_recipient":"Mom","action_message":"I'm on my way home"}}

SPARK TO CLASSIFY:
"{content}"
"""


# ── LLM Calling ──────────────────────────────────────────────

def _call_ollama(prompt: str, timeout: int = 60) -> Optional[str]:
    """Call Ollama with format:json for reliable JSON output."""
    import urllib.request
    import urllib.error

    try:
        from config import QUALITY_GATE_OLLAMA_URL, QUALITY_GATE_MODEL
    except ImportError:
        QUALITY_GATE_OLLAMA_URL = "http://gandalf.local:11434"
        QUALITY_GATE_MODEL = "gemma3:12b"

    url = f"{QUALITY_GATE_OLLAMA_URL.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": QUALITY_GATE_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _call_haiku(prompt: str) -> Optional[str]:
    """Call Claude Haiku as fallback."""
    try:
        import anthropic
        from config import get_anthropic_api_key

        api_key = get_anthropic_api_key()
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"Haiku call failed: {e}")
        return None


# ── JSON Parsing ─────────────────────────────────────────────

def _extract_json(raw: str) -> Optional[dict]:
    """Robust JSON extraction: direct parse → strip fences → find {...}."""
    if not raw:
        return None

    # Try 1: Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try 2: Strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try 3: Find first {...} block (greedy from first { to last })
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ── Content Hash (matches postgres_store pattern) ────────────

def _compute_content_hash(content: str) -> str:
    """SHA-256 of normalized content. Matches postgres_store._compute_content_hash."""
    normalized = " ".join(content.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Classification ───────────────────────────────────────────

def classify_spark(content: str) -> tuple[Optional[SparkClassification], str]:
    """Classify a spark via LLM. Returns (classification, llm_source) tuple."""
    prompt = _build_prompt(content)

    # Try Ollama first (with format:json)
    raw = _call_ollama(prompt)
    source = "ollama"

    if raw is None:
        logger.info("Ollama unavailable, trying Haiku fallback")
        raw = _call_haiku(prompt)
        source = "haiku"

    if raw is None:
        logger.warning("All LLM providers failed for spark classification")
        return None, source

    logger.info(f"LLM response ({source}): {raw[:200]}...")

    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning(f"Failed to extract JSON from LLM response: {raw[:200]}")
        return None, source

    try:
        classification = SparkClassification(**parsed)
        return classification, source
    except Exception as e:
        logger.warning(f"Pydantic validation failed: {e}")
        return None, source


# ── Enrichment Tools ────────────────────────────────────────

MAX_ENRICHMENT_ROUNDS = 5

# Tool definitions for LLM tool-calling (Anthropic format)
ENRICHMENT_TOOLS_ANTHROPIC = [
    {
        "name": "search_memory",
        "description": "Search stored knowledge and memories (Coppermind). Good for finding previously stored facts, contacts, preferences, and project details.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search Apple Contacts for a person or business. Returns phone numbers and email addresses.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web via DuckDuckGo. Good for finding phone numbers, addresses, business info, and general facts.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
]

# Tool definitions for Ollama format
ENRICHMENT_TOOLS_OLLAMA = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in ENRICHMENT_TOOLS_ANTHROPIC
]


def _tool_search_memory(query: str, mem) -> str:
    """Search Coppermind knowledge base via hybrid semantic + BM25."""
    try:
        results = mem.learning.postgres_store.search_knowledge(
            agent_name=mem.agent_name, query=query, limit=5, threshold=0.2
        )
        if not results:
            return "No results found in memory."
        return "\n".join(f"- {r['content']}" for r in results[:5])
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")
        return f"Memory search error: {e}"


def _tool_search_contacts(query: str) -> str:
    """Search Apple Contacts via SSH to ben.local."""
    # Sanitize query for AppleScript string literal
    safe_query = query.replace("\\", "\\\\").replace('"', '\\"')

    applescript = f'''tell application "Contacts"
    set matchingPeople to every person whose name contains "{safe_query}"
    set output to ""
    repeat with p in matchingPeople
        set pName to name of p
        set output to output & "Name: " & pName & linefeed
        repeat with ph in phones of p
            set output to output & "  Phone: " & (value of ph) & " (" & (label of ph) & ")" & linefeed
        end repeat
        repeat with em in emails of p
            set output to output & "  Email: " & (value of em) & " (" & (label of em) & ")" & linefeed
        end repeat
    end repeat
    return output
end tell'''

    try:
        # Pipe AppleScript via stdin to avoid shell parsing issues with
        # multiline scripts passed as SSH command arguments
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=3",
                "-o", "ServerAliveInterval=2",
                "-o", "ServerAliveCountMax=1",
                REMINDER_SSH_HOST,
                "osascript",
            ],
            input=applescript.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        output = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning(f"Contacts search failed: {stderr}")
            return f"Contacts search error: {stderr}"
        return output or f"No contacts found matching '{query}'."
    except subprocess.TimeoutExpired:
        logger.warning("Contacts SSH timed out (Mac may be asleep)")
        return "Could not search contacts — Mac is unreachable."
    except Exception as e:
        logger.warning(f"Contacts search failed: {e}")
        return f"Contacts search error: {e}"


def _tool_search_web(query: str) -> str:
    """Search the web via DuckDuckGo."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web results found."
        return "\n".join(
            f"- {r['title']}: {r['body']}" for r in results
        )
    except ImportError:
        logger.warning("ddgs package not installed")
        return "Web search unavailable (package not installed)."
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return f"Web search error: {e}"


def _tool_resolve_contact(name: str, contact_type: str = "any", mem=None) -> str:
    """Resolve a contact name to email/phone using memory + Apple Contacts.

    Multi-signal resolution:
    1. Coppermind memory — finds stored preferences, past interactions
    2. Apple Contacts — returns all matching contacts with phones + emails

    Args:
        name: Person's name to resolve
        contact_type: "email", "phone", or "any"
        mem: Coppermind memory object (optional)

    Returns:
        Combined results from memory + contacts for the enrichment agent to pick from.
    """
    results = []

    # Signal 1: Coppermind memory
    if mem:
        try:
            memory_results = mem.learning.postgres_store.search_knowledge(
                agent_name=mem.agent_name,
                query=f"{name} email phone contact",
                limit=5,
                threshold=0.2,
            )
            if memory_results:
                results.append("From memory:")
                for r in memory_results[:5]:
                    results.append(f"  - {r['content']}")
        except Exception as e:
            logger.warning(f"Memory search for contact failed: {e}")

    # Signal 2: Apple Contacts
    try:
        contacts_result = _tool_search_contacts(name)
        if contacts_result and "No contacts found" not in contacts_result:
            results.append("From Apple Contacts:")
            results.append(f"  {contacts_result}")
    except Exception as e:
        logger.warning(f"Contacts search for {name} failed: {e}")

    if not results:
        return f"No contact information found for '{name}'."
    return "\n".join(results)


# Tool definitions for contact resolution (added to enrichment tools)
RESOLVE_CONTACT_TOOL_ANTHROPIC = {
    "name": "resolve_contact",
    "description": "Resolve a person's name to their contact info (email, phone). "
                   "Searches stored memories and Apple Contacts. Returns all matches "
                   "for you to pick the best one.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name to look up"},
            "contact_type": {
                "type": "string",
                "description": "Type of contact info needed: 'email', 'phone', or 'any'",
                "default": "any",
            },
        },
        "required": ["name"],
    },
}

RESOLVE_CONTACT_TOOL_OLLAMA = {
    "type": "function",
    "function": {
        "name": RESOLVE_CONTACT_TOOL_ANTHROPIC["name"],
        "description": RESOLVE_CONTACT_TOOL_ANTHROPIC["description"],
        "parameters": RESOLVE_CONTACT_TOOL_ANTHROPIC["input_schema"],
    },
}


def _resolve_recipient_email(name: str, mem) -> Optional[dict]:
    """Run a focused agent loop to resolve a name to an email address.

    Returns {"email": "addr@example.com", "display_name": "Full Name"} or None.
    """
    system_prompt = (
        "You are resolving a contact's email address. Use the available tools to "
        "find the best email address for the given person. When you have found it, "
        "respond with ONLY a JSON object: {\"email\": \"addr@example.com\", \"display_name\": \"Full Name\"}. "
        "If you cannot find an email, respond with: {\"email\": null, \"display_name\": null}"
    )

    user_message = f"Find the email address for: {name}"
    messages = [{"role": "user", "content": user_message}]

    # Tools: resolve_contact + search_memory + search_contacts
    tools = ENRICHMENT_TOOLS_ANTHROPIC + [RESOLVE_CONTACT_TOOL_ANTHROPIC]

    tool_executors = {
        "search_memory": lambda args: _tool_search_memory(args["query"], mem),
        "search_contacts": lambda args: _tool_search_contacts(args["query"]),
        "search_web": lambda args: _tool_search_web(args["query"]),
        "resolve_contact": lambda args: _tool_resolve_contact(
            args["name"], args.get("contact_type", "any"), mem
        ),
    }

    try:
        result = _run_agent_loop_haiku(
            system_prompt, messages, tools, tool_executors,
        )
        if result:
            parsed = _extract_json(result)
            if parsed and parsed.get("email"):
                return parsed
    except Exception as e:
        logger.warning(f"Contact resolution failed: {e}")

    return None


def _send_email_via_mac(to_address: str, to_name: str, subject: str, body: str) -> bool:
    """SSH to ben.local and send an email via Mail.app helper script.

    Same SSH subprocess pattern as _create_apple_reminder().
    Returns True on success, False on failure.
    """
    payload = {
        "to_address": to_address,
        "to_name": to_name,
        "subject": subject,
        "body": body,
    }
    json_bytes = json.dumps(payload).encode("utf-8")

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=3",
                "-o", "ServerAliveInterval=2",
                "-o", "ServerAliveCountMax=1",
                REMINDER_SSH_HOST,
                "python3", EMAIL_SCRIPT,
            ],
            input=json_bytes,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"Email sent to {to_name} <{to_address}>: {subject}")
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning(f"Email script failed: {stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("SSH email timed out (Mac may be asleep)")
        return False
    except Exception as e:
        logger.warning(f"SSH email failed: {e}")
        return False


def _compose_and_send_email(content: str, classification, mem, learning_id: str) -> None:
    """Orchestrate the full email action flow.

    1. Resolve contact → find email address
    2. Compose email → Haiku generates subject + body
    3. Confirm → voice + dialog confirmation
    4. Send → SSH to Mac Mail.app
    5. Track → metadata updated at every step
    """
    from confirmation import confirm_action

    recipient_name = classification.action_recipient
    action_message = classification.action_message or content

    # Step 1: Resolve contact
    _update_metadata(mem, learning_id, {"action_status": "resolving_contact"})
    contact = _resolve_recipient_email(recipient_name, mem)

    if not contact or not contact.get("email"):
        logger.warning(f"Could not resolve email for '{recipient_name}'")
        _update_metadata(mem, learning_id, {
            "action_status": "contact_not_found",
            "action_recipient": recipient_name,
        })
        return

    to_email = contact["email"]
    to_name = contact.get("display_name", recipient_name)

    _update_metadata(mem, learning_id, {
        "action_status": "composing",
        "action_resolved_email": to_email,
        "action_resolved_name": to_name,
    })

    # Step 2: Compose email via Haiku
    compose_prompt = (
        f"Compose a brief, friendly email from Ben to {to_name}.\n"
        f"The message to convey: {action_message}\n"
        f"Original voice transcript: \"{content}\"\n\n"
        f"Respond with ONLY a JSON object:\n"
        f'{{"subject": "...", "body": "..."}}\n\n'
        f"Keep the email natural and conversational, matching a typical quick email tone. "
        f"Don't be overly formal. The body should be just the email text, no greeting/signature needed."
    )
    raw_email = _call_haiku(compose_prompt)
    if not raw_email:
        logger.warning("Email composition failed (Haiku unavailable)")
        _update_metadata(mem, learning_id, {"action_status": "compose_failed"})
        return

    email_data = _extract_json(raw_email)
    if not email_data or not email_data.get("subject"):
        logger.warning(f"Email composition returned invalid JSON: {raw_email[:200]}")
        _update_metadata(mem, learning_id, {"action_status": "compose_failed"})
        return

    subject = email_data["subject"]
    body = email_data["body"]

    _update_metadata(mem, learning_id, {
        "action_status": "confirming",
        "action_subject": subject,
    })

    # Step 3: Confirm with user
    confirmation_details = f"To: {to_name} <{to_email}>\nSubject: {subject}\n\n{body}"
    result = confirm_action(
        summary=f"Send email to {to_name}?",
        details=confirmation_details,
    )

    if result == "approved":
        # Step 4: Send
        _update_metadata(mem, learning_id, {"action_status": "sending"})
        sent = _send_email_via_mac(to_email, to_name, subject, body)
        if sent:
            _update_metadata(mem, learning_id, {"action_status": "sent"})
            logger.info(f"email_sent: id={learning_id} to={to_email}")
        else:
            _update_metadata(mem, learning_id, {"action_status": "send_failed"})
            logger.warning(f"email_send_failed: id={learning_id}")
    elif result == "rejected":
        _update_metadata(mem, learning_id, {"action_status": "cancelled"})
        logger.info(f"email_cancelled: id={learning_id}")
    elif result == "timeout":
        _update_metadata(mem, learning_id, {"action_status": "timeout"})
        logger.info(f"email_timeout: id={learning_id}")
    else:
        _update_metadata(mem, learning_id, {"action_status": "confirm_error"})
        logger.warning(f"email_confirm_error: id={learning_id} result={result}")


def _run_agent_loop_haiku(system_prompt: str, messages: list,
                          tools: list, tool_executors: dict) -> Optional[str]:
    """Run enrichment agent loop via Claude Haiku with native tool calling."""
    try:
        import anthropic
        from config import get_anthropic_api_key

        api_key = get_anthropic_api_key()
        if not api_key:
            return None

        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.warning(f"Haiku agent setup failed: {e}")
        return None

    for round_num in range(MAX_ENRICHMENT_ROUNDS):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            logger.warning(f"Haiku agent round {round_num} failed: {e}")
            return None

        # Check if the model wants to use tools
        if response.stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    executor = tool_executors.get(tool_name)
                    if executor:
                        try:
                            result = executor(tool_input)
                        except Exception as e:
                            result = f"Tool error: {e}"
                        logger.info(f"enrichment_tool: {tool_name}({tool_input}) → {result[:100]}...")
                    else:
                        result = f"Unknown tool: {tool_name}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append assistant response and tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Model is done — extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return None

    logger.warning("Haiku agent hit max rounds")
    return None


def _run_agent_loop_ollama(system_prompt: str, messages: list,
                           tools: list, tool_executors: dict) -> Optional[str]:
    """Run enrichment agent loop via Ollama (mistral-small:24b) with tool calling."""
    import urllib.request
    import urllib.error

    try:
        from config import QUALITY_GATE_OLLAMA_URL
    except ImportError:
        QUALITY_GATE_OLLAMA_URL = "http://gandalf.local:11434"

    url = f"{QUALITY_GATE_OLLAMA_URL.rstrip('/')}/api/chat"

    # Convert messages to Ollama format
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            ollama_messages.append({"role": "user", "content": msg["content"]})

    for round_num in range(MAX_ENRICHMENT_ROUNDS):
        payload = json.dumps({
            "model": "mistral-small:24b",
            "messages": ollama_messages,
            "tools": ENRICHMENT_TOOLS_OLLAMA,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Ollama agent round {round_num} failed: {e}")
            return None

        message = data.get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls:
            # Append assistant message
            ollama_messages.append(message)

            # Execute each tool call
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})
                executor = tool_executors.get(tool_name)
                if executor:
                    try:
                        result = executor(tool_args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.info(f"enrichment_tool: {tool_name}({tool_args}) → {result[:100]}...")
                else:
                    result = f"Unknown tool: {tool_name}"
                ollama_messages.append({"role": "tool", "content": result})
        else:
            # No tool calls — return the content
            content = message.get("content", "")
            return content if content else None

    logger.warning("Ollama agent hit max rounds")
    return None


def enrich_spark(content: str, classification, mem) -> Optional[str]:
    """Run agentic enrichment loop. Returns enrichment text or None.

    Tries Haiku first, falls back to Ollama mistral-small:24b.
    Enrichment failure never blocks task creation.
    """
    queries = classification.enrichment_queries or []
    queries_text = ", ".join(queries)

    system_prompt = (
        "You are an assistant that looks up information to enrich a task. "
        "Use the available tools to find the requested information. "
        "When you have found what you need, respond with a concise summary "
        "of the information found. Include specific details like phone numbers, "
        "email addresses, or other facts. If you cannot find the information, "
        "say so briefly."
    )

    user_message = (
        f"The user captured this thought:\n\"{content}\"\n\n"
        f"They asked to look up: {queries_text}\n\n"
        f"Please search for this information using the available tools."
    )

    messages = [{"role": "user", "content": user_message}]

    tool_executors = {
        "search_memory": lambda args: _tool_search_memory(args["query"], mem),
        "search_contacts": lambda args: _tool_search_contacts(args["query"]),
        "search_web": lambda args: _tool_search_web(args["query"]),
    }

    try:
        result = _run_agent_loop_haiku(
            system_prompt, list(messages),
            ENRICHMENT_TOOLS_ANTHROPIC, tool_executors,
        )
        if result:
            logger.info(f"enrichment_complete: source=haiku result={result[:100]}...")
            return result

        logger.info("Haiku enrichment failed, trying Ollama fallback")
        result = _run_agent_loop_ollama(
            system_prompt, list(messages),
            ENRICHMENT_TOOLS_OLLAMA, tool_executors,
        )
        if result:
            logger.info(f"enrichment_complete: source=ollama result={result[:100]}...")
            return result

        logger.warning("All enrichment providers failed")
        return None
    except Exception as e:
        logger.warning(f"Enrichment failed (non-fatal): {e}")
        return None


# ── Apple Reminders via SSH ──────────────────────────────────

def _create_apple_reminder(title: str, context: Optional[str] = None,
                           due_date: Optional[str] = None,
                           body: str = "") -> bool:
    """SSH to ben.local and create an Apple Reminder via helper script.

    Returns True on success, False on failure.
    """
    list_name = CONTEXT_TO_LIST.get(context, "Anywhere")

    payload = {
        "title": title,
        "list": list_name,
        "body": body,
    }
    if due_date:
        payload["due_date"] = due_date

    json_bytes = json.dumps(payload).encode("utf-8")

    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "ConnectTimeout=3",
                "-o", "ServerAliveInterval=2",
                "-o", "ServerAliveCountMax=1",
                REMINDER_SSH_HOST,
                "python3", REMINDER_SCRIPT,
            ],
            input=json_bytes,
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            logger.info(f"Apple Reminder created: '{title}' in '{list_name}'")
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            logger.warning(f"Reminder script failed: {stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("SSH to ben.local timed out (Mac may be asleep)")
        return False
    except Exception as e:
        logger.warning(f"SSH reminder failed: {e}")
        return False


def _create_coppermind_task(mem, title: str, body: str = "") -> bool:
    """Fallback: create a Coppermind task when Mac is unreachable."""
    try:
        task_content = title
        if body:
            task_content = f"{title} — {body}"
        mem.learning.postgres_store.store_knowledge(
            agent_name=mem.agent_name,
            category="fact",
            content=task_content,
            source="spark_evaluator",
            confidence=0.8,
            learning_intensity="routine",
            source_type="explicit_user",
            source_confidence=1.0,
            memory_type="task",
            metadata={"spark_task": True, "reminder_fallback": True},
            skip_quality_gates=True,
        )
        logger.info(f"Coppermind task created as reminder fallback: '{title}'")
        return True
    except Exception as e:
        logger.warning(f"Coppermind task fallback failed: {e}")
        return False


# ── Main Evaluation Pipeline ────────────────────────────────

def evaluate_and_update(learning_id: str, content: str, source: str, mem) -> None:
    """Background task: classify spark, update DB, route actionable items.

    This runs after the spark is already stored (store-first pattern).
    Failures are graceful — the spark stays with safe defaults.
    """
    logger.info(f"evaluation_started: id={learning_id} content={content[:50]}...")

    # Step 1: Classify via LLM
    classification, llm_source = classify_spark(content)

    if classification is None:
        logger.warning(f"evaluation_failed: id={learning_id} reason=llm_failed")
        # Mark as evaluated (failed) so we don't retry forever
        _update_metadata(mem, learning_id, {"evaluated": True, "eval_failed": True})
        return

    logger.info(
        f"llm_response: id={learning_id} category={classification.category} "
        f"intensity={classification.intensity} actionable={classification.is_actionable} "
        f"confidence={classification.confidence} source={llm_source}"
    )

    # Step 2: Cap confidence (LLM overconfident by 15-25%)
    db_confidence = min(0.7, classification.confidence * 0.8)

    # Step 3: Determine cleaned content
    cleaned = classification.cleaned_content or content

    # Step 4: Recompute embedding + content_hash from cleaned content
    try:
        from lib.postgres_store import compute_embedding
        new_embedding = compute_embedding(cleaned)
    except Exception as e:
        logger.warning(f"Embedding computation failed: {e}, keeping original")
        new_embedding = None

    new_content_hash = _compute_content_hash(cleaned)

    # Step 5: Build metadata update
    metadata_update = {
        "evaluated": True,
        "eval_source": llm_source,
        "eval_category": classification.category,
        "eval_reasoning": classification.reasoning,
        "is_actionable": classification.is_actionable,
    }

    # Resolve due date from LLM's due_day + due_time (Python does the math)
    resolved_due_date = resolve_due_date(classification.due_day, classification.due_time)
    # Fall back to LLM's due_date if it provided one and we couldn't resolve
    if not resolved_due_date and classification.due_date:
        resolved_due_date = classification.due_date

    if resolved_due_date:
        logger.info(f"resolved_due_date: id={learning_id} "
                     f"due_day={classification.due_day} due_time={classification.due_time} "
                     f"→ {resolved_due_date}")

    if classification.is_actionable and classification.task_title:
        metadata_update["task_title"] = classification.task_title
        metadata_update["task_context"] = classification.context
        if resolved_due_date:
            metadata_update["task_due_date"] = resolved_due_date

    # Step 6: Atomic DB update
    try:
        _update_spark_row(
            mem=mem,
            learning_id=learning_id,
            content=cleaned,
            category=classification.category,
            intensity=classification.intensity,
            confidence=db_confidence,
            content_hash=new_content_hash,
            embedding=new_embedding,
            evergreen=classification.is_actionable,
            metadata_update=metadata_update,
        )
    except Exception as e:
        logger.error(f"DB update failed: id={learning_id} error={e}")
        return

    # Step 6.5: Agentic enrichment (if requested)
    enrichment_text = None
    if classification.needs_enrichment and classification.enrichment_queries:
        try:
            enrichment_text = enrich_spark(content, classification, mem)
            _update_metadata(mem, learning_id, {
                "enrichment_result": enrichment_text,
                "enrichment_status": "success" if enrichment_text else "no_results",
                "enrichment_queries": classification.enrichment_queries,
            })
        except Exception as e:
            logger.warning(f"Enrichment failed (non-fatal): id={learning_id} error={e}")
            _update_metadata(mem, learning_id, {
                "enrichment_status": "error",
                "enrichment_queries": classification.enrichment_queries,
            })

    # Step 6.7: Action execution (email, text, Slack)
    if classification.action_type and classification.action_recipient:
        _update_metadata(mem, learning_id, {
            "action_type": classification.action_type,
            "action_recipient": classification.action_recipient,
        })
        if classification.action_type == "send_email":
            try:
                _compose_and_send_email(content, classification, mem, learning_id)
            except Exception as e:
                logger.warning(f"Email action failed (non-fatal): id={learning_id} error={e}")
                _update_metadata(mem, learning_id, {"action_status": "error"})
        elif classification.action_type in ("send_text", "send_slack"):
            _update_metadata(mem, learning_id, {"action_status": "not_implemented"})
            logger.info(f"action_not_implemented: id={learning_id} type={classification.action_type}")
        # Fall through to reminder creation regardless

    # Step 7: Route actionable items to Apple Reminders
    if classification.is_actionable and classification.task_title:
        reminder_body = cleaned if cleaned != classification.task_title else ""
        if enrichment_text:
            reminder_body = f"{reminder_body}\n\n{enrichment_text}".strip()
        # Annotate if an email action was completed
        if classification.action_type == "send_email":
            # Check action_status from metadata to annotate reminder
            try:
                conn = mem.learning.postgres_store._get_conn()
                cur = conn.cursor()
                try:
                    cur.execute(
                        "SELECT metadata->>'action_status' FROM agent_knowledge WHERE id = %s",
                        (learning_id,),
                    )
                    row = cur.fetchone()
                    action_status = row[0] if row else None
                    if action_status == "sent":
                        reminder_body = f"[EMAIL SENT]\n{reminder_body}".strip()
                    elif action_status in ("cancelled", "timeout"):
                        reminder_body = f"[EMAIL {action_status.upper()} — action needed]\n{reminder_body}".strip()
                finally:
                    cur.close()
            except Exception:
                pass  # Non-fatal — reminder still created
        success = _create_apple_reminder(
            title=classification.task_title,
            context=classification.context,
            due_date=resolved_due_date,
            body=reminder_body,
        )

        if not success:
            # Fallback to Coppermind task
            _create_coppermind_task(mem, classification.task_title, reminder_body)
            _update_metadata(mem, learning_id, {"reminder_failed": True})
            logger.info(f"reminder_fallback: id={learning_id} used coppermind task")
        else:
            logger.info(f"reminder_created: id={learning_id} title='{classification.task_title}'")

    logger.info(f"evaluation_complete: id={learning_id}")


# ── DB Update Helpers ────────────────────────────────────────

def _update_spark_row(mem, learning_id: str, content: str, category: str,
                      intensity: str, confidence: float, content_hash: str,
                      embedding: Optional[list], evergreen: bool,
                      metadata_update: dict) -> None:
    """Atomically update a spark row with classification results."""
    conn = mem.learning.postgres_store._get_conn()
    cur = conn.cursor()

    try:
        # Build the UPDATE dynamically based on what we have
        set_clauses = [
            "content = %s",
            "category = %s",
            "learning_intensity = %s",
            "confidence = %s",
            "content_hash = %s",
        ]
        params = [content, category, intensity, confidence, content_hash]

        if embedding is not None:
            set_clauses.append("embedding = %s")
            params.append(embedding)

        if evergreen:
            set_clauses.append("evergreen = true")

        # Merge metadata
        set_clauses.append(
            "metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb"
        )
        params.append(json.dumps(metadata_update))

        params.append(learning_id)

        sql = f"""
            UPDATE agent_knowledge
            SET {', '.join(set_clauses)}
            WHERE id = %s
        """
        cur.execute(sql, params)
        conn.commit()

        if cur.rowcount == 0:
            logger.warning(f"No row found for learning_id={learning_id}")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _update_metadata(mem, learning_id: str, metadata_update: dict) -> None:
    """Update only metadata fields on a spark row."""
    try:
        conn = mem.learning.postgres_store._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE agent_knowledge
            SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            """,
            (json.dumps(metadata_update), learning_id),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"Metadata update failed: id={learning_id} error={e}")


# ── Startup Helpers ──────────────────────────────────────────

def get_pending_sparks(mem, min_age_seconds: int = 120) -> list:
    """Find sparks that haven't been evaluated yet (older than min_age_seconds)."""
    try:
        conn = mem.learning.postgres_store._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, content, source
            FROM agent_knowledge
            WHERE memory_type = 'spark'
              AND (metadata->>'evaluated')::boolean IS NOT TRUE
              AND created_at < NOW() - (%s * INTERVAL '1 second')
            ORDER BY created_at ASC
            LIMIT 50
            """,
            (min_age_seconds,),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"id": str(r[0]), "content": r[1], "source": r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"Failed to query pending sparks: {e}")
        return []


def prewarm_ollama() -> None:
    """Send a trivial prompt to load gemma3:12b into VRAM."""
    import urllib.request
    import urllib.error

    try:
        from config import QUALITY_GATE_OLLAMA_URL, QUALITY_GATE_MODEL
    except ImportError:
        QUALITY_GATE_OLLAMA_URL = "http://gandalf.local:11434"
        QUALITY_GATE_MODEL = "gemma3:12b"

    url = f"{QUALITY_GATE_OLLAMA_URL.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": QUALITY_GATE_MODEL,
        "prompt": "Hello",
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        logger.info(f"Ollama pre-warm complete: {QUALITY_GATE_MODEL}")
    except Exception as e:
        logger.warning(f"Ollama pre-warm failed (non-fatal): {e}")
