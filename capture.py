"""Coppermind Capture API — store sparks from anywhere."""

import base64 as b64_module
import hashlib
import os
import time
from typing import List, Optional
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.environ["CAPTURE_API_KEY"]
COPPERMIND_PATH = os.environ.get("COPPERMIND_PATH", "/workspace/coppermind")

# Add Coppermind to import path
sys.path.insert(0, COPPERMIND_PATH)

from evaluator import evaluate_and_update, get_pending_sparks, prewarm_ollama

mem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mem
    from memory import PersonalMemory

    mem = PersonalMemory()

    # Pre-warm Ollama (background thread so startup isn't blocked)
    threading.Thread(target=prewarm_ollama, daemon=True).start()

    # Startup sweep: re-evaluate pending sparks older than 2 minutes
    # Use bounded pool to avoid overwhelming Ollama/DB with concurrent requests
    pending = get_pending_sparks(mem, min_age_seconds=120)
    if pending:
        logger.info(f"Startup sweep: {len(pending)} pending sparks to evaluate")
        pool = ThreadPoolExecutor(max_workers=3)
        for spark in pending:
            pool.submit(evaluate_and_update,
                        spark["id"], spark["content"], spark["source"], mem)

    yield


app = FastAPI(
    title="Coppermind Capture",
    description="Store sparks into your Coppermind from Alexa, Siri, or anywhere.",
    version="0.2.0",
    lifespan=lifespan,
)

VALID_CATEGORIES = {"fact", "pattern", "preference", "decision", "gotcha"}
VALID_INTENSITIES = {"routine", "urgent", "error_recovery", "breakthrough", "critical_failure"}


class Spark(BaseModel):
    content: str
    category: str = "fact"
    intensity: str = "routine"
    source: str = "capture_api"


def verify_key(x_api_key: str):
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _check_server_rate_limit(r_client, key_prefix: str = "hermes:rate", limit: int = 30, window: int = 60):
    """Redis sliding window rate limiter. Returns (allowed, retry_after_secs)."""
    try:
        import time as _time
        now = _time.time()
        bucket = f"{key_prefix}:{int(now // window)}"
        pipe = r_client.pipeline()
        pipe.incr(bucket)
        pipe.expire(bucket, window * 2)
        results = pipe.execute()
        count = results[0]
        if count > limit:
            retry_after = window - (now % window)
            return False, int(retry_after) + 1
        return True, 0
    except Exception as e:
        logger.warning(f"Rate limit check failed (allowing): {e}")
        return True, 0


def _compute_content_hash(content: str) -> str:
    """SHA-256 of normalized content for dedup."""
    normalized = " ".join(content.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@app.post("/store")
def store(spark: Spark, background_tasks: BackgroundTasks,
          x_api_key: str = Header(default=None)):
    """Store a spark in your Coppermind."""
    logger.info(f"Received spark: content={spark.content[:50]}... category={spark.category} source={spark.source}")
    verify_key(x_api_key)

    if spark.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Use: {VALID_CATEGORIES}")

    if spark.intensity not in VALID_INTENSITIES:
        raise HTTPException(status_code=400, detail=f"Invalid intensity. Use: {VALID_INTENSITIES}")

    learning_id = _store_spark(spark.content, spark.source)

    # Queue background evaluation
    if learning_id and learning_id != "None":
        background_tasks.add_task(
            evaluate_and_update, learning_id, spark.content, spark.source, mem
        )

    return {"id": learning_id, "stored": learning_id != "None"}


def _store_spark(content: str, source: str) -> str:
    """Shared storage logic for all capture sources.

    Stores original transcript and content hash in metadata,
    marks as unevaluated for background processing.
    """
    original_hash = _compute_content_hash(content)

    learning_id = mem.learning.postgres_store.store_knowledge(
        agent_name=mem.agent_name,
        category="fact",
        content=content,
        source=source,
        confidence=0.8,
        learning_intensity="routine",
        source_type="explicit_user",
        source_confidence=1.0,
        memory_type="spark",
        metadata={
            "spark": True,
            "capture_source": source,
            "original_transcript": content,
            "original_content_hash": original_hash,
            "evaluated": False,
        },
        skip_quality_gates=True,
    )
    return str(learning_id)


@app.post("/alexa")
async def alexa(request: Request, background_tasks: BackgroundTasks):
    """Alexa skill endpoint."""
    body = await request.json()
    request_type = body.get("request", {}).get("type", "")
    logger.info(f"Alexa request_type={request_type}")
    if request_type == "IntentRequest":
        intent_name = body["request"]["intent"]["name"]
        slots = body["request"]["intent"].get("slots", {})
        logger.info(f"Alexa intent={intent_name} slots={slots}")

    if request_type == "LaunchRequest":
        return JSONResponse(content={
            "version": "1.0",
            "response": {
                "outputSpeech": {"type": "PlainText", "text": "Say capture, then your thought."},
                "reprompt": {"outputSpeech": {"type": "PlainText", "text": "Say capture, then what you want to save."}},
                "shouldEndSession": False,
            },
        })

    if request_type == "IntentRequest":
        intent = body["request"]["intent"]["name"]

        if intent == "CaptureSparkIntent":
            slots = body["request"]["intent"].get("slots", {})
            content = slots.get("spark_content", {}).get("value", "")

            if not content:
                return JSONResponse(content={
                    "version": "1.0",
                    "response": {
                        "outputSpeech": {"type": "PlainText", "text": "I didn't catch that. Say capture, then your thought."},
                        "reprompt": {"outputSpeech": {"type": "PlainText", "text": "Say capture, then what you want to save."}},
                        "shouldEndSession": False,
                    },
                })

            logger.info(f"Alexa spark: {content[:50]}...")
            learning_id = _store_spark(content, "alexa")

            # Queue background evaluation
            if learning_id and learning_id != "None":
                background_tasks.add_task(
                    evaluate_and_update, learning_id, content, "alexa", mem
                )

            # Echo back the spark content
            echo = content[:80]
            return JSONResponse(content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": f"Got it. {echo}"},
                    "shouldEndSession": True,
                },
            })

        if intent in ("AMAZON.CancelIntent", "AMAZON.StopIntent"):
            return JSONResponse(content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": "Goodbye."},
                    "shouldEndSession": True,
                },
            })

        if intent == "AMAZON.FallbackIntent":
            return JSONResponse(content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": "I didn't understand. Say capture, then your thought."},
                    "reprompt": {"outputSpeech": {"type": "PlainText", "text": "Say capture, then what you want to save."}},
                    "shouldEndSession": False,
                },
            })

        # Any other intent — log it so we can debug
        logger.info(f"Alexa unhandled intent: {intent_name}")
        return JSONResponse(content={
            "version": "1.0",
            "response": {
                "outputSpeech": {"type": "PlainText", "text": "I didn't catch that. Say capture, then your thought."},
                "reprompt": {"outputSpeech": {"type": "PlainText", "text": "Say capture, then what you want to save."}},
                "shouldEndSession": False,
            },
        })

    # SessionEndedRequest or unknown
    return JSONResponse(content={
        "version": "1.0",
        "response": {"shouldEndSession": True},
    })


@app.post("/evaluate-pending")
def evaluate_pending(background_tasks: BackgroundTasks,
                     x_api_key: str = Header(default=None)):
    """Admin endpoint: re-evaluate sparks that failed or were missed."""
    verify_key(x_api_key)

    pending = get_pending_sparks(mem, min_age_seconds=0)
    for spark in pending:
        background_tasks.add_task(
            evaluate_and_update, spark["id"], spark["content"], spark["source"], mem
        )

    return {"queued": len(pending)}


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


# ── Hermes Messaging Endpoint ───────────────────────────────


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class ImageData(BaseModel):
    media_type: str
    data: str


class MessageRequest(BaseModel):
    content: str
    channel: str = "imessage"
    model_override: Optional[str] = None
    images: Optional[List[ImageData]] = None


class MessageResponse(BaseModel):
    reply: str
    model_used: str
    cost_usd: float
    intent: Optional[str] = None


@app.post("/message", response_model=MessageResponse)
def handle_message(req: MessageRequest, x_api_key: str = Header(default=None)):
    """Process an incoming message through Hermes and return a reply."""
    logger.info(f"Hermes message: channel={req.channel} content={req.content[:80]}... images={len(req.images) if req.images else 0}")
    verify_key(x_api_key)

    # Server-side rate limiting
    try:
        from pathlib import Path
        password = Path("~/.redis-password").expanduser().read_text().strip()
        import redis
        r_client = redis.Redis(
            host=os.environ.get("COPPERMIND_REDIS_HOST", "100.64.219.124"),
            port=6379, password=password, decode_responses=True,
            socket_timeout=2, socket_connect_timeout=2,
        )
        allowed, retry_after = _check_server_rate_limit(r_client)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )
    except Exception as e:
        logger.warning(f"Rate limit setup failed (allowing): {e}")

    # Validate images if present
    images = None
    if req.images:
        if len(req.images) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 images per message")
        validated = []
        for img in req.images:
            if img.media_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(status_code=400, detail=f"Unsupported image type: {img.media_type}. Allowed: {ALLOWED_IMAGE_TYPES}")
            if len(img.data) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image base64 data exceeds 5MB limit")
            try:
                b64_module.b64decode(img.data, validate=True)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
            validated.append({"media_type": img.media_type, "data": img.data})
        images = validated

    from hermes import process_message
    result = process_message(req.content, req.channel, req.model_override, mem, images=images)

    return MessageResponse(**result)


SEND_ALLOWED_RECIPIENTS = {"coppermindvolacci@icloud.com"}


class SendRequest(BaseModel):
    recipient: str
    text: str
    source: str = "api"


@app.post("/send")
def send_message(req: SendRequest, x_api_key: str = Header(default=None)):
    """Send a proactive iMessage to Ben via the Mac relay."""
    logger.info(f"Proactive send: recipient={req.recipient} source={req.source} text={req.text[:80]}...")
    verify_key(x_api_key)

    if req.recipient not in SEND_ALLOWED_RECIPIENTS:
        raise HTTPException(status_code=403, detail=f"Recipient not in allowlist")

    from hermes import send_imessage_proactive
    success = send_imessage_proactive(req.recipient, req.text)

    if not success:
        raise HTTPException(status_code=502, detail="Failed to send iMessage via Mac relay")

    return {"sent": True, "recipient": req.recipient}
