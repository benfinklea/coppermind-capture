"""Coppermind Capture API — store sparks from anywhere."""

import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
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

mem = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mem
    from memory import PersonalMemory

    mem = PersonalMemory()
    yield


app = FastAPI(
    title="Coppermind Capture",
    description="Store sparks into your Coppermind from Alexa, Siri, or anywhere.",
    version="0.1.0",
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


@app.post("/store")
def store(spark: Spark, x_api_key: str = Header(default=None)):
    """Store a spark in your Coppermind."""
    logger.info(f"Received spark: content={spark.content[:50]}... category={spark.category} source={spark.source}")
    verify_key(x_api_key)

    if spark.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Use: {VALID_CATEGORIES}")

    if spark.intensity not in VALID_INTENSITIES:
        raise HTTPException(status_code=400, detail=f"Invalid intensity. Use: {VALID_INTENSITIES}")

    learning_id = _store_spark(spark.content, spark.source)
    return {"id": learning_id, "stored": learning_id != "None"}


def _store_spark(content: str, source: str) -> str:
    """Shared storage logic for all capture sources."""
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
        metadata={"spark": True, "capture_source": source},
        skip_quality_gates=True,
    )
    return str(learning_id)


@app.post("/alexa")
async def alexa(request: Request):
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

            return JSONResponse(content={
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "PlainText", "text": "Stored."},
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


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
