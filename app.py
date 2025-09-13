import os, json
from time import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
import requests

app = FastAPI(title="Agent Brain for Samaira’s", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Simple business data (demo) ---
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Mon–Sat 10am–6pm; Sun Closed"
PRICING = "$80 to $1100"
POLICY = "24h cancel; $25 late; 50% no-show; deposits for groups"

# --- Optional local LLM (Ollama) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
USE_OLLAMA = os.getenv("USE_OLLAMA", "auto")  # 'yes'|'no'|'auto'

class AgentRequest(BaseModel):
    input: str
    conversation: dict | None = None
    model_config = ConfigDict(extra='ignore')

@app.get("/ping")
def ping():
    return {"ok": True, "service": "agent-brain", "business": BIZ_NAME}

@app.get("/agent")
def agent_get():
    return {"output": "Hello from Samaira’s backend."}

def rule_based_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["hour", "open", "close", "when are you open"]):
        return f"We’re open {HOURS}."
    if any(k in t for k in ["price", "how much", "cost", "rate"]):
        return f"Our pricing ranges are {PRICING}."
    if any(k in t for k in ["book", "appointment", "schedule", "reserve"]):
        return "I can request a booking. May I have your full name, phone, email, the service you want, and a preferred day/time window?"
    if any(k in t for k in ["cancel", "refund", "policy", "late"]):
        return f"Our policy is: {POLICY}. Would you like me to email the full policy?"
    if any(k in t for k in ["insurance", "medical", "diagnosis"]):
        return "I’m not able to advise on medical or insurance matters. I can transfer you to a team member if you’d like."
    return "I can help with hours, pricing, services, and booking requests. What would you like to know?"

def ollama_reply(text: str) -> str:
    system = ("You are Sarah, a calm, warm, helpful assistant for a spa. "
              "Answer briefly (1–2 sentences). If outside scope, offer to transfer to a human.")
    prompt = f"System: {system}\nUser: {text}\nAssistant:"
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=10)
        r.raise_for_status()
        out = (r.json().get("response") or "").strip()
        return out or rule_based_reply(text)
    except Exception:
        return rule_based_reply(text)

def generate_reply(user_text: str) -> str:
    if USE_OLLAMA == "yes":
        return ollama_reply(user_text)
    if USE_OLLAMA == "no":
        return rule_based_reply(user_text)
    try:
        if requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok:
            return ollama_reply(user_text)
    except Exception:
        pass
    return rule_based_reply(user_text)

# Manual test endpoint
@app.post("/agent")
async def agent(req: AgentRequest):
    user_text = (req.input or "").strip()
    if not user_text:
        return {"output": "Hello! How can I help you today?"}
    return {"output": generate_reply(user_text)}

# ===== OpenAI-compatible endpoints for ElevenLabs =====
@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "samaira-agent", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    messages = payload.get("messages") or []
    user_text = ""
    for m in reversed(messages):
        if m and m.get("role") == "user":
            user_text = (m.get("content") or "").strip()
            break
    reply = generate_reply(user_text)
    model = payload.get("model", "samaira-agent")
    stream = bool(payload.get("stream"))

    if not stream:
        return {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": int(time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())}
        }

    # SSE streaming
    def gen():
        chunk = {
            "id": "chatcmpl-demo",
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
