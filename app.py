import os, json
from time import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict
import requests

app = FastAPI(title="Agent Brain for Samaira’s", version="0.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Business data (demo) ---
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Monday to Saturday 10am–6pm; Sunday Closed"
PRICING = "$80 to $1100"
POLICY = "24 hour cancel; $25 late; 50% no-show; deposits for groups"

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

# --- Simple FAQ logic ---
def rule_based_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["hour", "open", "close", "when are you open"]):
        return f"We’re open {HOURS}."
    if any(k in t for k in ["price", "how much", "cost", "rate"]):
        return f"Our pricing ranges are {PRICING}."
    if any(k in t for k in ["service", "treatment", "what do you offer", "menu"]):
        return ("We offer massages, facials, body scrubs, wellness packages, "
                "and relaxation therapies. Would you like details on a specific service?")
    if any(k in t for k in ["location", "where are you", "address", "directions"]):
        return "We are located in New Hyde Park, NY."
    if any(k in t for k in ["book", "appointment", "schedule", "reserve"]):
        return "I can request a booking. May I have your full name, phone, email, the service you want, and a preferred day/time window?"
    if any(k in t for k in ["cancel", "refund", "policy", "late"]):
        return f"Our policy is: {POLICY}. Would you like me to email the full policy?"
    if any(k in t for k in ["insurance", "medical", "diagnosis"]):
        return "I’m not able to advise on medical or insurance matters. I can transfer you to a team member if you’d like."
    if any(k in t for k in ["payment", "card", "cash", "pay"]):
        return "We accept credit cards, debit cards, and cash."
    if any(k in t for k in ["gift card", "voucher", "certificate"]):
        return "Yes, we offer gift cards for all services and packages. They make a great present!"
    return "I can help with hours, services, pricing, booking, policies, payments, and gift cards. What would you like to know?"

# --- Optional Ollama ---
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

# --- Manual test endpoint ---
@app.post("/agent")
async def agent(req: AgentRequest):
    user_text = (req.input or "").strip()
    if not user_text:
        return {"output": "Hello! How can I help you today?"}
    return {"output": generate_reply(user_text)}

# --- Twilio entrypoint: returns TwiML to start audio stream ---
@app.post("/twilio-voice")
def twilio_voice():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://samaras-agent-backend.onrender.com/twilio-stream"/>
  </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")

# --- WebSocket endpoint: Twilio streams audio here ---
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            # Right now we just log/ack — integration with ElevenLabs realtime comes next
            print("Twilio stream frame:", msg[:100])  # preview first chars
            await ws.send_text('{"event":"keepalive"}')
    except WebSocketDisconnect:
        print("Twilio stream disconnected")

# --- OpenAI-compatible endpoints (for ElevenLabs web demo) ---
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

    def gen():
        chunk = {
            "id": "chatcmpl-demo",
            "object": "chat.completion.chunk",
            "created": int(time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": reply},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
