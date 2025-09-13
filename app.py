import os, json, base64, asyncio
from time import time
import audioop
import requests
import websockets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Agent Brain for Samaira’s", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Demo business data ----
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Monday to Saturday 10am–6pm; Sunday Closed"
PRICING = "$80 to $1100"
POLICY = "Free 24 hour cancellation; $25 late; 50% no-show; deposits for groups"

# ---- Optional local LLM (Ollama) ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
USE_OLLAMA = os.getenv("USE_OLLAMA", "auto")  # 'yes'|'no'|'auto'

# ---- ElevenLabs Realtime ----
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY")
ELEVEN_AGENT_ID = os.getenv("ELEVEN_AGENT_ID")
ELEVEN_WS       = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVEN_AGENT_ID}"

def get_eleven_ws():
    """Try to fetch signed WS URL for private agents, fallback to static WS."""
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
            params={"agent_id": ELEVEN_AGENT_ID},
            headers={"xi-api-key": ELEVEN_API_KEY},
            timeout=5,
        )
        if r.ok and r.json().get("signed_url"):
            return r.json()["signed_url"]
    except Exception:
        pass
    return ELEVEN_WS

class AgentRequest(BaseModel):
    input: str
    conversation: dict | None = None
    model_config = ConfigDict(extra='ignore')

# ---------- Health / simple test ----------
@app.get("/ping")
def ping():
    return {"ok": True, "service": "agent-brain", "business": BIZ_NAME}

@app.get("/agent")
def agent_get():
    return {"output": "Hello from Samaira’s backend."}

# ---------- Simple FAQ logic ----------
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

# ---------- Manual POST test ----------
@app.post("/agent")
async def agent(req: AgentRequest):
    user_text = (req.input or "").strip()
    if not user_text:
        return {"output": "Hello! How can I help you today?"}
    return {"output": generate_reply(user_text)}

# ---------- Twilio: return TwiML to start media stream ----------
@app.post("/twilio-voice")
def twilio_voice():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://samaras-agent-backend.onrender.com/twilio-stream"
            track="inbound_audio"
            statusCallbackEvent="start mark stop"/>
  </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")

# ---------- Twilio <Stream> ⇄ ElevenLabs Realtime bridge ----------
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    try:
        async with websockets.connect(get_eleven_ws()) as elws:
            samples_accum = bytearray()
            last_commit = asyncio.get_event_loop().time()

            async def twilio_to_eleven():
                nonlocal samples_accum, last_commit
                while True:
                    msg = await ws.receive_text()
                    data = json.loads(msg)
                    ev = data.get("event")

                    if ev == "start":
                        await ws.send_text(json.dumps({"event": "mark", "mark": {"name": "started"}}))
                        continue

                    if ev == "media":
                        b64 = data["media"]["payload"]
                        mulaw_8k = base64.b64decode(b64)
                        pcm16_8k = audioop.ulaw2lin(mulaw_8k, 2)
                        pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)
                        samples_accum.extend(pcm16_16k)

                        now = asyncio.get_event_loop().time()
                        if len(samples_accum) > 3200 or (now - last_commit) > 0.12:
                            chunk_b64 = base64.b64encode(bytes(samples_accum)).decode()
                            await elws.send(json.dumps({
                                "user_audio_chunk": chunk_b64
                            }))
                            samples_accum.clear()
                            last_commit = now

                    elif ev == "stop":
                        if samples_accum:
                            chunk_b64 = base64.b64encode(bytes(samples_accum)).decode()
                            await elws.send(json.dumps({
                                "user_audio_chunk": chunk_b64
                            }))
                            samples_accum.clear()
                        break

            async def eleven_to_twilio():
                async for raw in elws:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    if data.get("type") == "audio" and "audio_event" in data:
                        b64 = data["audio_event"].get("audio_base_64")
                        if not b64:
                            continue
                        pcm16_16k = base64.b64decode(b64)
                        pcm16_8k, _ = audioop.ratecv(pcm16_16k, 2, 1, 16000, 8000, None)
                        mulaw_8k = audioop.lin2ulaw(pcm16_8k, 2)
                        payload = base64.b64encode(mulaw_8k).decode()
                        await ws.send_text(json.dumps({"event": "media", "media": {"payload": payload}}))

            async def keepalive():
                while True:
                    await asyncio.sleep(8)
                    await ws.send_text(json.dumps({"event": "mark", "mark": {"name": "keepalive"}}))

            await asyncio.gather(twilio_to_eleven(), eleven_to_twilio(), keepalive())

    except WebSocketDisconnect:
        return
    except Exception as e:
        print("Stream bridge error:", e)
        return

# ---------- OpenAI-compatible endpoints ----------
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
