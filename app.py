import os, json, base64, asyncio
from time import time
import audioop
import requests
import websockets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Agent Brain for Samaira’s", version="1.3-stable")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------- Business data (demo) ----------
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Monday to Saturday 10am–6pm; Sunday Closed"
PRICING = "$80 to $1100"
POLICY = "24h cancel; $25 late; 50% no-show; deposits for groups"

# --------- Optional local LLM (unused by phone path, safe to keep) ----------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
USE_OLLAMA = os.getenv("USE_OLLAMA", "auto")  # 'yes'|'no'|'auto'

# --------- ElevenLabs realtime config ----------
ELEVEN_API_KEY  = os.getenv("ELEVEN_API_KEY")
ELEVEN_AGENT_ID = os.getenv("ELEVEN_AGENT_ID")
ELEVEN_WS_FALLBACK = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVEN_AGENT_ID}"

def get_eleven_ws():
    """Prefer signed URL (private agents) else static WS."""
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
            params={"agent_id": ELEVEN_AGENT_ID},
            headers={"xi-api-key": ELEVEN_API_KEY},
            timeout=6,
        )
        j = r.json() if r.ok else {}
        if j.get("signed_url"):
            return j["signed_url"]
    except Exception as e:
        print("signed-url fetch failed:", e)
    return ELEVEN_WS_FALLBACK

# ---------- health & simple chat ----------
class AgentRequest(BaseModel):
    input: str
    conversation: dict | None = None
    model_config = ConfigDict(extra='ignore')

@app.get("/ping")
def ping():
    return {"ok": True, "service": "agent-brain", "business": BIZ_NAME}

def rule_based_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["hour", "open", "close", "when are you open"]):
        return f"We’re open {HOURS}."
    if any(k in t for k in ["price", "how much", "cost", "rate"]):
        return f"Our pricing ranges are {PRICING}."
    if any(k in t for k in ["service", "treatment", "what do you offer", "menu"]):
        return ("We offer massages, facials, body scrubs, wellness packages, "
                "and relaxation therapies. Want details on something specific?")
    if any(k in t for k in ["location", "where are you", "address", "directions"]):
        return "We are in New Hyde Park, NY."
    if any(k in t for k in ["book", "appointment", "schedule", "reserve"]):
        return "I can request a booking. May I have your name, phone, email, service, and preferred time?"
    if any(k in t for k in ["cancel", "refund", "policy", "late"]):
        return f"Our policy: {POLICY}."
    return "I can help with hours, services, pricing, booking, and policies. What would you like to know?"

@app.post("/agent")
async def agent(req: AgentRequest):
    txt = (req.input or "").strip()
    return {"output": rule_based_reply(txt) if txt else "Hello! How can I help you today?"}

# ---------- Twilio: return TwiML to start media stream ----------
@app.post("/twilio-voice")
def twilio_voice():
    # No track attribute; Twilio chooses valid track automatically.
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://samaras-agent-backend.onrender.com/twilio-stream"/>
  </Connect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")

# ---------- Twilio <Stream> ⇄ ElevenLabs Realtime bridge ----------
@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    # Twilio requires subprotocol 'audio' for Media Streams
    await ws.accept(subprotocol="audio")
    print("Twilio WS accepted")

    # Connect to ElevenLabs
    eleven_ws_url = get_eleven_ws()
    print("Connecting to ElevenLabs WS:", eleven_ws_url)
    try:
        async with websockets.connect(eleven_ws_url, ping_interval=15, ping_timeout=30) as elws:
            print("ElevenLabs WS connected")
            samples_accum = bytearray()
            last_commit = asyncio.get_event_loop().time()

            async def twilio_to_eleven():
                nonlocal samples_accum, last_commit
                while True:
                    msg = await ws.receive_text()
                    data = json.loads(msg); ev = data.get("event")

                    if ev == "start":
                        await ws.send_text(json.dumps({"event": "mark", "mark": {"name": "started"}}))
                        print("Twilio stream started")
                        continue

                    if ev == "media":
                        b64 = data["media"]["payload"]
                        mulaw_8k = base64.b64decode(b64)
                        pcm16_8k = audioop.ulaw2lin(mulaw_8k, 2)                     # μ-law → PCM16
                        pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)  # 8k → 16k
                        samples_accum.extend(pcm16_16k)

                        now = asyncio.get_event_loop().time()
                        if len(samples_accum) > 3200 or (now - last_commit) > 0.12:
                            chunk_b64 = base64.b64encode(bytes(samples_accum)).decode()
                            await elws.send(json.dumps({"user_audio_chunk": chunk_b64}))
                            samples_accum.clear()
                            last_commit = now

                    elif ev == "stop":
                        print("Twilio stream stop received")
                        if samples_accum:
                            chunk_b64 = base64.b64encode(bytes(samples_accum)).decode()
                            await elws.send(json.dumps({"user_audio_chunk": chunk_b64}))
                            samples_accum.clear()
                        break

            async def eleven_to_twilio():
                async for raw in elws:
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    # ElevenLabs outbound audio
                    if payload.get("type") == "audio" and "audio_event" in payload:
                        b64 = payload["audio_event"].get("audio_base_64")
                        if not b64:
                            continue
                        pcm16_16k = base64.b64decode(b64)
                        pcm16_8k, _ = audioop.ratecv(pcm16_16k, 2, 1, 16000, 8000, None)  # 16k → 8k
                        mulaw_8k = audioop.lin2ulaw(pcm16_8k, 2)                            # PCM16 → μ-law
                        await ws.send_text(json.dumps({"event": "media",
                                                       "media": {"payload": base64.b64encode(mulaw_8k).decode()}}))

            async def keepalive():
                while True:
                    await asyncio.sleep(8)
                    await ws.send_text(json.dumps({"event": "mark", "mark": {"name": "keepalive"}}))

            await asyncio.gather(twilio_to_eleven(), eleven_to_twilio(), keepalive())

    except WebSocketDisconnect:
        print("Twilio WS disconnected")
        return
    except Exception as e:
        print("Stream bridge error:", e)
        return

# ---------- Minimal OpenAI-compatible endpoints (not used by phone) ----------
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
    reply = rule_based_reply(user_text)
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
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
