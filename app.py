import os, json
from time import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Agent Brain for Samaira’s", version="2.0-turnbased")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --------- Business data ----------
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Monday to Saturday 10am to 6pm; Sunday Closed"
PRICING = "$80 to $1100"
POLICY = "24 hours free to cancel; $25 late; 50% no-show; deposits for groups"

# --------- Rule-based replies ----------
def rule_based_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["hour", "hours","open", "close", "when are you open"]):
        return f"We’re open {HOURS}."
    if any(k in t for k in ["pricing", "how much", "cost", "rate"]):
        return f"Our pricing ranges are {PRICING}."
    if any(k in t for k in ["service", "services", "treatment", "what do you offer", "menu"]):
        return ("We offer massages, facials, body scrubs, wellness packages, "
                "and relaxation therapies. Want details on something specific?")
    if any(k in t for k in ["location", "where are you", "address", "directions"]):
        return "We are in New Hyde Park, NY."
    if any(k in t for k in ["book", "appointment", "schedule", "reserve"]):
        return "I can request a booking. May I have your name, phone, email, service, and preferred time?"
    if any(k in t for k in ["cancel", "refund", "policy", "late"]):
        return f"Our policy: {POLICY}."
    return "I can help with hours, services, pricing, booking, and policies. What would you like to know?"

# --------- API model ----------
class AgentRequest(BaseModel):
    input: str
    conversation: dict | None = None
    model_config = ConfigDict(extra='ignore')

@app.get("/ping")
def ping():
    return {"ok": True, "service": "agent-brain", "business": BIZ_NAME}

@app.post("/agent")
async def agent(req: AgentRequest):
    txt = (req.input or "").strip()
    return {"output": rule_based_reply(txt) if txt else "Hello! How can I help you today?"}

# ---------- Twilio turn-based voice (free trial safe) ----------

# First interaction: greet and capture speech
@app.post("/twilio-voice")
def twilio_voice():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" action="/twilio-next" method="POST" timeout="3" speechTimeout="auto" language="en-US">
    <Say voice="Polly.Joanna">Welcome to Samaira’s Spa and Wellness. How can I help you today?</Say>
  </Gather>
  <Say voice="Polly.Joanna">I didn’t catch that. Please call again.</Say>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")

# Handle caller speech, reply, then loop
@app.post("/twilio-next")
async def twilio_next(request: Request):
    form = await request.form()
    user_text = (form.get("SpeechResult") or "").strip()

    reply = rule_based_reply(user_text) if user_text else "Could you repeat that?"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">{reply}</Say>
  <Pause length="1"/>
  <Redirect method="POST">/twilio-voice</Redirect>
</Response>"""
    return PlainTextResponse(twiml, media_type="application/xml")

# ---------- Minimal OpenAI-compatible endpoints ----------
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
