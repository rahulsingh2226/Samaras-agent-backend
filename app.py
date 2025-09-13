import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import ConfigDict  # ← added
import requests

app = FastAPI(title="Agent Brain for Samaira’s", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Basic business data (loaded earlier from your CSV in our template) ---
BIZ_NAME = "Samaira’s Spa and Wellness"
HOURS = "Mon–Sat 10am–6pm; Sun Closed"
PRICING = "$80 to $1100"
POLICY = "24h cancel; $25 late; 50% no-show; deposits for groups"

# --- Ollama optional settings ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
USE_OLLAMA = os.getenv("USE_OLLAMA", "auto")  # 'yes' | 'no' | 'auto'

class AgentRequest(BaseModel):
    input: str
    conversation: dict | None = None
    # ← added: ignore any extra fields ElevenLabs sends
    model_config = ConfigDict(extra='ignore')

@app.get("/ping")
def ping():
    return {"ok": True, "service": "agent-brain", "business": BIZ_NAME}

# ← added: simple GET endpoint so ElevenLabs can probe /agent
@app.get("/agent")
def agent_get():
    return {"output": "Hello from Samaira’s backend."}

def rule_based_reply(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["hour", "open", "close", "when are you open"]):
        return f"We’re open {HOURS}."
    if any(k in t for k in ["price", "how much", "cost", "rate"]):
        return f"Our pricing ranges are {PRICING}."
    if any(k in t for k in ["book", "appointment", "schedule", "reserve"]):
        return "I can help request a booking. May I have your full name, phone, email, the service you want, and a preferred day/time window?"
    if any(k in t for k in ["cancel", "refund", "policy", "late"]):
        return f"Our policy is: {POLICY}. Would you like me to email the full policy?"
    if any(k in t for k in ["insurance", "medical", "diagnosis"]):
        return "I’m not able to advise on medical or insurance matters. I can connect you to a team member if you’d like."
    return "I can help with hours, pricing, services, and booking requests. What would you like to know?"

def ollama_reply(text: str) -> str:
    system = (
        "You are Sarah, a calm, warm, helpful assistant for a spa. "
        "Answer briefly (1-2 sentences). If asked outside spa scope, say you can transfer to a human."
    )
    prompt = f"System: {system}\nUser: {text}\nAssistant:"
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        out = data.get("response", "").strip()
        return out or rule_based_reply(text)
    except Exception:
        return rule_based_reply(text)

@app.post("/agent")
async def agent(req: AgentRequest):
    user_text = (req.input or "").strip()
    if not user_text:
        return {"output": "Hello! How can I help you today?"}

    # decide whether to try Ollama
    use_llm = False
    if USE_OLLAMA == "yes":
        use_llm = True
    elif USE_OLLAMA == "no":
        use_llm = False
    else:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            use_llm = resp.ok
        except Exception:
            use_llm = False

    reply = ollama_reply(user_text) if use_llm else rule_based_reply(user_text)
    return {"output": reply}
