"""
agent.py — Webhook server, LangGraph agentic tool-use loop, RAG pipeline, conversation memory
Gym WhatsApp Agent (LangGraph + Groq edition)
"""

import os
import json
import logging
import re
import numpy as np
from pathlib import Path

from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv

# ── LangChain / LangGraph imports ─────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict

from sentence_transformers import SentenceTransformer

import db
import whatsapp as wa

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

db.init_db()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
MODEL          = os.getenv("MODEL", "llama-3.3-70b-versatile")
VERIFY_TOKEN   = os.getenv("WHATSAPP_VERIFY_TOKEN", "my_verify_token")
MAX_HISTORY    = 20   # max messages kept per user
TOP_K          = 5

KNOWLEDGE_PATH = Path("knowledge.txt")
MEDIA_DIR      = Path("media")

HEAD_TRAINER_IMAGES = [
    "priya_sharma.jpg",
    "rahul_verma.jpg",
    "neha_kapoor.jpg",
    "arjun_mehta.jpg",
    "sonal_jain.jpg",
    "vikram_singh.jpg",
]

PLAN_CATALOG = {
    "basic": {
        "name": "Basic Plan",
        "price": "Rs. 1,499/month",
        "summary": "Gym floor access, locker room, and 2 complimentary group classes per week.",
    },
    "premium": {
        "name": "Premium Plan",
        "price": "Rs. 2,499/month",
        "summary": "Basic benefits plus unlimited classes, 1 PT session/month, sauna, steam, and priority booking.",
    },
    "elite": {
        "name": "Annual Elite Plan",
        "price": "Rs. 19,999/year",
        "summary": "Premium benefits plus 3 PT sessions/month, 2 guest passes/month, nutrition consult, and gym bag.",
    },
}

# ── RAG Pipeline Setup ────────────────────────────────────────────────────────
log.info("Loading embedding model…")
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def _load_corpus() -> tuple[list[str], np.ndarray]:
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    log.info(f"[RAG] Encoding {len(chunks)} chunks…")
    vectors = _embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return chunks, np.array(vectors, dtype=np.float32)


CORPUS_CHUNKS, CORPUS_VECTORS = _load_corpus()
log.info(f"[RAG] Corpus ready — {len(CORPUS_CHUNKS)} chunks, {CORPUS_VECTORS.shape[1]}-dim embeddings.")


# ── LangChain Tools ───────────────────────────────────────────────────────────
# We use a context var to pass sender_phone into tools that need it
_current_sender: str = ""


@tool
def search_knowledge_base(query: str) -> str:
    """Search the gym's knowledge base for membership plans, pricing, operating hours,
    branch locations, amenities, policies, offers, trainers, and class schedules.
    Always call this before answering any factual question about the gym."""
    q_vec = _embed_model.encode([query], normalize_embeddings=True)[0]
    scores = CORPUS_VECTORS @ q_vec
    top_indices = np.argsort(-scores)[:TOP_K]
    results = "\n\n".join(CORPUS_CHUNKS[i] for i in top_indices)
    log.info(f"[RAG] Query='{query}' → top scores: {scores[top_indices]}")
    return results


@tool
def send_trainer_profiles() -> str:
    """Send photos and profiles of all gym trainers to the member via WhatsApp."""
    sent = []
    for fname in HEAD_TRAINER_IMAGES:
        path = MEDIA_DIR / fname
        if path.exists():
            try:
                trainer_name = fname.replace("_", " ").rsplit(".", 1)[0].title()
                wa.send_image(_current_sender, str(path), caption=f"Trainer: {trainer_name}")
                sent.append(fname)
            except Exception as e:
                log.warning(f"[TOOL] Could not send {fname}: {e}")
    if sent:
        return f"Trainer profile images sent: {', '.join(sent)}"
    return "No trainer images found in the media directory."


@tool
def book_trial_class(date: str, time: str, name: str = "") -> str:
    """Book a free trial class for a prospective member.
    Args:
        date: Trial date in YYYY-MM-DD format.
        time: Trial time in HH:MM (24-hour) format.
        name: Name of the person booking the trial (optional).
    """
    result = db.book_trial(phone=_current_sender, date=date, time=time, name=name)
    if result == "success":
        return f"Trial class successfully booked for {date} at {time}."
    return f"Could not book trial: {result}"


@tool
def get_membership_details() -> str:
    """Retrieve the current membership details for the member who sent this message."""
    member = db.get_member(_current_sender)
    if member:
        return (
            f"Member: {member['name']}\n"
            f"Plan: {member['plan_type']}\n"
            f"Start Date: {member['start_date']}\n"
            f"Expiry Date: {member['expiry_date']}"
        )
    return "No membership found for this phone number. This person may be a prospect, not a registered member."


@tool
def get_next_class(class_type: str) -> str:
    """Find the next upcoming scheduled session for a specific class type.
    Args:
        class_type: Class type name, e.g. 'Yoga', 'HIIT', 'Zumba', 'Strength', 'Pilates', 'Spin'.
    """
    cls = db.get_next_class(class_type)
    if cls:
        return (
            f"Class ID: {cls['id']}\n"
            f"Type: {cls['class_type']}\n"
            f"Date: {cls['date']}\n"
            f"Time: {cls['time']}\n"
            f"Instructor: {cls['instructor']}\n"
            f"Available Slots: {cls['slots'] - cls['booked']}"
        )
    return f"No upcoming {class_type} classes found."


@tool
def register_for_class(class_id: int) -> str:
    """Register the current member for a specific scheduled class using its class ID.
    Args:
        class_id: The numeric ID of the class to register for (obtained from get_next_class).
    """
    result = db.register_for_class(_current_sender, class_id)
    if result == "success":
        return f"Successfully registered for class ID {class_id}."
    return f"Registration failed: {result}"


TOOLS = [
    search_knowledge_base,
    send_trainer_profiles,
    book_trial_class,
    get_membership_details,
    get_next_class,
    register_for_class,
]

# ── LangGraph Agent Setup ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, friendly, and professional assistant for FitLife Gym in Alwar, Rajasthan.

Your role is to assist both prospective members (people interested in joining) and existing members with:
- Membership plans, pricing, and joining process
- Class schedules, bookings, and registrations
- Trainer information and profiles
- Operating hours and branch locations
- Policies, amenities, and current offers
- Trial class bookings

IMPORTANT RULES:
1. Always use the search_knowledge_base tool FIRST before answering any factual question about the gym (pricing, timings, policies, trainers, etc.)
2. Never invent gym-specific details — only use information retrieved from the tools
3. Be warm, encouraging, and conversational in tone
4. Keep responses concise and easy to read on a mobile screen
5. If a member asks about "my membership" or "my plan", use get_membership_details
6. For class queries, use get_next_class to fetch live data, not just your context
7. Respond in the same language the user writes in (Hindi or English)

CONVERSATION QUALITY:
- If retrieved context contains plan names, prices, or benefits, answer confidently. Do not say "I only have details..." or ask if you should try to find information that is already in the context.
- When someone asks for "all plans", list Basic, Premium, and Annual Elite together with price and 1-2 key benefits each.
- If the user says "1499 plan", "add 1499", or similar, treat it as interest in the Basic Plan (₹1,499/month). Confirm the plan clearly and ask for only the next missing signup detail.
- For joining/signup intent, use this flow: confirm selected plan, ask for full name, then tell them they can complete payment at the branch by cash, UPI, or card with a government photo ID and one passport-size photo.
- Ask one question at a time. Do not ask for payment method, ID, and photo all in the same message.
- Avoid sounding robotic or uncertain. Use short WhatsApp-friendly replies with simple bullets when listing plans.
- Format important lists with WhatsApp-friendly bullets and bold labels using *label*.
- End with one helpful next step, not multiple choices.
- Never mention tools, databases, retrieved context, model errors, or internal implementation details.
- If the user message is ambiguous, make the most likely gym-related interpretation and ask one short clarifying question.
"""

llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL)
llm_with_tools = llm.bind_tools(TOOLS)


def _last_human_text(messages: list[BaseMessage]) -> str:
    """Return the latest user message text from a message list."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _messages_without_tool_artifacts(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop tool-call bookkeeping before a no-tools fallback LLM call."""
    cleaned: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            if msg.content:
                cleaned.append(AIMessage(content=msg.content))
            continue
        cleaned.append(msg)
    return cleaned


def _answer_with_retrieved_context(messages: list[BaseMessage], error: Exception) -> AIMessage:
    """Fallback for Groq tool-use parser failures.

    Groq can occasionally reject a generated tool call before LangGraph gets a
    chance to execute ToolNode. For knowledge-base questions, retrieve context
    locally and complete without provider-side function calling.
    """
    query = _last_human_text(messages)
    context = search_knowledge_base.invoke({"query": query or "gym information"})
    fallback_system = SystemMessage(
        content=(
            f"{SYSTEM_PROMPT}\n\n"
            "The knowledge base has already been searched locally. Answer using "
            "only this retrieved context. Do not mention tools, retrieval, provider "
            "errors, or internal implementation details. If the context does not "
            "contain the answer, say so and ask a concise follow-up question.\n\n"
            f"Retrieved context:\n{context}"
        )
    )
    response = llm.invoke([fallback_system] + _messages_without_tool_artifacts(messages))
    log.warning(f"[LLM] Used no-tool RAG fallback after Groq error: {error}")
    return response


# ── LangGraph State ───────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: AgentState) -> AgentState:
    """LLM node: prepend system prompt and call the model."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        if "tool_use_failed" not in str(e):
            raise
        response = _answer_with_retrieved_context(state["messages"], e)
    log.info(f"[LLM] finish_reason={getattr(response, 'response_metadata', {}).get('finish_reason', '?')}")
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Router: if last message has tool calls, go to tools node, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Build the graph
tool_node = ToolNode(TOOLS)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("llm")
graph_builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "llm")

agent_graph = graph_builder.compile()
log.info("[GRAPH] LangGraph agent compiled successfully.")


# ── Conversation Memory (in-process, keyed by phone number) ──────────────────
# Stores list[BaseMessage] per user
_conversation_history: dict[str, list[BaseMessage]] = {}
_lead_state: dict[str, dict[str, str]] = {}


def _clean_text(text: str) -> str:
    """Normalize user text for lightweight intent matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_greeting(text: str) -> bool:
    cleaned = _clean_text(text)
    greetings = {"hi", "hii", "hiii", "hello", "hey", "namaste", "namaskar", "hola"}
    return cleaned in greetings or cleaned.startswith(("hi ", "hii ", "hello ", "hey "))


def _mentions_plan(text: str) -> bool:
    cleaned = _clean_text(text)
    plan_words = ("plan", "plans", "membership", "pricing", "price", "cost", "fees", "fee", "1499", "2499", "19999")
    return any(word in cleaned for word in plan_words)


def _wants_all_plans(text: str) -> bool:
    cleaned = _clean_text(text)
    return _mentions_plan(cleaned) and any(word in cleaned for word in ("all", "explore", "show", "list", "options", "membership"))


def _wants_monthly_plans(text: str) -> bool:
    cleaned = _clean_text(text)
    return _mentions_plan(cleaned) and any(word in cleaned for word in ("monthly", "month", "per month"))


def _selected_plan_key(text: str) -> str | None:
    cleaned = _clean_text(text)
    if "1499" in cleaned or "1,499" in cleaned or "basic" in cleaned:
        return "basic"
    if "2499" in cleaned or "2,499" in cleaned or "premium" in cleaned:
        return "premium"
    if "19999" in cleaned or "19,999" in cleaned or "elite" in cleaned or "annual" in cleaned or "yearly" in cleaned:
        return "elite"
    return None


def _has_join_intent(text: str) -> bool:
    cleaned = _clean_text(text)
    return any(word in cleaned for word in ("join", "add", "buy", "start", "activate", "take", "enroll", "enrol", "signup", "sign up"))


def _is_affirmation(text: str) -> bool:
    cleaned = _clean_text(text)
    return cleaned in {"yes", "yeah", "yep", "ok", "okay", "sure", "haan", "ha", "ji", "yes please"}


def _is_cancel(text: str) -> bool:
    cleaned = _clean_text(text)
    return cleaned in {"cancel", "stop", "leave it", "no", "not now", "later", "restart", "reset"}


def _is_thanks(text: str) -> bool:
    cleaned = _clean_text(text)
    return cleaned in {"thanks", "thank you", "thx", "ok thanks", "okay thanks", "great thanks"}


def _looks_like_name(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 2 or len(cleaned) > 60:
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    return bool(re.fullmatch(r"[A-Za-z .'-]+", cleaned))


def _format_all_plans(include_elite: bool = True) -> str:
    lines = ["Here are FitLife Gym's membership options:"]
    keys = ["basic", "premium", "elite"] if include_elite else ["basic", "premium"]
    for key in keys:
        plan = PLAN_CATALOG[key]
        lines.append(f"\n*{plan['name']}* - {plan['price']}\n{plan['summary']}")
    lines.append("\nAll plans include a free first-time trainer assessment. Students and senior citizens get 15% off with valid ID.")
    lines.append("\nWhich plan would you like to start with?")
    return "\n".join(lines)


def _format_welcome() -> str:
    return (
        "Namaste! Welcome to FitLife Gym, Alwar.\n\n"
        "I can help you with:\n"
        "- Membership plans\n"
        "- Trial class booking\n"
        "- Class schedule\n"
        "- Trainers, timings, locations, and offers\n\n"
        "What would you like to explore first?"
    )


def _start_signup(sender_phone: str, plan_key: str) -> str:
    plan = PLAN_CATALOG[plan_key]
    _lead_state[sender_phone] = {"flow": "signup", "plan": plan_key, "step": "name"}
    return (
        f"Great choice. I can start your signup for *{plan['name']}* ({plan['price']}).\n\n"
        "Please share your full name."
    )


def _continue_signup(sender_phone: str, user_message: str) -> str | None:
    state = _lead_state.get(sender_phone)
    if not state or state.get("flow") != "signup":
        return None

    plan = PLAN_CATALOG[state["plan"]]
    step = state.get("step")

    if step == "name":
        if not _looks_like_name(user_message):
            return "Please send your full name, for example: Rajat Paliwal."
        state["name"] = user_message.strip()
        state["step"] = "branch"
        return (
            f"Thanks, {state['name']}. I have noted *{plan['name']}* for you.\n\n"
            "Which branch is more convenient for you?\n"
            "- Main Alwar Branch, near Circuit House\n"
            "- Bansur Road Branch"
        )

    if step == "branch":
        cleaned = _clean_text(user_message)
        if "bansur" in cleaned:
            state["branch"] = "Bansur Road Branch"
        elif "main" in cleaned or "circuit" in cleaned or "alwar" in cleaned:
            state["branch"] = "Main Alwar Branch"
        else:
            return "Please choose one branch: Main Alwar Branch or Bansur Road Branch."
        state["step"] = "done"
        return (
            f"Perfect. Signup request noted:\n"
            f"- Name: {state['name']}\n"
            f"- Plan: {plan['name']} ({plan['price']})\n"
            f"- Branch: {state['branch']}\n\n"
            "To complete joining, please visit the branch with a government photo ID and one passport-size photo. "
            "Payment can be done by cash, UPI, or card.\n\n"
            "Would you also like to book a free first-time trainer assessment?"
        )

    if step == "assessment_time":
        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", user_message)
        time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", user_message)
        if not date_match or not time_match:
            return "Please send the date and time like this: 2026-06-08 at 10:00."

        date = date_match.group(1)
        time = time_match.group(0)
        result = db.book_trial(phone=sender_phone, date=date, time=time, name=state.get("name", ""))
        if result == "success":
            state["step"] = "complete"
            return (
                f"Done. Your free trainer assessment request is booked for {date} at {time}.\n\n"
                "Our team will confirm the slot on WhatsApp. See you at FitLife."
            )
        return "I could not save that booking right now. Please try another date and time."

    return None


def _quick_reply(sender_phone: str, user_message: str) -> str | None:
    """Handle common high-confidence WhatsApp flows without model drift."""
    cleaned = _clean_text(user_message)
    state = _lead_state.get(sender_phone)

    if _is_cancel(cleaned):
        _lead_state.pop(sender_phone, None)
        return "No problem. I have paused this flow. Tell me whenever you want to continue."

    if _is_thanks(cleaned):
        return "You're welcome. Happy to help."

    continued = _continue_signup(sender_phone, user_message)
    if continued:
        return continued

    if _is_greeting(cleaned):
        return _format_welcome()

    if _wants_monthly_plans(cleaned):
        return _format_all_plans(include_elite=False)

    if _wants_all_plans(cleaned):
        return _format_all_plans(include_elite=True)

    selected_plan = _selected_plan_key(cleaned)
    if selected_plan and (_has_join_intent(cleaned) or "plan" in cleaned):
        return _start_signup(sender_phone, selected_plan)

    if state and state.get("flow") == "signup" and state.get("step") == "done" and _is_affirmation(cleaned):
        state["step"] = "assessment_time"
        return (
            "Lovely. Your free trainer assessment is included with the plan.\n\n"
            "Please tell me your preferred date and time, for example: 2026-06-08 at 10:00."
        )

    return None


def run_agent(sender_phone: str, user_message: str) -> str:
    """Run the LangGraph agent for a given sender and message."""
    global _current_sender
    _current_sender = sender_phone

    history = _conversation_history.setdefault(sender_phone, [])
    history.append(HumanMessage(content=user_message))

    quick_reply = _quick_reply(sender_phone, user_message)
    if quick_reply:
        history.append(AIMessage(content=quick_reply))
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]
        return quick_reply

    # Trim to last MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    try:
        result = agent_graph.invoke({"messages": history})
    except Exception as e:
        log.error(f"[AGENT] Graph invocation failed: {e}")
        # Roll back the user message so history stays clean
        if history and isinstance(history[-1], HumanMessage):
            history.pop()
        return "I'm having trouble processing that right now. Please try again in a moment."

    # Update history with all new messages produced by the graph
    new_messages = result["messages"]
    _conversation_history[sender_phone] = new_messages[-MAX_HISTORY:]

    # Return the last AI text response
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content.strip()

    return "I'm sorry, I couldn't generate a response. Please try again."


# ── FastAPI Webhook Server ────────────────────────────────────────────────────
app = FastAPI(title="Gym WhatsApp RAG Agent (LangGraph)")


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification handshake."""
    params   = request.query_params
    mode     = params.get("hub.mode")
    token    = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        log.info("[WEBHOOK] Verification successful.")
        return Response(content=challenge, media_type="text/plain")

    log.warning("[WEBHOOK] Verification failed.")
    return Response(status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """Handle incoming WhatsApp message payloads."""
    try:
        payload = await request.json()
        log.debug(f"[WEBHOOK] Payload: {json.dumps(payload)[:500]}")
    except Exception:
        return Response(status_code=400)

    try:
        value   = payload["entry"][0]["changes"][0]["value"]
        if "messages" not in value:
            return Response(status_code=200)  # Status callback — ignore

        msg_obj   = value["messages"][0]
        sender    = msg_obj["from"]  # E.164 without +

        if msg_obj.get("type") == "text":
            user_text = msg_obj["text"]["body"].strip()
        else:
            wa.send_text(sender, "I can only process text messages right now. Please type your question.")
            return Response(status_code=200)

        log.info(f"[MSG] From={sender}: {user_text[:100]}")

    except (KeyError, IndexError) as e:
        log.warning(f"[WEBHOOK] Could not parse payload: {e}")
        return Response(status_code=200)

    try:
        reply = run_agent(sender, user_text)
        if reply:
            wa.send_text(sender, reply)
    except Exception as e:
        log.exception(f"[AGENT] Error processing message from {sender}: {e}")
        try:
            wa.send_text(sender, "Sorry, something went wrong on my end. Please try again in a moment.")
        except Exception:
            pass

    return Response(status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "corpus_chunks": len(CORPUS_CHUNKS)}


@app.post("/reset/{phone}")
async def reset_history(phone: str):
    """Clear conversation history for a specific phone number."""
    _conversation_history.pop(phone, None)
    _lead_state.pop(phone, None)
    log.info(f"[RESET] History cleared for {phone}")
    return {"status": "cleared", "phone": phone}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
