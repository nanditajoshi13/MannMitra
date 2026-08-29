"""
app.py
======
MANNMITRA - Streamlit chat application and live analytics dashboard.

Orchestrates risk_analysis.RiskAnalyzer, rag_engine.RAGEngine, and
Google GenAI (gemini-1.5-flash) into an empathetic peer-listener chat
experience with a live triage sidebar.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional

import streamlit as st

from risk_analysis import RiskAnalyzer, EmotionResult, RiskAssessment
from rag_engine import RAGEngine, RAGContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mannmitra.app")

# --------------------------------------------------------------------------
# Environment / API key harmonization
# --------------------------------------------------------------------------
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

_GENAI_AVAILABLE = False
_genai_client = None
try:
    from google import genai as _google_genai

    if os.environ.get("GEMINI_API_KEY"):
        _genai_client = _google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        _GENAI_AVAILABLE = True
    else:
        logger.warning("No GEMINI_API_KEY / GOOGLE_API_KEY found in environment.")
except Exception as exc:  # noqa: BLE001
    logger.warning("google-genai unavailable (%s). LLM responses will use a static fallback.", exc)
    _GENAI_AVAILABLE = False

SYSTEM_PROMPT = (
    "You are MannMitra, a warm, empathetic peer listener for Indian college "
    "students. You are NOT a therapist and never diagnose. Respond in 2-3 "
    "concise, natural sentences, in a caring, non-clinical, conversational "
    "tone - like a supportive friend, not a textbook. Validate the "
    "student's feelings before offering any gentle suggestion. If helpful "
    "context about a coping technique or campus resource is provided to "
    "you, weave it naturally into your own words as a friendly suggestion "
    "- never quote titles, filenames, or headers verbatim, and never say "
    "things like '[Source: ...]'. Keep responses grounded, human, and "
    "free of clinical jargon. Talk in non-repeating semi-casual patterns"
)

TIER_BADGES = {"GREEN": "🟢 GREEN", "YELLOW": "🟡 YELLOW", "RED": "🔴 RED"}
TIER_COLORS = {"GREEN": "#2ecc71", "YELLOW": "#f1c40f", "RED": "#e74c3c"}


# --------------------------------------------------------------------------
# Cached singletons
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading risk analyzer...")
def get_risk_analyzer() -> RiskAnalyzer:
    return RiskAnalyzer()


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_rag_engine() -> RAGEngine:
    return RAGEngine()


# --------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------
def generate_reply(user_text: str, chat_history: List[dict], rag_context: RAGContext) -> str:
    if not _GENAI_AVAILABLE or _genai_client is None:
        return (
            "I hear you, and I'm really glad you shared that with me. "
            "Thanks for opening up - want to tell me a bit more about what's "
            "been going on?"
        )

    context_note = ""
    if rag_context.is_used and rag_context.retrieved_documents:
        joined = "\n\n".join(rag_context.retrieved_documents[:2])
        context_note = (
            "\n\nHelpful background you may draw on (do not quote directly, "
            f"paraphrase naturally):\n{joined}"
        )

    convo_lines = []
    for turn in chat_history[-6:]:
        role = "Student" if turn["role"] == "user" else "MannMitra"
        convo_lines.append(f"{role}: {turn['content']}")
    convo_text = "\n".join(convo_lines)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Recent conversation:\n{convo_text}\n\n"
        f"Student's latest message: {user_text}"
        f"{context_note}\n\n"
        "Respond as MannMitra in 2-3 sentences:"
    )

    try:
        response = _genai_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )
        text = (response.text or "").strip()
        return text if text else "I'm here with you - can you tell me a little more?"
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini generation failed: %s", exc)
        return (
            "I'm here and listening, even though I'm having a little trouble "
            "finding the right words right now. Can you tell me more about "
            "how you're feeling?"
        )


# --------------------------------------------------------------------------
# Streamlit page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="MannMitra", page_icon="🧠", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "tier", "emotion", "rag_used"}
if "history_scores" not in st.session_state:
    st.session_state.history_scores = []
if "last_assessment" not in st.session_state:
    st.session_state.last_assessment = None
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None
if "last_rag" not in st.session_state:
    st.session_state.last_rag = None

analyzer = get_risk_analyzer()
rag_engine = get_rag_engine()

# --------------------------------------------------------------------------
# Sidebar - live analytics
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 MannMitra")
    st.caption("Live wellbeing analytics")

    st.subheader("Triage Status")
    if st.session_state.last_assessment is not None:
        tier = st.session_state.last_assessment.tier
        score = st.session_state.last_assessment.score
    else:
        tier, score = "GREEN", 0.0

    st.markdown(f"### {TIER_BADGES[tier]}")
    st.progress(min(1.0, max(0.0, score)))
    st.caption(f"Composite Distress Score: {score:.2f}")

    st.subheader("Emotion Breakdown")
    if st.session_state.last_emotion is not None and st.session_state.last_emotion.all_emotions:
        top_emotions = sorted(
            st.session_state.last_emotion.all_emotions.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
        for label, prob in top_emotions:
            st.write(f"**{label}**")
            st.progress(min(1.0, max(0.0, float(prob))))
    else:
        st.caption("No emotion data yet - send a message to begin.")

    st.subheader("Knowledge Retrieval (RAG)")
    if st.session_state.last_rag is not None and st.session_state.last_rag.is_used:
        sources = ", ".join(sorted(set(st.session_state.last_rag.sources)))
        st.success(f"✓ Injected: {sources}")
    else:
        st.info("Standby - no relevant resource retrieved.")

    st.divider()
    st.subheader("📞 Verified Helplines (India)")
    st.markdown(
        "- **Tele-MANAS**: `14416` (24/7, free)\n"
        "- **KIRAN**: `1800-599-0019` (24/7, toll-free)"
    )
    st.caption(
        "If you or someone you know is in immediate danger, please contact "
        "local emergency services or go to the nearest hospital."
    )

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.header("Chat with MannMitra")
st.caption("A private space to talk things through. This is peer support, not a substitute for professional care.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "tier" in msg:
            st.caption(
                f"Tier: {msg['tier']} · Emotion: {msg.get('emotion', 'n/a')} · "
                f"Knowledge Injected: {'Yes' if msg.get('rag_used') else 'No'}"
            )

user_input = st.chat_input("Share what's on your mind...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    emotion, assessment = analyzer.assess_risk(user_input, st.session_state.history_scores)
    st.session_state.history_scores.append(assessment.score)
    st.session_state.last_assessment = assessment
    st.session_state.last_emotion = emotion

    if assessment.is_crisis:
        rag_context = RAGContext(is_used=False)
        reply = RiskAnalyzer.get_crisis_response()
    else:
        rag_context = rag_engine.retrieve(user_input, assessment.score)
        history_for_llm = st.session_state.messages
        reply = generate_reply(user_input, history_for_llm, rag_context)

    st.session_state.last_rag = rag_context

    with st.chat_message("assistant"):
        st.markdown(reply)
        st.caption(
            f"Tier: {assessment.tier} · Emotion: {emotion.label} · "
            f"Knowledge Injected: {'Yes' if rag_context.is_used else 'No'}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "tier": assessment.tier,
            "emotion": emotion.label,
            "rag_used": rag_context.is_used,
        }
    )

    st.rerun()