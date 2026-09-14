"""Client-demo interface for TechSolAi Enterprise Policy AI."""
from __future__ import annotations

import streamlit as st

from app.service import PolicyAssistantService
from src.generation.rag_engine import INSUFFICIENT

st.set_page_config(page_title="Enterprise Policy AI | TechSolAi", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --ink:#10253f; --blue:#1769e0; --blue-dark:#0c4dad; --muted:#5d6b7c; --line:#dce5ef; }
.stApp { background:linear-gradient(180deg,#f6f9fd 0,#fff 24rem); color:var(--ink); }
.block-container { max-width:1060px; padding-top:2.25rem; padding-bottom:4rem; }
[data-testid="stSidebar"] { background:#0d2745; }
[data-testid="stSidebar"] * { color:#edf5ff; }
.side-brand { font-size:1.3rem; font-weight:800; letter-spacing:.02em; margin-bottom:.25rem; }
.side-tagline { color:#9fc8ff !important; font-weight:650; margin-bottom:1.6rem; }
.side-feature { border-top:1px solid rgba(255,255,255,.12); padding:.8rem 0; font-size:.9rem; }
.eyebrow { color:var(--blue); font-size:.78rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.hero { background:#fff; border:1px solid var(--line); border-radius:20px; padding:2.2rem 2.4rem; box-shadow:0 16px 45px rgba(16,37,63,.08); }
.hero h1 { color:var(--ink); font-size:2.35rem; line-height:1.15; margin:.45rem 0 .5rem; }
.hero p { color:var(--muted); font-size:1.08rem; margin:0; }
.hero .tagline { color:var(--blue-dark); font-size:.92rem; font-weight:750; margin-top:1.15rem; }
.section-kicker { color:var(--muted); font-size:.86rem; margin-top:-.55rem; }
.answer-card { background:#fff; border:1px solid var(--line); border-left:4px solid var(--blue); border-radius:14px; padding:1.25rem 1.5rem; box-shadow:0 8px 24px rgba(16,37,63,.06); }
.answer-label { color:var(--blue-dark); font-size:.74rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.65rem; }
.source-meta { color:var(--muted); font-size:.84rem; }
.response-meta { color:var(--muted); font-size:.78rem; text-align:right; margin-top:.45rem; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.15rem .25rem; }
div[data-testid="stExpander"] { border-color:var(--line); background:#fff; }
.demo-note { background:#f1f7ff; border:1px solid #cfe2fb; border-radius:11px; padding:.8rem 1rem; color:#31506f; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)

EXAMPLES = (
    "How should a team member report harassment?",
    "What happens when an employee leaves GitLab?",
    "How should a team member record sick time?",
    "How can a US team member request their personnel file?",
    "What is the right to disconnect for GitLab France employees?",
    "Which provider supplies private health insurance for Australian employees?",
)

@st.cache_resource(show_spinner=False)
def get_service() -> PolicyAssistantService:
    return PolicyAssistantService.from_environment()

def choose_example(question: str) -> None:
    st.session_state.question_input = question

with st.sidebar:
    st.markdown('<div class="side-brand">TechSolAi</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tagline">We Deal in Solutions.</div>', unsafe_allow_html=True)
    st.markdown("### Enterprise Policy AI")
    st.markdown('<div class="side-feature">✓ Grounded policy answers</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-feature">✓ Source-backed responses</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-feature">✓ Jurisdiction-aware policy retrieval</div>', unsafe_allow_html=True)
    st.caption("Designed for HR, operations, and compliance teams.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">TechSolAi</div>
  <h1>Enterprise Policy AI</h1>
  <p>AI-Powered HR Policy &amp; Employee Knowledge Assistant</p>
  <div class="tagline">We Deal in Solutions.</div>
</div>
""", unsafe_allow_html=True)
st.write("")
st.markdown('<div class="demo-note"><b>Demonstration environment.</b> Responses are grounded in indexed public policy material. For employment decisions, confirm applicable requirements with your HR or legal team.</div>', unsafe_allow_html=True)

try:
    service = get_service()
except Exception:
    st.error("Enterprise Policy AI is temporarily unavailable. Please contact the demo administrator or try again shortly.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

st.write("")
st.subheader("Ask a policy question")
st.markdown('<div class="section-kicker">Get a concise answer grounded in the available policy sources.</div>', unsafe_allow_html=True)

with st.form("policy_question_form", clear_on_submit=False):
    question = st.text_area("Question", key="question_input", placeholder="For example: How should a team member record sick time?", height=100, label_visibility="collapsed")
    submitted = st.form_submit_button("Ask Policy AI", type="primary", use_container_width=True, disabled=not service.available)

st.markdown("#### Try an example")
example_columns = st.columns(2)
for index, example in enumerate(EXAMPLES):
    example_columns[index % 2].button(example, key=f"example_{index}", use_container_width=True, on_click=choose_example, args=(example,))

if not service.available:
    st.info("Answer generation is not currently available. Please contact the demo administrator.")

if submitted and service.available:
    if not question.strip():
        st.warning("Enter a policy question before selecting Ask Policy AI.")
    else:
        with st.spinner("Reviewing the relevant policy sources…"):
            try:
                response = service.ask(question)
                st.session_state.history.insert(0, response)
            except Exception:
                st.error("We couldn’t complete the policy review right now. Please try again in a moment.")

for response in st.session_state.history:
    st.divider()
    st.markdown(f"**Question:** {response.question}")
    if response.grounded:
        st.markdown('<div class="answer-card"><div class="answer-label">Policy answer</div>', unsafe_allow_html=True)
        st.markdown(response.answer)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("We couldn’t find sufficient policy evidence to answer this question reliably.")
        st.caption(INSUFFICIENT)
    st.markdown(f'<div class="response-meta">Response time: {response.latency_ms / 1000:.2f} seconds</div>', unsafe_allow_html=True)
    if response.sources:
        st.markdown("#### Sources and evidence")
        for source_index, source in enumerate(response.sources):
            st.markdown(f"**[{source.citation}] {source.title}**")
            location = source.country if source.country and source.country != "Not specified" else "General policy"
            st.markdown(f'<div class="source-meta">{source.category} · {location}</div>', unsafe_allow_html=True)
            if source.source_url:
                st.markdown(f"[View policy source ↗]({source.source_url})")
            with st.expander(f"View cited evidence from {source.title}", expanded=False):
                st.write(source.evidence or "Evidence text is unavailable for this source.")
            if source_index < len(response.sources) - 1:
                st.write("")
