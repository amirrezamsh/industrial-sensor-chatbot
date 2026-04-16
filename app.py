import streamlit as st
import pandas as pd
import datetime
import time
import os
from src.agent.core import check_ollama_connection
from src.agent.core import generate_ollma_response
from src.utils.file_utils import validate_dataset_structure, scan_dataset_metadata
from src.agent.prompts import build_router_prompt, build_responder_prompt
from src.config import FEATURES_DIR
import streamlit as st
import datetime
import time
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SenseTimeAI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  THEME CSS  –  Navy/Steel palette
#  #0A2647  #144272  #205295  #2C74B3
# ─────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
    --navy-900:   #0A2647;
    --navy-700:   #144272;
    --blue-500:   #205295;
    --blue-400:   #2C74B3;
    --blue-300:   #4a9fd4;
    --blue-200:   #7ec8e3;
    --blue-glow:  rgba(44,116,179,0.25);
    --blue-glow2: rgba(44,116,179,0.08);

    --surface-0:  #060f1e;
    --surface-1:  #080f1d;
    --surface-2:  #0d1c31;
    --surface-3:  #102040;

    --text-hi:    #e8f4ff;
    --text-mid:   #8bafc8;
    --text-lo:    #3d5a72;

    --border-dim: rgba(44,116,179,0.18);
    --border-mid: rgba(44,116,179,0.35);
    --border-hi:  rgba(44,116,179,0.65);

    --r-sm: 8px;
    --r-md: 14px;
    --r-lg: 22px;

    --glow-sm:  0 0 12px rgba(44,116,179,0.3);
    --glow-md:  0 0 24px rgba(44,116,179,0.4), 0 0 48px rgba(44,116,179,0.15);
    --panel-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px var(--border-dim);
}

/* ── Global Reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
section.main {
    background-color: var(--surface-0) !important;
    color: var(--text-hi) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Subtle grid texture on main bg */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(44,116,179,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(44,116,179,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Radial glows in corners */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 10% 5%, rgba(20,66,114,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 90% 90%, rgba(32,82,149,0.10) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Kill white topbar ── */
header[data-testid="stHeader"],
header[data-testid="stHeader"] *,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: var(--surface-0) !important;
    background-color: var(--surface-0) !important;
    border-bottom: 1px solid var(--border-dim) !important;
}
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] button {
    color: var(--text-mid) !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-mid) !important;
    border-radius: var(--r-sm) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1e 0%, #060b18 100%) !important;
    border-right: 1px solid var(--border-dim) !important;
    box-shadow: 6px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

/* Top accent line in sidebar */
[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue-400), var(--blue-300), transparent);
    margin-bottom: 8px;
}

/* Sidebar logo/branding area */
.sidebar-brand {
    padding: 20px 16px 24px;
    border-bottom: 1px solid var(--border-dim);
    margin-bottom: 20px;
}
.sidebar-brand .brand-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, var(--navy-700), var(--blue-500));
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 10px;
    box-shadow: var(--glow-sm);
    border: 1px solid var(--border-mid);
}
.sidebar-brand .brand-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.05rem;
    color: var(--text-hi);
    letter-spacing: 0.04em;
}
.sidebar-brand .brand-sub {
    font-size: 0.7rem;
    color: var(--text-mid);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Sidebar text input */
[data-testid="stSidebar"] .stTextInput input {
    background: var(--surface-3) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-hi) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    caret-color: var(--blue-300) !important;
    padding: 10px 12px !important;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--blue-400) !important;
    box-shadow: var(--glow-sm) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stTextInput label {
    color: var(--text-mid) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--navy-700), var(--blue-500)) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-hi) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.55rem 1.2rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, var(--blue-500), var(--blue-400)) !important;
    box-shadow: var(--glow-md) !important;
    transform: translateY(-1px) !important;
    border-color: var(--blue-300) !important;
}

/* Sidebar alerts */
[data-testid="stSidebar"] .stAlert {
    background: rgba(10,38,71,0.6) !important;
    border-radius: var(--r-sm) !important;
    border-left: 3px solid var(--blue-400) !important;
}

/* ── TITLE / Heading ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: 0.06em !important;
    background: linear-gradient(110deg, #7ec8e3 0%, var(--blue-400) 40%, #4a9fd4 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: none !important;
    margin: 0 !important;
}

h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--blue-300) !important;
    letter-spacing: 0.04em !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--r-md) !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: var(--panel-shadow) !important;
    animation: slideUp 0.3s ease-out both;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(135deg, rgba(20,66,114,0.7), rgba(32,82,149,0.5)) !important;
    border-color: var(--border-mid) !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: linear-gradient(135deg, rgba(6,15,30,0.95), rgba(8,15,29,0.9)) !important;
    border-color: var(--border-dim) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]):hover {
    border-color: var(--border-hi) !important;
    box-shadow: var(--panel-shadow), var(--glow-sm) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] a,
[data-testid="stChatMessage"] label,
[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown * {
    color: #e8f4ff !important;
    line-height: 1.8 !important;
    font-size: 0.93rem !important;
}

/* Extra specificity for user bubble text */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown * {
    color: #ffffff !important;
}

/* Extra specificity for assistant bubble text */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown * {
    color: #e8f4ff !important;
}

[data-testid="stChatMessage"] code {
    background: rgba(10,38,71,0.8) !important;
    color: var(--blue-300) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
}
[data-testid="stChatMessage"] pre {
    background: rgba(6,9,18,0.9) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-sm) !important;
}

/* ── Bottom Chat Input ── */
[data-testid="stBottom"],
[data-testid="stBottom"] * {
    background: var(--surface-0) !important;
    background-color: var(--surface-0) !important;
}
[data-testid="stBottom"] {
    border-top: 1px solid var(--border-dim) !important;
    box-shadow: 0 -20px 60px rgba(6,15,30,0.95) !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div {
    background: #102040 !important;
    background-color: #102040 !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 18px !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.4), var(--glow-sm) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] > div > div:focus-within {
    border-color: var(--blue-400) !important;
    box-shadow: var(--glow-md), inset 0 2px 8px rgba(0,0,0,0.3) !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: transparent !important;
    background-color: transparent !important;
    color: #e8f4ff !important;
    caret-color: var(--blue-300) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.92rem !important;
    border: none !important;
    outline: none !important;
}

/* Placeholder text color */
[data-testid="stChatInput"] textarea::placeholder {
    color: #3d5a72 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--blue-500), var(--blue-400)) !important;
    border-radius: 12px !important;
    box-shadow: var(--glow-sm) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: var(--glow-md) !important;
}

/* ── Pills / Suggestion chips ── */
[data-testid="stPills"],
[data-testid="stPills"] > div {
    gap: 10px !important;
    background: transparent !important;
}

/* Force dark background on all pill button states — override Streamlit's white default */
[data-testid="stPills"] button,
[data-testid="stPills"] button:not([aria-pressed="true"]),
[data-testid="stPills"] > div button,
div[data-testid="stPills"] button {
    background: #0d1c31 !important;
    background-color: #0d1c31 !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 999px !important;
    color: #8bafc8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stPills"] button *,
[data-testid="stPills"] button div,
[data-testid="stPills"] button span,
[data-testid="stPills"] button p {
    background: transparent !important;
    background-color: transparent !important;
    color: inherit !important;
}
[data-testid="stPills"] button:hover {
    background: rgba(32,82,149,0.5) !important;
    background-color: rgba(32,82,149,0.5) !important;
    border-color: var(--blue-400) !important;
    color: #e8f4ff !important;
    box-shadow: var(--glow-sm) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stPills"] button[aria-pressed="true"] {
    background: var(--blue-400) !important;
    background-color: var(--blue-400) !important;
    border-color: var(--blue-300) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: var(--glow-md) !important;
}

/* ── Restart / top buttons ── */
[data-testid="stMain"] .stButton > button,
[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-mid) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.35rem 1rem !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
[data-testid="stMain"] .stButton > button:hover,
[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: var(--blue-glow) !important;
    border-color: var(--blue-400) !important;
    color: var(--blue-300) !important;
    box-shadow: var(--glow-sm) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--blue-400) !important; }

/* ── Status / italic ── */
em {
    color: var(--text-mid) !important;
    font-style: normal !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1;   }
}

/* ── Alerts ── */
.stAlert {
    background: rgba(10,38,71,0.5) !important;
    border-radius: var(--r-sm) !important;
    border-left: 3px solid var(--blue-400) !important;
}
.stAlert p { color: var(--text-hi) !important; }

/* ── Title row ── */
[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 14px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid var(--border-dim) !important;
    margin-bottom: 16px !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: var(--r-md) !important;
    border: 1px solid var(--border-mid) !important;
    box-shadow: var(--panel-shadow) !important;
}

/* ── Dialog ── */
[data-testid="stModal"], [role="dialog"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--r-lg) !important;
    box-shadow: 0 24px 80px rgba(0,0,0,0.7), var(--glow-md) !important;
}
[role="dialog"] p { color: var(--text-hi) !important; }
[role="dialog"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue-500), var(--blue-400)) !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar            { width: 5px; height: 5px; }
::-webkit-scrollbar-track      { background: var(--surface-0); }
::-webkit-scrollbar-thumb      { background: var(--navy-700); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover{ background: var(--blue-400); }

::selection {
    background: var(--blue-400) !important;
    color: white !important;
}

/* ── Landing screen – centered layout ── */
.landing-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    padding: 40px 20px;
    text-align: center;
}

.landing-headline {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: 0.08em;
    background: linear-gradient(110deg, #7ec8e3 0%, #2C74B3 45%, #4a9fd4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.landing-sub {
    font-size: 0.9rem;
    color: var(--text-mid);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 40px;
}

.landing-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(44,116,179,0.12);
    border: 1px solid var(--border-mid);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.75rem;
    color: var(--blue-300);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 48px;
}

.landing-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--blue-400), transparent);
    margin: 0 auto 36px;
}

.suggestion-label {
    font-size: 0.72rem;
    color: var(--text-lo);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Status badge in sidebar */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-online  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.status-offline { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

.sidebar-section-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-lo);
    margin-bottom: 8px;
    padding-left: 2px;
}

.sidebar-info-block {
    background: rgba(10,38,71,0.35);
    border: 1px solid var(--border-dim);
    border-radius: var(--r-sm);
    padding: 12px 14px;
    font-size: 0.78rem;
    color: var(--text-mid);
    line-height: 1.6;
    margin-top: 16px;
}

.sidebar-info-block strong {
    color: var(--blue-300);
    font-weight: 600;
}

/* Dataset loaded indicator */
.dataset-loaded {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: var(--r-sm);
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #86efac;
    margin-top: 12px;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

SUGGESTIONS = {
    "📊  Feature importance":    "Which features in my entire dataset better separate the faulty samples from the normal ones?",
    "🔍  Dataset metadata":      "What are the different sensors in my dataset? What are the different sensor types? Also provide me with different fault details and conditions.",
    "💡  My capabilities":       "Can you tell me what things you are able to do?",
}

FEATURES_DIR = "/tmp/sensetime_features"
os.makedirs(FEATURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  DIALOG
# ─────────────────────────────────────────────
@st.dialog("Existing analysis found")
def existing_folder_dialog():
    st.write(
        "An analysis for a folder with the same name already exists. "
        "Would you like to recompute the features?"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Keep existing", type="primary", use_container_width=True):
            st.session_state.FEATURES_PATH = os.path.basename(st.session_state.DATASET_PATH)
            st.rerun()
    with col2:
        if st.button("Recompute", use_container_width=True):
            st.session_state.FEATURES_PATH = (
                f"{os.path.basename(st.session_state.DATASET_PATH)}_{int(time.time())}"
            )
            st.rerun()


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "messages":               [],
    "DATASET_PATH":           None,
    "FEATURES_PATH":          None,
    "ROUTER_PROMPT":          build_router_prompt([], [], [], []),
    "RESPONDER_PROMPT":       build_responder_prompt([], [], [], []),
    "prev_question_timestamp": datetime.datetime.fromtimestamp(0),
    "initial_question":       None,
    "selected_suggestion":    None,
    "validation_status":      None,
    "validation_message":     None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  OLLAMA CHECK
# ─────────────────────────────────────────────
if not check_ollama_connection():
    st.error("⚠️  Ollama is not connected — please start the Ollama server and refresh.")
    st.stop()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">📡</div>
        <div class="brand-name">SenseTimeAI</div>
        <div class="brand-sub">Sensor Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)

    dataset_path = st.text_input(
        "Local dataset path",
        value="",
        placeholder="/path/to/your/dataset",
        label_visibility="collapsed",
    )

    if st.button("✦  Validate Dataset", use_container_width=True):
        p = dataset_path.strip()
        if p and os.path.exists(p) and os.path.isdir(p):
            is_ok, reason = validate_dataset_structure(p)
            if is_ok:
                st.session_state.validation_status  = "success"
                st.session_state.validation_message = "Dataset structure is valid."
                st.session_state.DATASET_PATH       = p
                names, types, conds, faults = scan_dataset_metadata(p)
                st.session_state.ROUTER_PROMPT    = build_router_prompt(names, types, conds, faults)
                st.session_state.RESPONDER_PROMPT = build_responder_prompt(names, types, conds, faults)

                folder_name = os.path.basename(p)
                if folder_name in os.listdir(FEATURES_DIR):
                    existing_folder_dialog()
                else:
                    st.session_state.FEATURES_PATH = folder_name
            else:
                st.session_state.validation_status  = "error"
                st.session_state.validation_message = f"Invalid structure: {reason}"
                st.session_state.DATASET_PATH       = None
        else:
            st.session_state.validation_status  = "error"
            st.session_state.validation_message = "Path not found or not a directory."
            st.session_state.DATASET_PATH       = None

    if st.session_state.validation_status == "success":
        st.success(f"✅ {st.session_state.validation_message}")
        folder = os.path.basename(st.session_state.DATASET_PATH)
        st.markdown(f'<div class="dataset-loaded">📁 {folder}</div>', unsafe_allow_html=True)
    elif st.session_state.validation_status == "error":
        st.error(f"❌ {st.session_state.validation_message}")

    st.markdown("""
    <div class="sidebar-info-block">
        <strong>Supported formats</strong><br>
        CSV · Parquet · HDF5<br><br>
        <strong>Expected structure</strong><br>
        Folders per condition, files per sensor
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FLAGS
# ─────────────────────────────────────────────
has_history        = len(st.session_state.messages) > 0
just_typed         = bool(st.session_state.get("initial_question"))
just_clicked       = bool(st.session_state.get("selected_suggestion"))
first_interaction  = just_typed or just_clicked


# ─────────────────────────────────────────────
#  LANDING SCREEN
# ─────────────────────────────────────────────
if not first_interaction and not has_history:

    st.markdown("""
    <div class="landing-wrapper">
        <div class="landing-badge">
            <span>●</span> Sensor Intelligence Engine
        </div>
        <div class="landing-headline">SenseTime<span style="opacity:0.5">AI</span></div>
        <div class="landing-sub">Raw sensor data → instant analysis</div>
        <div class="landing-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # Centered chat input
    _, center, _ = st.columns([1, 2.5, 1])
    with center:
        st.chat_input(
            "Ask anything about your sensor data…",
            key="initial_question",
        )

    st.stop()


# ─────────────────────────────────────────────
#  ACTIVE CHAT VIEW — title row
# ─────────────────────────────────────────────
title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("SenseTimeAI", anchor=False, width="stretch")

    def clear_conversation():
        for k in ["messages", "initial_question", "selected_suggestion",
                  "DATASET_PATH", "FEATURES_PATH", "validation_status", "validation_message"]:
            if k in st.session_state:
                st.session_state[k] = None if k in ["DATASET_PATH","FEATURES_PATH","validation_status","validation_message","initial_question","selected_suggestion"] else []
        st.session_state.ROUTER_PROMPT    = build_router_prompt([], [], [], [])
        st.session_state.RESPONDER_PROMPT = build_responder_prompt([], [], [], [])
        st.session_state.messages         = []

    st.button("↺ New Chat", on_click=clear_conversation)


# ─────────────────────────────────────────────
#  RESOLVE CURRENT USER MESSAGE
# ─────────────────────────────────────────────
followup_input = st.chat_input("Ask a follow-up question…")

user_message = followup_input
if not user_message:
    if just_typed:
        user_message = st.session_state.initial_question
    if just_clicked:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]


# ─────────────────────────────────────────────
#  RENDER HISTORY
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("images"):
            for fig in msg["images"]:
                st.pyplot(fig)
        if msg["role"] == "assistant":
            st.container()   # ghost-message fix
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
#  HANDLE NEW MESSAGE
# ─────────────────────────────────────────────
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.text(user_message.replace("$", r"\$"))

    with st.chat_message("assistant"):
        # Rate-limit
        now       = datetime.datetime.now()
        time_diff = now - st.session_state.prev_question_timestamp
        st.session_state.prev_question_timestamp = now
        if time_diff < MIN_TIME_BETWEEN_REQUESTS:
            wait = (MIN_TIME_BETWEEN_REQUESTS - time_diff).total_seconds()
            time.sleep(wait)

        history_copy = st.session_state.messages.copy()

        status_slot = st.empty()
        text_slot   = st.empty()

        response_text    = ""
        response_figures = []

        gen = generate_ollma_response(user_message, history_copy)

        for chunk in gen:
            if isinstance(chunk, tuple):
                kind, val = chunk
                if kind == "__STATUS__":
                    if val == "COMPLETE":
                        status_slot.empty()
                    else:
                        status_slot.markdown(f"*{val}*")
                elif kind == "__FIGURES__":
                    response_figures = val
                continue

            response_text += chunk
            text_slot.markdown(response_text + "▌")
            time.sleep(0.008)

        text_slot.markdown(response_text)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "images":  response_figures,
    })