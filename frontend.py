import asyncio
import html
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from main import app


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Constants
# =============================================================================

APP_TITLE = "AI Travel Booking System"
TRAVEL_PLAN_DIR = Path(__file__).resolve().parent / "travel_plans"

QUICK_PROMPTS = [
    "Plan a 7-day Japan trip from Hyderabad under ₹2 lakhs",
    "Plan a 5-day Paris trip from Bengaluru for a couple",
    "Plan a Dubai weekend trip from Hyderabad",
    "Plan a 10-day budget backpacking trip to Bali",
]

DESTINATIONS = [
    (
        "🇯🇵 Tokyo",
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=500&q=75",
    ),
    (
        "🇫🇷 Paris",
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=500&q=75",
    ),
    (
        "🇹🇭 Bangkok",
        "https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=500&q=75",
    ),
    (
        "🇮🇹 Rome",
        "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=500&q=75",
    ),
    (
        "🇦🇪 Dubai",
        "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=500&q=75",
    ),
]

REQUIRED_ENV_VARS = {
    "GROQ_API_KEY": "Groq",
    "TAVILY_API_KEY": "Tavily",
    "OPENWEATHER_API_KEY": "OpenWeather",
    "AVIATION_STACK_API_KEY": "AviationStack",
}


# =============================================================================
# Session state
# =============================================================================

def initialize_session_state() -> None:
    """Initialize persistent Streamlit session values."""
    defaults = {
        "thread_id": f"traveler_{uuid.uuid4().hex[:8]}",
        "travel_query": "",
        "last_result": None,
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #070b12;
    --surface: #0d1521;
    --surface-2: #101c2c;
    --border: #1d3047;
    --text: #eaf4ff;
    --muted: #8fb0cc;
    --primary: #4ea8f0;
    --primary-dark: #0d4a8a;
    --success: #50c878;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(30, 94, 160, 0.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(79, 168, 240, 0.08), transparent 35%),
        var(--bg);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero-wrapper {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 1.6rem;
    min-height: 300px;
    border: 1px solid rgba(92, 157, 220, 0.22);
    box-shadow: 0 22px 70px rgba(0,0,0,0.35);
}

.hero-bg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.30) saturate(0.8);
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(90deg, rgba(4,10,18,0.82), rgba(4,10,18,0.25)),
        linear-gradient(0deg, rgba(4,10,18,0.62), transparent);
}

.hero-content {
    position: relative;
    z-index: 2;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2.7rem;
}

.hero-badge {
    width: fit-content;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    border: 1px solid rgba(78,168,240,0.45);
    background: rgba(78,168,240,0.12);
    color: #82c4f6;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(2rem, 5vw, 3.7rem);
    line-height: 1.05;
    font-weight: 800;
    margin: 0;
    max-width: 850px;
}

.hero-subtitle {
    margin-top: 1rem;
    color: #b5cce0;
    font-size: 1.03rem;
    line-height: 1.7;
    max-width: 780px;
}

/* Cards */
.panel {
    background: linear-gradient(160deg, rgba(15,26,42,0.98), rgba(9,17,28,0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.35rem;
    box-shadow: 0 14px 40px rgba(0,0,0,0.20);
}

.destination-card {
    position: relative;
    overflow: hidden;
    min-height: 108px;
    border-radius: 14px;
    border: 1px solid rgba(120,170,220,0.16);
}

.destination-card img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.48);
    transition: transform 0.25s ease;
}

.destination-card:hover img {
    transform: scale(1.05);
}

.destination-card span {
    position: absolute;
    left: 0.8rem;
    bottom: 0.75rem;
    color: #fff;
    font-weight: 700;
    font-size: 0.9rem;
    text-shadow: 0 2px 8px rgba(0,0,0,0.8);
}

.section-title {
    color: var(--text);
    font-size: 1.08rem;
    font-weight: 700;
    margin: 1.7rem 0 0.75rem;
}

.section-kicker {
    color: var(--primary);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

.agent-card {
    background: #0c1725;
    border: 1px solid #1b3048;
    border-radius: 14px;
    padding: 1rem 1.05rem;
    height: 100%;
}

.agent-card-title {
    color: #eaf4ff;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.agent-card-body {
    color: #9dbbd4;
    font-size: 0.88rem;
    line-height: 1.55;
}

.metric-card {
    background: #0d1724;
    border: 1px solid #1e3249;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    color: var(--primary);
    font-size: 1.75rem;
    font-weight: 800;
}

.metric-label {
    color: #8ba9c2;
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Streamlit widgets */
.stTextArea textarea,
.stTextInput input {
    background: #09131f !important;
    border: 1px solid #1d334b !important;
    border-radius: 12px !important;
    color: #eaf4ff !important;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #4ea8f0 !important;
    box-shadow: 0 0 0 2px rgba(78,168,240,0.18) !important;
}

div[data-testid="stButton"] > button {
    width: 100%;
    border: 1px solid rgba(100,180,245,0.25) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2182d5, #0e4f91) !important;
    color: white !important;
    min-height: 3.1rem;
    box-shadow: 0 10px 28px rgba(16,97,170,0.30);
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: #4ea8f0 !important;
}

div[data-testid="stDownloadButton"] > button {
    width: 100%;
    background: #153552 !important;
    color: #edf7ff !important;
    border: 1px solid #2d5d86 !important;
    border-radius: 12px !important;
}

[data-testid="stStatusWidget"] {
    background: #0b1623 !important;
    border: 1px solid #1e344d !important;
    border-radius: 14px !important;
}

.stAlert {
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] {
    background: #080e17 !important;
    border-right: 1px solid #17263a;
}

.sidebar-title {
    color: #edf7ff;
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.sidebar-chip {
    background: #0e1a29;
    border: 1px solid #192f47;
    border-radius: 10px;
    padding: 0.55rem 0.75rem;
    margin-bottom: 0.45rem;
    color: #a9c7df;
    font-size: 0.84rem;
}

.small-muted {
    color: #7293ad;
    font-size: 0.78rem;
    line-height: 1.5;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Helper functions
# =============================================================================

def get_secret(name: str) -> str | None:
    """Read a secret from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name)


def get_missing_integrations() -> list[str]:
    """Return user-friendly names for integrations with missing credentials."""
    missing = []
    for env_name, display_name in REQUIRED_ENV_VARS.items():
        if not get_secret(env_name):
            missing.append(display_name)
    return missing


def normalize_thread_id(value: str) -> str:
    """Create a safe LangGraph thread identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return cleaned[:80] or f"traveler_{uuid.uuid4().hex[:8]}"


def extract_message_content(value: Any) -> str:
    """Safely extract text from a LangChain message or plain object."""
    if value is None:
        return ""

    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def result_text(result: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty textual result from possible state keys."""
    for key in keys:
        value = result.get(key)
        if value:
            return extract_message_content(value)

    return ""


def final_response_from_state(result: dict[str, Any]) -> str:
    """Resolve the final answer from common LangGraph state shapes."""
    direct = result_text(
        result,
        "final_response",
        "final_plan",
        "travel_plan",
        "response",
    )
    if direct:
        return direct

    messages = result.get("messages") or []
    if messages:
        return extract_message_content(messages[-1])

    return ""


async def invoke_travel_graph(
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the asynchronous LangGraph workflow."""
    return await app.ainvoke(
        initial_state,
        config=config,
    )


def run_travel_graph(
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the asynchronous graph from Streamlit.

    Streamlit normally executes the script without an active event loop, so
    asyncio.run is appropriate both locally and on Streamlit Community Cloud.
    """
    return asyncio.run(
        invoke_travel_graph(
            initial_state=initial_state,
            config=config,
        )
    )


def create_markdown_plan(
    *,
    user_query: str,
    thread_id: str,
    flight_results: str,
    hotel_results: str,
    weather_results: str,
    itinerary: str,
    final_response: str,
    llm_calls: int,
) -> tuple[str, str]:
    """Build downloadable Markdown content and a timestamped filename."""
    now = datetime.now()
    filename = f"travel_plan_{now.strftime('%Y%m%d_%H%M%S')}.md"

    content = f"""# AI Travel Plan

**Query:** {user_query}  
**Generated:** {now.strftime("%Y-%m-%d %H:%M:%S")}  
**Session ID:** {thread_id}

---

## ✈️ Flight Information

{flight_results or "No flight information was available."}

---

## 🏨 Hotel Recommendations

{hotel_results or "No hotel information was available."}

---

## 🌦️ Weather Information

{weather_results or "No weather information was available."}

---

## 🗓️ Day-wise Itinerary

{itinerary or "No itinerary was generated."}

---

## 🧠 Final Travel Plan

{final_response or "No final travel plan was generated."}

---

**LLM calls:** {llm_calls}
"""

    return filename, content


def save_plan_locally(filename: str, content: str) -> tuple[bool, str]:
    """
    Save the plan when the deployment filesystem allows it.

    Streamlit Cloud storage is ephemeral, so the download button remains the
    reliable way for the user to keep a plan.
    """
    try:
        TRAVEL_PLAN_DIR.mkdir(parents=True, exist_ok=True)
        file_path = TRAVEL_PLAN_DIR / filename
        file_path.write_text(content, encoding="utf-8")
        return True, str(file_path)
    except OSError as exc:
        return False, str(exc)


def render_result_section(title: str, icon: str, content: str) -> None:
    """Render one agent result safely as Markdown."""
    with st.expander(f"{icon} {title}", expanded=False):
        if content:
            st.markdown(content)
        else:
            st.info(f"{title} did not return data, but the remaining workflow continued.")


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>🌍 AI Travel Planner</div>",
        unsafe_allow_html=True,
    )

    thread_input = st.text_input(
        "Session ID",
        value=st.session_state.thread_id,
        help="LangGraph uses this ID to preserve the conversation with MemorySaver.",
    )
    st.session_state.thread_id = normalize_thread_id(thread_input)

    if st.button("Create new session", use_container_width=True):
        st.session_state.thread_id = f"traveler_{uuid.uuid4().hex[:8]}"
        st.session_state.last_result = None
        st.session_state.travel_query = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div class='sidebar-title'>Technology</div>",
        unsafe_allow_html=True,
    )

    for technology in [
        "🔗 LangGraph StateGraph",
        "🧠 Groq · Llama 3.3 70B",
        "💾 MemorySaver",
        "🔌 Model Context Protocol",
        "🔍 Tavily Search",
        "🌦️ OpenWeather",
        "✈️ AviationStack",
    ]:
        st.markdown(
            f"<div class='sidebar-chip'>{technology}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='sidebar-title' style='margin-top:1.2rem;'>Agent pipeline</div>",
        unsafe_allow_html=True,
    )

    for pipeline_step in [
        "① Flight Agent",
        "② Hotel Agent",
        "③ Weather Agent",
        "④ Itinerary Agent",
        "⑤ Final Synthesis Agent",
    ]:
        st.markdown(
            f"<div class='sidebar-chip'>{pipeline_step}</div>",
            unsafe_allow_html=True,
        )

    missing_integrations = get_missing_integrations()
    st.markdown("---")
    if missing_integrations:
        st.warning(
            "Missing configuration: " + ", ".join(missing_integrations),
            icon="⚠️",
        )
    else:
        st.success("All API integrations configured.", icon="✅")

    st.markdown(
        "<div class='small-muted'>"
        "MemorySaver keeps state only while the running Streamlit process remains alive. "
        "For durable production memory, use a persistent checkpointer later."
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Main UI
# =============================================================================

st.markdown(
    """
<div class="hero-wrapper">
    <img
        class="hero-bg"
        src="https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1600&q=85"
        alt="Airplane flying above clouds"
    />
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <div class="hero-badge">Multi-agent AI travel intelligence</div>
        <h1 class="hero-title">Plan an entire journey with specialized AI agents.</h1>
        <div class="hero-subtitle">
            Flights, hotels, worldwide weather, budget guidance and a day-wise
            itinerary are coordinated through an asynchronous LangGraph workflow.
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

destination_columns = st.columns(len(DESTINATIONS))
for column, (name, image_url) in zip(destination_columns, DESTINATIONS):
    with column:
        st.markdown(
            f"""
<div class="destination-card">
    <img src="{image_url}" alt="{html.escape(name)}"/>
    <span>{html.escape(name)}</span>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='section-title'>Describe your journey</div>",
    unsafe_allow_html=True,
)

quick_columns = st.columns(len(QUICK_PROMPTS))
for index, (column, prompt) in enumerate(zip(quick_columns, QUICK_PROMPTS)):
    with column:
        if st.button(prompt, key=f"quick_prompt_{index}", use_container_width=True):
            st.session_state.travel_query = prompt
            st.rerun()

st.text_area(
    "Travel request",
    key="travel_query",
    placeholder=(
        "Example: Plan a complete 7-day Japan trip from Hyderabad for two people, "
        "including flights, hotels, weather, sightseeing and a ₹2 lakh budget."
    ),
    height=130,
    label_visibility="collapsed",
)

generate = st.button(
    "🚀 Generate my travel plan",
    type="primary",
    use_container_width=True,
)


# =============================================================================
# Graph execution
# =============================================================================

if generate:
    user_query = st.session_state.travel_query.strip()

    if not user_query:
        st.warning("Please describe your trip before generating the plan.")
    else:
        safe_thread_id = normalize_thread_id(st.session_state.thread_id)

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "destination": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "weather_info": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
            "errors": [],
        }

        config = {
            "configurable": {
                "thread_id": safe_thread_id,
            }
        }

        st.session_state.last_error = None

        with st.status(
            "Running the asynchronous multi-agent workflow...",
            expanded=True,
        ) as workflow_status:
            st.write("✈️ Flight Agent is checking routes and aviation information.")
            st.write("🏨 Hotel Agent is researching suitable stays.")
            st.write("🌦️ Weather Agent is checking destination conditions.")
            st.write("🗓️ Itinerary Agent is organizing the trip day by day.")
            st.write("🧠 Final Agent is combining all recommendations.")

            try:
                result = run_travel_graph(
                    initial_state=initial_state,
                    config=config,
                )
                st.session_state.last_result = result
                workflow_status.update(
                    label="Travel plan generated successfully",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                st.session_state.last_error = str(exc)
                workflow_status.update(
                    label="The workflow encountered an error",
                    state="error",
                    expanded=True,
                )

        if st.session_state.last_error:
            st.error(
                "The graph could not complete. Verify your API keys, MCP server "
                "configuration and application logs."
            )
            with st.expander("Technical error details"):
                st.code(st.session_state.last_error)


# =============================================================================
# Result rendering
# =============================================================================

if st.session_state.last_result:
    result = st.session_state.last_result

    flight_results = result_text(
        result,
        "flight_results",
        "flight_info",
        "flights",
    )
    hotel_results = result_text(
        result,
        "hotel_results",
        "hotel_info",
        "hotels",
    )
    weather_results = result_text(
        result,
        "weather_results",
        "weather_info",
        "weather",
    )
    itinerary = result_text(
        result,
        "itinerary",
        "day_wise_itinerary",
    )
    final_response = final_response_from_state(result)

    raw_llm_calls = result.get("llm_calls", 0)
    try:
        llm_calls = int(raw_llm_calls)
    except (TypeError, ValueError):
        llm_calls = 0

    st.markdown(
        "<div class='section-title'>Agent results</div>",
        unsafe_allow_html=True,
    )

    result_columns = st.columns(5)
    cards = [
        ("✈️", "Flight Agent", bool(flight_results)),
        ("🏨", "Hotel Agent", bool(hotel_results)),
        ("🌦️", "Weather Agent", bool(weather_results)),
        ("🗓️", "Itinerary Agent", bool(itinerary)),
        ("🧠", "Final Agent", bool(final_response)),
    ]

    for column, (icon, label, completed) in zip(result_columns, cards):
        with column:
            state_text = "Completed" if completed else "No data"
            st.markdown(
                f"""
<div class="agent-card">
    <div class="agent-card-title">{icon} {label}</div>
    <div class="agent-card-body">{state_text}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    metric_columns = st.columns(3)
    with metric_columns[0]:
        st.markdown(
            """
<div class="metric-card">
    <div class="metric-value">5</div>
    <div class="metric-label">Agents</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with metric_columns[1]:
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-value">{llm_calls}</div>
    <div class="metric-label">LLM calls</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with metric_columns[2]:
        completed_count = sum(1 for _, _, completed in cards if completed)
        st.markdown(
            f"""
<div class="metric-card">
    <div class="metric-value">{completed_count}/5</div>
    <div class="metric-label">Completed</div>
</div>
""",
            unsafe_allow_html=True,
        )

    render_result_section("Flight information", "✈️", flight_results)
    render_result_section("Hotel recommendations", "🏨", hotel_results)
    render_result_section("Weather information", "🌦️", weather_results)
    render_result_section("Day-wise itinerary", "🗓️", itinerary)

    st.markdown(
        "<div class='section-title'>Final travel plan</div>",
        unsafe_allow_html=True,
    )

    if final_response:
        with st.container(border=True):
            st.markdown(final_response)
    else:
        st.warning(
            "The graph completed, but no final response was found in the returned state."
        )

    filename, file_content = create_markdown_plan(
        user_query=st.session_state.travel_query,
        thread_id=st.session_state.thread_id,
        flight_results=flight_results,
        hotel_results=hotel_results,
        weather_results=weather_results,
        itinerary=itinerary,
        final_response=final_response,
        llm_calls=llm_calls,
    )

    saved, save_message = save_plan_locally(filename, file_content)

    download_column, save_column = st.columns([1, 2])
    with download_column:
        st.download_button(
            "⬇️ Download travel plan",
            data=file_content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )

    with save_column:
        if saved:
            st.success(
                "A local copy was saved in the travel_plans directory. "
                "On Streamlit Cloud, local files are temporary."
            )
        else:
            st.info(
                "The plan is ready to download. A server-side copy could not be "
                f"saved: {save_message}"
            )