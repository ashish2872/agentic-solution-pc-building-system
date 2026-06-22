# app.py
import streamlit as st
import copy
import json
from src.agent import pc_config_agent, DEFAULT_INITIAL_STATE, stream_final_response
from dotenv import load_dotenv
load_dotenv('.env')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PC Builder Agent",
    page_icon="🖥️",
    layout="wide"
)

# ── Session state init ────────────────────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state = copy.deepcopy(DEFAULT_INITIAL_STATE)

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat display history

if "build_ready" not in st.session_state:
    st.session_state.build_ready = False

if "awaiting_user" not in st.session_state:
    st.session_state.awaiting_user = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🖥️ AI PC Builder Agent")
st.caption("Tell me your budget, use case, and preferences — I'll build the perfect PC for you.")

# ── Sidebar — current build status ───────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Current Build")

    build = st.session_state.agent_state.get("current_build")

    if build:
        components = {
            "🧠 CPU": build.get("cpu"),
            "🎮 GPU": build.get("gpu"),
            "🧩 Motherboard": build.get("motherboard"),
            "💾 RAM": build.get("ram"),
            "💿 Storage": build.get("storage"),
            "⚡ PSU": build.get("psu"),
            "📦 Case": build.get("case"),
        }

        for label, component in components.items():
            if component:
                st.markdown(f"**{label}**")
                st.markdown(f"  {component.get('name', 'N/A')}")
                st.markdown(f"  `${component.get('price', 0):.2f}`")
                st.divider()

        total = build.get("total_price", 0)
        compatible = build.get("is_compatible", False)

        st.metric("💰 Total Price", f"${total:.2f}")
        st.metric(
            "✅ Compatible" if compatible else "❌ Not Compatible",
            "Yes" if compatible else "No"
        )

        if build.get("compatibility_notes"):
            st.info(build["compatibility_notes"])

        # Download build as JSON
        st.download_button(
            label="📥 Download Build (JSON)",
            data=json.dumps(build, indent=2),
            file_name="pc_build.json",
            mime="application/json"
        )

    else:
        st.info("No build assembled yet. Start chatting!")

    st.divider()

    # Agent logs expander
    logs = st.session_state.agent_state.get("logs", [])
    if logs:
        with st.expander("🔍 Agent Trace Logs", expanded=False):
            for log in logs:
                st.text(f"→ {log}")

    # Reset button
    if st.button("🔄 Start New Build", use_container_width=True):
        st.session_state.agent_state = copy.deepcopy(DEFAULT_INITIAL_STATE)
        st.session_state.messages = []
        st.session_state.build_ready = False
        st.session_state.awaiting_user = False
        st.rerun()

# ── Chat display ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("e.g. I want a $1000 gaming PC with AMD CPU and NVIDIA GPU...")

if user_input:
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Append to agent chat history
    st.session_state.agent_state["chat_history"].append({
        "role": "user",
        "content": user_input
    })

    # ── Run the agent graph ───────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking..."):
            try:
                result_state = pc_config_agent.invoke(
                    st.session_state.agent_state
                )
                st.session_state.agent_state = result_state
            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.stop()

        next_step = result_state.get("next_step")
        final_response = result_state.get("final_response")
        build = result_state.get("current_build")

        # ── Clarification needed ──────────────────────────────────────────
        if next_step == "awaiting_user":
            st.markdown(final_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })
            st.session_state.awaiting_user = True

        # ── Build ready — stream the explanation ──────────────────────────
        elif next_step == "end" and final_response == "build_ready" and build:
            requirements = result_state.get("user_requirements", {})

            # Stream the response token by token
            streamed_text = st.write_stream(
                stream_final_response(build, requirements)
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": streamed_text
            })
            st.session_state.build_ready = True

        # ── Generic response (error, no build) ───────────────────────────
        else:
            msg = final_response or "I wasn't able to assemble a build. Please try again."
            st.markdown(msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })

    # Rerun to update sidebar with new build
    st.rerun()
