# app.py
import streamlit as st
import copy
import os
import json
from dotenv import load_dotenv
load_dotenv('.env', override=True)

from src.agent import pc_config_agent, DEFAULT_INITIAL_STATE
from src.agents.response_agent import stream_final_response

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
    st.session_state.messages = []
if "build_ready" not in st.session_state:
    st.session_state.build_ready = False

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🖥️ AI PC Builder Agent")
st.caption("Tell me your budget, use case, and preferences — I'll build the perfect PC for you.")

# ── Sidebar — build reference panel ──────────────────────────────────────────
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

        st.download_button(
            label="📥 Download Build (JSON)",
            data=json.dumps(build, indent=2),
            file_name="pc_build.json",
            mime="application/json"
        )
    else:
        st.info("No build assembled yet. Start chatting!")

    st.divider()

    logs = st.session_state.agent_state.get("logs", [])
    if logs:
        with st.expander("🔍 Agent Trace Logs", expanded=False):
            for log in logs:
                st.text(f"→ {log}")

    if st.button("🔄 Start New Build", use_container_width=True):
        st.session_state.agent_state = copy.deepcopy(DEFAULT_INITIAL_STATE)
        st.session_state.messages = []
        st.session_state.build_ready = False
        st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("e.g. I want a $1000 gaming PC with AMD CPU and NVIDIA GPU...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Append to agent chat history
    st.session_state.agent_state["chat_history"].append({
        "role": "user",
        "content": user_input
    })

    # ── Run agent ─────────────────────────────────────────────────────────
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

        final_response = result_state.get("final_response")
        next_step = result_state.get("next_step")
        build = result_state.get("current_build")

        # ── Case 1: Build ready — stream explanation in main chat ─────────
        if final_response == "build_ready" and build:
            requirements = result_state.get("user_requirements", {})

            # Render a compact build card in the chat before the explanation
            with st.expander("📋 Build Summary", expanded=True):
                cols = st.columns(2)
                component_list = {
                    "🧠 CPU": build.get("cpu"),
                    "🎮 GPU": build.get("gpu"),
                    "🧩 Motherboard": build.get("motherboard"),
                    "💾 RAM": build.get("ram"),
                    "💿 Storage": build.get("storage"),
                    "⚡ PSU": build.get("psu"),
                    "📦 Case": build.get("case"),
                }
                items = [(k, v) for k, v in component_list.items() if v]
                for i, (label, comp) in enumerate(items):
                    with cols[i % 2]:
                        st.markdown(f"**{label}** — {comp.get('name', 'N/A')}")
                        st.markdown(f"`${comp.get('price', 0):.2f}`")

                st.markdown(f"**💰 Total: ${build.get('total_price', 0):.2f}**")
                compatible = build.get("is_compatible", False)
                st.markdown(
                    "✅ **Compatible**" if compatible else "⚠️ **Compatibility issues — see notes**"
                )
                if build.get("compatibility_notes"):
                    st.caption(build["compatibility_notes"])

            # Stream the conversational explanation below the card
            st.markdown("---")
            streamed_text = st.write_stream(
                stream_final_response(build, requirements)
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"[Build ready — see summary above]\n\n{streamed_text}"
            })
            st.session_state.build_ready = True

        # ── Case 2: Clarification / no-build / any other message ─────────
        else:
            msg = final_response or "I wasn't able to assemble a build. Please try again."
            st.markdown(msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })

    st.rerun()
