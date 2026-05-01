import streamlit as st
import time
from chat import FinancialAssistant

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (The "Premium" Polish)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus | Enterprise SEC RAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting Custom CSS to hide default Streamlit branding and polish the UI
st.markdown("""
    <style>
        /* Hide Streamlit Header and Footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Polish the chat bubbles */
        .stChatMessage {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        /* Custom Title Styling */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A; /* Corporate Blue */
            margin-bottom: 0rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# 2. SIDEBAR TELEMETRY (Building User Trust)
# -------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png",
             width=60)  # Generic professional icon
    st.markdown("### Nexus RAG Engine")
    st.caption("v2.1.0 | Air-Gapped Deployment")

    st.divider()

    # System Metrics
    st.markdown("#### System Telemetry")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Active Chunks", value="20,256")
        st.metric(label="Retrieval", value="Hybrid",
                  delta="RRF + BM25", delta_color="normal")
    with col2:
        st.metric(label="Tickers", value="10")
        st.metric(label="Re-ranker", value="Active",
                  delta="MS-MARCO", delta_color="normal")

    st.divider()
    st.markdown("#### LLM Synthesis")
    st.info("🧠 Model: **Llama-3 (Local)**\n\n🔒 Status: **Air-Gapped**\n\n🌡️ Temp: **0.1 (Strict)**")

    st.divider()
    # Reset Chat Button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.toast("Conversation history cleared.", icon="✅")


# -------------------------------------------------------------------
# 3. MAIN UI HEADER
# -------------------------------------------------------------------
st.markdown('<p class="main-title">🏛️ SEC Compliance Assistant</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Query over 20,000 regulatory documents across 10 Fortune 500 companies with strict citation gating.</p>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# 4. INITIALIZE STATE & BACKEND
# -------------------------------------------------------------------
# Boot up the heavy AI engine only once
if "assistant" not in st.session_state:
    with st.status("Booting Neural Search Engine...", expanded=True) as status:
        st.write("Connecting to Local ChromaDB...")
        time.sleep(0.5)  # Slight delay for visual UX
        st.write("Loading BM25 Sparse Index...")
        time.sleep(0.5)
        st.write("Warming up MS-MARCO Cross-Encoder...")
        st.session_state.assistant = FinancialAssistant()
        status.update(label="System Online", state="complete", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am synced to the latest 10-K filings for AAPL, MSFT, TSLA, JPM, and 6 other major equities. How can I assist your analysis today?"}
    ]


# -------------------------------------------------------------------
# 5. CHAT INTERFACE
# -------------------------------------------------------------------
# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("E.g., What are JPMorgan's primary risks regarding interest rates?"):

    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process AI Response with UX Spinners
    with st.chat_message("assistant"):
        # The st.status box looks highly professional for multi-step processes
        with st.status("Analyzing Corporate Filings...", expanded=True) as status:
            st.write("Executing Hybrid Search (Vector + Keyword)...")
            st.write("Re-ranking Top 15 candidates with Cross-Encoder...")
            st.write("Synthesizing context via local Llama-3...")

            # Execute the actual backend call
            start_time = time.time()
            response = st.session_state.assistant.generate_answer(prompt)
            end_time = time.time()

            status.update(
                label=f"Analysis complete ({end_time - start_time:.1f}s)", state="complete", expanded=False)

        # 3. Display the final answer
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
