import streamlit as st
import joblib
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from streamlit_lottie import st_lottie

from scripts.financials import get_live_features
from scripts.transcript_loader import get_qa_transcript
import streamlit as st
import pandas as pd
from datetime import datetime
# Import your fixed live features function from financials.py
from financials import get_live_features  

# 1. Set the browser tab title and icon
st.set_page_config(
    page_title="RiskSight - AI Financial Risk Auditor",
    page_icon="🛡️",
    layout="wide"
)

# 2. Get and format the current date dynamically
current_date = datetime.now().strftime("%B %d, %Y")

# 3. Display the App Name and Live Date Banner
st.title("🛡️ RiskSight")
st.markdown(f"**AI-Powered Corporate Distress & Risk Identification Pipeline** | 📅 *As of: {current_date}*")
st.markdown("---")
# --- INITIALIZE SESSION STATE (MEMORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SETUP & CONFIG ---
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(layout="wide", page_title="RiskSight Auditor", page_icon="📊")

# --- LOTTIE ANIMATION HELPER ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Pre-load animations (using reliable public Lottie URLs)
lottie_loading = load_lottieurl("https://lottie.host/8192b0fa-d205-4c07-ba71-6c5d985a1a1f/0z8146uLq3.json") # AI Brain Scan
lottie_success = load_lottieurl("https://lottie.host/9e419dc1-b1e2-411a-84fb-03de77cd893b/0O7tY6E1G6.json") # Green Check
lottie_alert = load_lottieurl("https://lottie.host/d206f470-34bb-4e6e-ae5c-9c9c43d81b4d/2Qh3f5C8zH.json")   # Red Alert

# --- STATE MANAGEMENT (Prevents UI from vanishing during chat) ---
if "audit_complete" not in st.session_state:
    st.session_state.audit_complete = False
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "audit_memo" not in st.session_state:
    st.session_state.audit_memo = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR: CONTROLS =================
with st.sidebar:
    st.title("RiskSight Engine")
    st.markdown("---")
    ticker_input = st.text_input("Enter Ticker Symbol (e.g., RELIANCE)")
    
    if st.button("Run AI Audit", type="primary", use_container_width=True):
        if not ticker_input:
            st.error("Please enter a ticker.")
        else:
            # 1. Reset state for new run
            st.session_state.ticker = ticker_input if ticker_input.endswith('.NS') else f"{ticker_input}.NS"
            st.session_state.messages = []
            MODEL_PATH = os.path.join("data", "risksight_nse_rf_model_v2.pkl")
            
            with st.spinner("Analyzing Financials & Market Data..."):
                try:
                    # 2. ML Prediction
                    rf_model = joblib.load(MODEL_PATH)
                    input_data = get_live_features(st.session_state.ticker)
                    input_data = input_data.reindex(columns=rf_model.feature_names_in_, fill_value=0)
                    
                    st.session_state.raw_data = input_data
                    st.session_state.prediction = rf_model.predict(input_data)[0]
                    
                    # 3. LLM RAG Pipeline
                    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_KEY)
                    qa_text = get_qa_transcript(st.session_state.ticker)
                    
                    if st.session_state.prediction == 1:
                        prompt = (
                            "You are a Senior Financial Auditor. A quantitative Machine Learning model has flagged "
                            f"{st.session_state.ticker} for potential financial distress based on balance sheet ratios.\n"
                            "Analyze the following recent market news to determine if there is a known "
                            "strategic reason for this or if the risk is unmitigated.\n\n"
                            f"Contextual Data: {qa_text}\n\n"
                            "Provide a final Audit Verdict: Is this risk acceptable or critical?"
                        )
                        sys_msg = f"I have completed the distress audit for {st.session_state.ticker}. What would you like to discuss?"
                    else:
                        prompt = (
                            "You are a Senior Financial Analyst. A quantitative ML model has evaluated "
                            f"{st.session_state.ticker} and confirmed it has a strong, healthy financial position.\n"
                            "Review the following recent market news to provide a brief executive summary of what is "
                            "driving this company's success or current market sentiment.\n\n"
                            f"Contextual Data: {qa_text}\n\n"
                            "Provide a short Positive Audit Memo highlighting key strengths and catalysts."
                        )
                        sys_msg = f"I have verified {st.session_state.ticker} is in good financial health. What would you like to know?"
                    
                    # Generate Memo
                    response = llm.invoke(prompt)
                    st.session_state.audit_memo = response.content
                    st.session_state.messages = [{"role": "assistant", "content": sys_msg}]
                    st.session_state.audit_complete = True
                    
                except Exception as e:
                    st.error(f"System Error: {e}")

# ================= MAIN INTERFACE: DASHBOARD =================
st.title("RiskSight Dashboard")

# Default Screen (Before Audit)
if not st.session_state.audit_complete:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if lottie_loading:
            st_lottie(lottie_loading, height=300, key="idle_anim")
        st.info("👈 Enter a ticker symbol in the sidebar to initiate the AI Audit.")

# Audit Complete Screen
else:
    st.subheader(f"Analysis for: **{st.session_state.ticker}**")
    st.markdown("---")
    
    # 1. TOP ROW: Financial Metric Cards
    # We extract safe values to display on the dashboard
    df = st.session_state.raw_data
    cr_val = round(df['Current Ratio'].iloc[0], 2) if 'Current Ratio' in df else "N/A"
    de_val = round(df['Debt Equity Ratio'].iloc[0], 2) if 'Debt Equity Ratio' in df else "N/A"
    pm_val = round(df['Net Profit Margin'].iloc[0] * 100, 2) if 'Net Profit Margin' in df else "N/A"
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Current Ratio", value=cr_val)
    m2.metric(label="Debt to Equity", value=de_val)
    m3.metric(label="Net Profit Margin", value=f"{pm_val}%")
    
    with m4:
        # Dynamic Status Animation
        if st.session_state.prediction == 1:
            st.error("Status: DISTRESS")
            if lottie_alert: st_lottie(lottie_alert, height=50, key="alert")
        else:
            st.success("Status: HEALTHY")
            if lottie_success: st_lottie(lottie_success, height=50, key="success")

    st.markdown("---")
    
    # 2. MAIN BODY: Clean Tabs Architecture
    tab1, tab2, tab3 = st.tabs(["📝 Executive Memo", "💬 AI Auditor Chat", "📊 Raw Model Data"])
    
    # TAB 1: The Report
    with tab1:
        st.write(st.session_state.audit_memo)
        
    # TAB 2: The Chat Interface
    with tab2:
        # Display history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
        # Chat input
        if chat_input := st.chat_input("Ask a follow-up question..."):
            st.chat_message("user").write(chat_input)
            st.session_state.messages.append({"role": "user", "content": chat_input})
            
            # Setup context and call LLM
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_KEY)
            system_context = f"You are an AI financial auditor. You wrote this memo:\n{st.session_state.audit_memo}\nAnswer based on this."
            messages_for_llm = [("system", system_context)] + [(m["role"], m["content"]) for m in st.session_state.messages]
            
            with st.spinner("Thinking..."):
                response = llm.invoke(messages_for_llm)
                
            st.chat_message("assistant").write(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
    # TAB 3: Data Inspection
    with tab3:
        st.markdown("##### Underlying Machine Learning Features")
        st.dataframe(st.session_state.raw_data.T, use_container_width=True)
