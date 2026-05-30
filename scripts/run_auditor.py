import streamlit as st
import joblib
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- THE BULLETPROOF PATH FIX ---
# Find the exact path to your scripts folder and force it into Python's priority list
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Now Python has no choice but to find these files directly
from financials import get_live_features
from transcript_loader import get_qa_transcript
# --------------------------------

# Load API Key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(layout="wide", page_title="RiskSight Auditor")
st.title("RiskSight: AI Junior Analyst")

# Initialize Session State for Memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audit_memo" not in st.session_state:
    st.session_state.audit_memo = None

# ================= SIDEBAR: AUDIT CONTROLS =================
ticker = st.sidebar.text_input("Enter Ticker (e.g., RELIANCE)")

if st.sidebar.button("Run Audit"):
    if not ticker:
        st.sidebar.error("Please enter a ticker.")
    else:
        # Format ticker
        ticker = ticker if ticker.endswith('.NS') else f"{ticker}.NS"
        MODEL_PATH = os.path.join("data", "risksight_nse_rf_model.pkl")
        
        try:
            # 1. Math Alarm
            rf_model = joblib.load(MODEL_PATH)
            input_data = get_live_features(ticker)
            input_data = input_data.reindex(columns=rf_model.feature_names_in_, fill_value=0)
            
            prediction = rf_model.predict(input_data)[0]
            
            if prediction == 1:
                st.sidebar.warning("🚨 Distress Flagged! Investigating...")
                
                # 2. RAG Analysis
                qa_text = get_qa_transcript(ticker)
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_KEY)
                
                prompt = (
                    "You are a Senior Financial Auditor. A Random Forest model has flagged "
                    f"{ticker} for financial distress based on quantitative ratios.\n"
                    "Analyze the following Earnings Call Q&A to see if management provides a "
                    "valid strategic reason for this.\n\n"
                    f"Q&A Transcript: {qa_text}\n\n"
                    "Provide a final verdict: Is this risk acceptable (strategic) or critical?"
                )
                
                with st.spinner("Generating Audit Memo..."):
                    response = llm.invoke(prompt)
                    st.session_state.audit_memo = response.content
                    
                    # Reset chat history for the new company
                    st.session_state.messages = [{"role": "assistant", "content": f"I have completed the audit for {ticker}. What would you like to discuss about the findings?"}]
                    
            else:
                st.sidebar.success("✅ Healthy Status. No investigation needed.")
                st.session_state.audit_memo = f"**{ticker}** is financially healthy based on our quantitative models. No further RAG investigation required."
                st.session_state.messages = []
                
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# ================= MAIN INTERFACE: SPLIT LAYOUT =================
col1, col2 = st.columns([1, 1])

# Left Column: The Report
with col1:
    st.subheader("📝 Audit Memo")
    if st.session_state.audit_memo:
        st.write(st.session_state.audit_memo)
    else:
        st.info("Enter a ticker in the sidebar and click 'Run Audit' to begin.")

# Right Column: The Chat
with col2:
    st.subheader("💬 Chat with the Auditor")
    
    # Render previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Chat Input Box
    if prompt := st.chat_input("Ask a follow-up about the audit..."):
        if not st.session_state.audit_memo:
            st.warning("Please run an audit first.")
        else:
            # 1. Display User Message
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Generate AI Response (with memory)
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_KEY)
            
            # Build the context string so the AI remembers what it wrote in the memo
            system_context = f"You are an AI financial auditor. You just wrote this audit memo:\n{st.session_state.audit_memo}\nAnswer the user's follow-up questions based on this."
            
            # Compile history
            messages_for_llm = [("system", system_context)]
            for m in st.session_state.messages:
                messages_for_llm.append((m["role"], m["content"]))
                
            with st.spinner("Thinking..."):
                response = llm.invoke(messages_for_llm)
                
            # 3. Display AI Response
            st.chat_message("assistant").write(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})