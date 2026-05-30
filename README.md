# 📈 RiskSight: AI Financial Auditor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://risksight-hkdtllhbswncfgku7qc5p4.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**RiskSight** is an end-to-end, AI-powered quantitative financial auditing tool. It acts as an autonomous Junior Analyst—fetching live market data, running a Machine Learning risk assessment, and generating executive-level financial memos using a Llama 3.1 RAG pipeline.

[👉 **Try the Live Application Here**](https://risksight-hkdtllhbswncfgku7qc5p4.streamlit.app/)

---

## 🌟 Key Features

* **Live Data Engineering:** Fetches real-time financial statements and market news via `yfinance`. Features a robust fallback calculation engine to handle missing API data (specifically tailored for international NSE stocks).
* **Quantitative ML Alarm:** Utilizes a custom-trained **Random Forest Classifier** evaluating 18 critical financial metrics (Current Ratio, Debt-to-Equity, Profit Margins) to instantly flag potential corporate distress.
* **Dual-Path LLM Auditing:** * *Bear Case (Distress):* If the ML flags a risk, the LLM analyzes recent market news to determine if the risk is strategic (e.g., acquisitions) or critical.
    * *Bull Case (Healthy):* If cleared, the LLM generates a positive executive summary highlighting market catalysts and strengths.
* **Conversational RAG Chat:** Users can interact with the AI Auditor via a chat interface to ask follow-up questions about the generated memo, maintaining context via session state.
* **Modern SaaS Dashboard:** Built with Streamlit, featuring a glassmorphism bento-grid layout, dynamic metric cards, and vector (Lottie) animations.

---

## 🏗️ Architecture

1.  **Data Ingestion:** User inputs a ticker (e.g., `TCS.NS`, `NVDA`). `financials.py` extracts raw balance sheet and income statement data.
2.  **Feature Engineering:** Missing API data points are mathematically reconstructed or safely imputed using industry-neutral baselines to prevent false positives.
3.  **Prediction:** The `scikit-learn` model assesses the 18-feature array and outputs a binary health classification.
4.  **Generative AI Analysis:** `transcript_loader.py` fetches the latest market context. `LangChain` and `Groq` feed this context into **Llama 3.1 8b**, prompting it to write a context-aware audit memo.
5.  **State Management:** Streamlit session states lock the data in place, allowing seamless transitions between the report and the interactive chat module.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Streamlit-Lottie
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
* **Generative AI:** LangChain, Groq API (Llama 3.1-8b-instant)
* **Data Pipelines:** yfinance
* **Deployment:** Streamlit Community Cloud

---

## 🚀 Local Installation

Want to run RiskSight locally? Follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/RiskSight.git](https://github.com/YOUR_USERNAME/RiskSight.git)
cd RiskSight
