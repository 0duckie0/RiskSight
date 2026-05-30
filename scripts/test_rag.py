import os
import sys
import ssl

# --- 1. THE SSL BYPASS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from dotenv import load_dotenv

# 2. Setup absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_path = os.path.join(BASE_DIR, "data", "sample_report.pdf")
env_path = os.path.join(BASE_DIR, ".env")

# 3. Load the hidden API Key
load_dotenv(env_path)
groq_key = os.environ.get("GROQ_API_KEY")
if not groq_key:
    print(f"❌ ERROR: Groq API key not found. Please check your .env file at {env_path}")
    sys.exit()

print("Initializing RiskSight RAG Engine (Groq Speed Demon Edition)...")

# --- 4. IMPORTS ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.retrievers import BM25Retriever
    from langchain_groq import ChatGroq # 👈 The new Groq Engine
except Exception as e:
    print(f"❌ Library Import Error: {e}")
    sys.exit()

# 5. Configure Groq LLM (Using Meta's Llama 3.1)
llm = ChatGroq(
    model="llama-3.1-8b-instant", # 👈 The brand new, active model!
    temperature=0,
    api_key=groq_key
)

try:
    # 6. Load and Chunk the PDF
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"✅ PDF chopped into {len(splits)} searchable chunks.")
    
    # 7. Build the BM25 Offline Search Engine
    print("Building Local BM25 Search Engine (Instant & Offline)...")
    retriever = BM25Retriever.from_documents(splits)
    retriever.k = 4 # Get top 4 best matching chunks
    
    # 8. Clean Manual RAG Execution
    print("\n=========================================")
    test_question = "What are the major risk factors or contingent liabilities mentioned in this report?"
    print(f"🗣️ Asking: {test_question}")
    
    # Search locally
    relevant_docs = retriever.invoke(test_question)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Construct the final prompt
    system_prompt = (
        "You are RiskSight, an expert financial auditor.\n"
        "Use the following retrieved context from an annual report to answer the question.\n"
        "If you don't know the answer, say that you cannot find it in the provided document.\n\n"
        f"Context:\n{context_text}"
    )
    
    # 9. Send to Groq (Lightning Fast)
    print("🤖 RiskSight analyzing findings via Groq & Llama 3...")
    response = llm.invoke([
        ("system", system_prompt),
        ("human", test_question)
    ])
    
    print("\n🤖 RiskSight RAG Answer:")
    print(response.content)
    print("=========================================\n")

except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find '{pdf_path}'")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")