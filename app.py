import os
import sys
import streamlit as st
from pathlib import Path
from cryptography.fernet import Fernet

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CORPUS_PATH     = "/kaggle/input/zyro-dynamics-hr-corpus/"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_PROVIDER    = "groq"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
RETRIEVER_K     = 6
MMR_FETCH_K     = 20
MMR_LAMBDA      = 0.7

IN_SCOPE_TOPICS = [
    "leave","vacation","sick","maternity","paternity","wfh","work from home",
    "remote","hybrid","salary","ctc","compensation","benefits","appraisal",
    "performance","pip","review","code of conduct","ethics","harassment","posh",
    "icc","onboarding","offboarding","separation","probation","travel","expense",
    "reimbursement","it policy","data security","device","employee","hr","policy",
    "handbook","zyro","notice period","resignation","termination","bonus","grade",
    "attendance","working hours","holiday","joining","full and final","f&f",
    "el","sl","cl","earned leave","sick leave","casual leave",
]

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. "
    "Your question appears to be outside the scope of HR policies. "
    "Please contact the relevant department for assistance."
)

# ── Build RAG pipeline (cached so it only runs once per session) ──────────────
@st.cache_resource(show_spinner="Loading HR knowledge base... please wait ⏳")
def build_pipeline():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Load documents
    loader    = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(
            doc.metadata.get("source", "")
        )

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    # Vector store + hybrid retriever
    vectorstore        = FAISS.from_documents(chunks, embeddings)
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": MMR_FETCH_K,
                       "lambda_mult": MMR_LAMBDA},
    )
    bm25_retriever   = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVER_K
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )

    # LLM
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=LLM_MODEL, temperature=0.0, max_tokens=1024)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0,
                                     max_output_tokens=1024)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, max_tokens=1024)

    # Prompts
    rag_prompt = ChatPromptTemplate.from_template("""
You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the provided HR policy context below.
Be accurate, concise, and professional.

Rules:
- Base your answer strictly on the context provided.
- If the context does not contain enough information, say so clearly.
- Do not make up policies, numbers, or dates not mentioned in the context.
- Cite the source document name when relevant.
- Keep answers focused and structured when listing multiple points.

Context:
{context}

Employee Question: {question}

Answer:""")

    oos_prompt = ChatPromptTemplate.from_template("""
You are a classifier for an HR chatbot at Zyro Dynamics.
Determine if the question is related to HR policies or employee matters.
Reply with exactly one word — YES if HR-related, NO if not.
Question: {question}""")

    return hybrid_retriever, llm, rag_prompt, oos_prompt


def format_docs(docs):
    parts = []
    for doc in docs:
        src    = doc.metadata.get("source_file", "HR Policy")
        page   = doc.metadata.get("page", "")
        header = f"[{src}" + (f", p.{int(page)+1}]" if page != "" else "]")
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def keyword_in_scope(question):
    q = question.lower()
    return any(t in q for t in IN_SCOPE_TOPICS)


def ask(question, retriever, llm, rag_prompt, oos_prompt):
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()

    in_scope = keyword_in_scope(question)
    if not in_scope:
        try:
            verdict = (oos_prompt | llm | parser).invoke(
                {"question": question}
            ).strip().upper()
            in_scope = verdict.startswith("YES")
        except Exception:
            in_scope = True

    if not in_scope:
        return REFUSAL_MESSAGE, []

    docs    = retriever.invoke(question)
    context = format_docs(docs)
    answer  = (rag_prompt | llm | parser).invoke(
        {"context": context, "question": question}
    )
    sources = list(dict.fromkeys(
        d.metadata.get("source_file", "HR Policy") for d in docs
    ))
    return answer.strip(), sources


# ── API key setup ─────────────────────────────────────────────────────────────
def load_api_keys():
    try:
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        if LLM_PROVIDER == "groq":
            os.environ["GROQ_API_KEY"] = s.get_secret("GROQ_API_KEY")
        elif LLM_PROVIDER == "gemini":
            os.environ["GOOGLE_API_KEY"] = s.get_secret("GOOGLE_API_KEY")
        else:
            os.environ["OPENAI_API_KEY"] = s.get_secret("OPENAI_API_KEY")
    except Exception:
        pass  # Keys already in env (local dev)

load_api_keys()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about HR policies, leave, compensation, WFH, and more.")

with st.sidebar:
    st.header("📋 About")
    st.markdown(
        "This AI assistant answers your HR questions based on **Zyro Dynamics' "
        "official policy documents**.\n\n"
        "**Topics covered:**\n"
        "- Leave & Attendance\n"
        "- Work From Home\n"
        "- Salary & Benefits\n"
        "- Performance Reviews\n"
        "- Code of Conduct\n"
        "- Travel & Expenses\n"
        "- IT & Data Security\n"
        "- POSH Policy\n"
        "- Onboarding & Offboarding"
    )
    st.divider()
    st.caption("Powered by Zyro Dynamics RAG Pipeline")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Load pipeline
retriever, llm, rag_prompt, oos_prompt = build_pipeline()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm the Zyro Dynamics HR Assistant. How can I help you today?",
        "sources": [],
    })

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# Input
if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources = ask(prompt, retriever, llm, rag_prompt, oos_prompt)
        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )