# ✅ Unified Teacher-Facing App – Streamlit Prototype with File Uploads

import streamlit as st
from pathlib import Path
from datetime import datetime
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 1. App Config
# -----------------------------
st.set_page_config(page_title="Curriculum CoDesigner", layout="centered")
st.title("🧠 Curriculum CoDesigner – AI Thinking Partner")

# -----------------------------
# 2. File Upload Section
# -----------------------------
with st.expander("📁 Upload Custom Documents (optional)"):
    uploaded_files = st.file_uploader("Upload PDFs for inspiration (e.g., existing lessons)", type="pdf", accept_multiple_files=True)
    temp_dir = Path("data/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    custom_docs = []
    if uploaded_files:
        for file in uploaded_files:
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)
            custom_docs.extend(split_docs)
        st.success(f"✅ Loaded and split {len(custom_docs)} chunks from uploaded PDFs.")

# -----------------------------
# 3. Input from Teacher
# -----------------------------
with st.form("unit_form"):
    topic = st.text_input("Unit Topic (e.g., Climate Change)", "ecosystems and human impact")
    grade = st.selectbox("Grade Level", ["6th", "7th", "8th", "middle school"])
    context = st.text_area("Describe your students or community context", "Black and Latinx students in Los Angeles")
    submitted = st.form_submit_button("Generate Unit Outline")

# -----------------------------
# 4. Setup RAG Components
# -----------------------------
if submitted:
    st.info("Loading models and retriever...")

    prompt_path = Path("prompts") / "unit_outline_prompt.txt"
    unit_prompt = load_prompt_from_file(prompt_path)

    vectorstore_path = "data/embeddings/faiss_index"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings)

    if custom_docs:
        custom_vectorstore = FAISS.from_documents(custom_docs, embeddings)
        retriever = custom_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    else:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

    rag_chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["topic"]),
            "topic": lambda x: x["topic"],
            "student_context": lambda x: x["student_context"],
            "grade_level": lambda x: x.get("grade_level", "middle school")
        })
        | unit_prompt
        | llm
        | StrOutputParser()
    )

    data = {
        "topic": topic,
        "student_context": context,
        "grade_level": grade
    }

    st.success("Generating unit outline... Please wait ⏳")
    unit_output = rag_chain.invoke(data)

    # Display
    st.subheader("📘 Draft Unit Outline")
    st.markdown(unit_output)

    # Save
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"unit_outline_{topic.replace(' ', '_')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(unit_output)
    st.success(f"✅ Saved to: {filename.name}")