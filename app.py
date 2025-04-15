# ✅ Updated Unified Teacher-Facing App – Streamlit Prototype

import streamlit as st
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# Assume a utility function for loading prompts
def load_prompt_from_file(prompt_path):
    with open(prompt_path, encoding="utf-8") as f:
        return PromptTemplate.from_template(f.read())

# -----------------------------
# 1. App Config
# -----------------------------
st.set_page_config(page_title="Curriculum CoDesigner", layout="centered")
st.title("🧠 Curriculum CoDesigner – AI Thinking Partner")

# -----------------------------
# 2. Input from Teacher
# -----------------------------
with st.form("unit_form"):
    topic = st.text_input("Unit Topic (e.g., Climate Change)", "ecosystems and human impact")
    grade = st.selectbox("Grade Level", ["6th", "7th", "8th", "middle school"])
    context = st.text_area("Describe your students or community context", "Black and Latinx students in Los Angeles")
    submitted = st.form_submit_button("Generate Unit Outline")

# -----------------------------
# 3. Setup RAG Components
# -----------------------------
if submitted:
    st.info("Loading models and retriever...")

    prompt_path = Path("prompts") / "unit_outline_prompt.txt"
    unit_prompt = load_prompt_from_file(prompt_path)

    vectorstore_path = "data/embeddings/faiss_index"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    llm = ChatOpenAI(model="gpt-4", temperature=0.3)

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
