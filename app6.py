# ✅ Updated Unified Teacher-Facing App – Streamlit Prototype with Review + Export Page

import streamlit as st
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# CRP Pedagogical Resources - shown in sidebar
CRP_RESOURCES = """
### 🌟 What is Culturally Responsive Pedagogy (CRP)?
Culturally Responsive Pedagogy emphasizes using students' cultural backgrounds, experiences, and perspectives as valuable resources for teaching and learning.

- **Validate Students' Identities:** Affirm and celebrate diverse cultural identities.
- **Multiple Ways of Knowing:** Encourage students to bring their cultural, linguistic, and experiential knowledge into the classroom.
- **Social Relevance:** Connect learning to issues that are significant within students' communities.

[Learn more about CRP here](https://www.tolerance.org/professional-development/culturally-responsive-teaching)
"""

from pathlib import Path
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fpdf import FPDF

# Streamlit general config
st.set_page_config(page_title="Curriculum CoDesigner", layout="centered")
st.title("🧠 Curriculum CoDesigner – AI Thinking Partner")

# --------------------------------------------------
# State & Session Initialization
# --------------------------------------------------
default_session_state = {
    "unit_output": "",
    "expanded_lessons": "",
    "reflection_text": "",
    "topic": "",
    "grade": "middle school",
    "context": "",
    "submit_inputs": False,
    "custom_docs": []
}
for key, default in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "unit_output" not in st.session_state:
    st.session_state.unit_output = None  # or a default value like {} or ""

if "expanded_lessons" not in st.session_state:
    st.session_state.expanded_lessons = None


# Utility: load prompt from file
def load_prompt_from_file(prompt_path):
    with open(prompt_path, encoding="utf-8") as f:
        return PromptTemplate.from_template(f.read())

# --------------------------------------------------
# State & Session
# --------------------------------------------------
if "unit_output" not in st.session_state:
    st.session_state.unit_output = ""
if "expanded_lessons" not in st.session_state:
    st.session_state.expanded_lessons = ""
if "reflection_text" not in st.session_state:
    st.session_state.reflection_text = ""

# --------------------------------------------------
# Sidebar & Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
st.sidebar.markdown(CRP_RESOURCES)
page = st.sidebar.radio(
    "Go to:",
    ["1️⃣ Upload & Inputs", "2️⃣ Unit Builder", "3️⃣ Lesson Expansion", "4️⃣ Reflection", "5️⃣ Review & Export"]
)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# Page 1: Upload & Inputs
# --------------------------------------------------
if page == "1️⃣ Upload & Inputs":
    with st.expander("📁 Upload Custom Documents (optional)"):
        uploaded_files = st.file_uploader(
            "Upload PDFs for inspiration",
            type="pdf",
            accept_multiple_files=True
        )
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # We'll store custom docs in session
        st.session_state.custom_docs = []
        if uploaded_files:
            for file in uploaded_files:
                file_path = temp_dir / file.name
                # Save file to local
                with open(file_path, "wb") as f:
                    f.write(file.read())

                # Load & split
                loader = PyMuPDFLoader(str(file_path))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = splitter.split_documents(docs)

                st.session_state.custom_docs.extend(split_docs)

            st.success(f"✅ Loaded {len(st.session_state.custom_docs)} chunks from your PDFs.")

    with st.form("unit_form"):
        # Basic user inputs
        st.session_state.topic = st.text_input("Unit Topic", "ecosystems and human impact")
        st.session_state.grade = st.selectbox("Grade Level", ["6th", "7th", "8th", "middle school"])
        st.session_state.context = st.text_area(
            "Describe your student/community context",
            "Black and Latinx students in LA"
        )
        st.session_state.submit_inputs = st.form_submit_button("Generate Unit Outline")

# --------------------------------------------------
# Page 2: Unit Builder
# --------------------------------------------------
if page == "2️⃣ Unit Builder" and st.session_state.get("submit_inputs"):
    st.info("🔄 Generating outline using your topic and context...")



    # 1) Load the prompt
    unit_prompt = load_prompt_from_file(Path("prompts") / "unit_outline_prompt.txt")

    # 2) Load existing vector store (dangerous deserialization if local)
    vectorstore_path = "data/embeddings/faiss_index"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    # 3) If user uploaded custom docs, build a mini store for them
    if st.session_state.get("custom_docs"):
        custom_vectorstore = FAISS.from_documents(st.session_state.custom_docs, embeddings)
        retriever = custom_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    else:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})

    # 4) LLM and chain
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
        "topic": st.session_state.topic,
        "student_context": st.session_state.context,
        "grade_level": st.session_state.grade
    }
    # 5) Invoke chain
    st.session_state.unit_output = rag_chain.invoke(data)

    # 6) Display
    st.subheader("📘 Generated Unit Outline")
    st.markdown(st.session_state.unit_output)

# --------------------------------------------------
# Page 3: Lesson Expansion
# --------------------------------------------------
if page == "3️⃣ Lesson Expansion" and st.session_state.unit_output:
    st.subheader("📚 Expand Into Lessons")

    with st.form("lesson_form"):
        num_lessons = st.slider("Number of lessons", 2, 10, 4)
        expand_button = st.form_submit_button("Expand Lessons")

    if expand_button:
        lesson_prompt = load_prompt_from_file(Path("prompts") / "lesson_expander_prompt.txt")
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

        lesson_chain = (
            RunnableMap({
                "unit_outline": lambda x: st.session_state.unit_output,
                "topic": lambda x: st.session_state.topic,
                "num_lessons": lambda x: x["num_lessons"]
            })
            | lesson_prompt
            | llm
            | StrOutputParser()
        )
        lesson_data = {"num_lessons": num_lessons}
        st.session_state.expanded_lessons = lesson_chain.invoke(lesson_data)
        st.markdown(st.session_state.expanded_lessons)

# --------------------------------------------------
# Page 4: Reflection
# --------------------------------------------------
if page == "4️⃣ Reflection" and st.session_state.expanded_lessons:
    st.subheader("🪞 Teacher Reflection")
    st.info("Tip: Think about how your unit design incorporates aspects of CRP. See sidebar for guidance.")

    with st.form("reflection_form"):
        q1 = st.text_area("1. How does this unit reflect the cultural identities of your students?")
        q2 = st.text_area("2. Where can students bring in multiple ways of knowing?")
        q3 = st.text_area("3. What could make this more locally meaningful?")
        q4 = st.text_area("4. What are open questions you still have about equity in this unit?")
        submit_reflection = st.form_submit_button("💾 Save Reflection")

    if submit_reflection:
        st.session_state.reflection_text = f"""
**Reflection for Unit: {st.session_state.topic}**

**1. Cultural Identities:**
{q1}

**2. Multiple Ways of Knowing:**
{q2}

**3. Local Relevance:**
{q3}

**4. Open Questions:**
{q4}
"""
        st.success("✅ Reflection saved. Proceed to Review & Export tab.")

# --------------------------------------------------
# Page 5: Review & Export
# --------------------------------------------------
if page == "5️⃣ Review & Export" and st.session_state.unit_output and st.session_state.expanded_lessons:
    st.subheader("📦 Review Final Bundle")
    st.markdown("🔍 **Reflect**: Have you addressed the principles of Culturally Responsive Pedagogy in your curriculum?")

    # Display content
    st.markdown("### 🧾 Unit Plan")
    st.markdown(st.session_state.unit_output)

    st.markdown("### 🧩 Lessons")
    st.markdown(st.session_state.expanded_lessons)

    st.markdown("### 🪞 Reflection")
    st.markdown(st.session_state.reflection_text)

    # Combine all
    full_text = f"""
{st.session_state.unit_output}

{st.session_state.expanded_lessons}

{st.session_state.reflection_text}
"""

    # Export to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in full_text.split("\n"):
        pdf.multi_cell(0, 10, line)

    # Save
    from pathlib import Path
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"unit_bundle_{st.session_state.topic.replace(' ', '_')}.pdf"
    pdf.output(str(filename))

    # Download
    with open(filename, "rb") as f:
        st.download_button("📄 Download Full Unit PDF", data=f, file_name=filename.name)
    st.success("✅ Your full curriculum design has been bundled!")
