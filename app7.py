# ✅ Streamlit App: Teacher Input + Unit Outline Generator with Evaluation Mode + CSV Logging (No Lesson Expansion)

import re
import json
import csv
from pathlib import Path
from datetime import datetime
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# Load API keys from .env file 
load_dotenv(override=True)

# -----------------------------
# Utility Functions
# -----------------------------
def load_prompt_from_file(prompt_path):
    with open(prompt_path, encoding="utf-8") as f:
        return PromptTemplate.from_template(f.read())

def slugify(text, max_length=60):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_length]

def extract_sections(markdown_text):
    sections = {
        "title": None,
        "phenomenon": None,
        "driving_question": None,
        "summary": None,
        "lesson_sets": {},
        "investigations": {},
        "ngss": [],
        "reflection_prompts": []
    }

    lines = markdown_text.splitlines()
    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("## Unit Title"):
            current_section = "title"
        elif line.startswith("### Anchoring Phenomenon"):
            current_section = "phenomenon"
        elif line.startswith("### Driving Question"):
            current_section = "driving_question"
        elif line.startswith("### Storyline Arc Summary") or line.startswith("### Introduction"):
            current_section = "summary"
        elif line.startswith("### Lesson Sets"):
            current_section = "lesson_sets"
        elif line.startswith("### Key Investigations"):
            current_section = "investigations"
        elif line.startswith("### NGSS Performance Expectations"):
            current_section = "ngss"
        elif line.startswith("### Suggested Teacher Reflection Prompts"):
            current_section = "reflection_prompts"
        elif current_section == "lesson_sets" and re.match(r"^\d+\.\s+\*\*Lesson", line):
            match = re.match(r"^(\d+)\.\s+\*\*(Lesson.*?)\*\*[:\s]*(.*)", line)
            if match:
                idx, title, desc = match.groups()
                sections["lesson_sets"][f"Lesson {idx}: {title}"] = desc.strip()
        elif current_section == "investigations" and re.match(r"-\s+Investigation", line):
            match = re.match(r"-\s+(Investigation \d+):\s*(.*)", line)
            if match:
                key, value = match.groups()
                sections["investigations"][key] = value.strip()
        elif current_section == "ngss" and line.startswith("- "):
            sections["ngss"].append(line[2:].strip())
        elif current_section == "reflection_prompts" and line.startswith("- "):
            sections["reflection_prompts"].append(line[2:].strip())
        elif current_section in ["title", "phenomenon", "driving_question", "summary"] and line:
            if not sections[current_section]:
                sections[current_section] = line
            else:
                sections[current_section] += f" {line}"

    return sections

# -----------------------------
# Load Prompt and Vectorstore
# -----------------------------
unit_prompt = load_prompt_from_file(Path("prompts") / "unit_outline_prompt.txt")
vectorstore_path = "data/embeddings/faiss_index"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 7})
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Curriculum CoDesigner", layout="centered")
st.title("🧠 Curriculum CoDesigner – Unit Planner")

with st.sidebar:
    st.markdown("""
    ### Introduction:
    The goal of this interview is to understand your thinking as you prepare to teach a lesson, and especially how you use ChatGPT as part of that planning. We will not be “grading” or otherwise assessing your lesson plan; our focus is on your planning process. We are hoping to both better understand how teachers actually think about planning (because we don’t think we have very good, or very updated, understandings about this) and how emerging tools like ChatGPT could potentially be useful, and how they might not be useful.    
    ### 🧾 How to Use
    1. Input a general topic or more specific learning objectives
    2. Choose a grade level
    3. Enter any additional context for your class or overall unit.
    4. Click to generate a unit outline.
    5. Copy what is generated immediately into a Google Doc and share that document with the researcher
    6. Annotate on the Google Doc (but you can go back to the unit generator at any time)

    """)
    eval_mode = st.checkbox("Enable Evaluation Mode", value=False)
    st.markdown("""
    Enable evaluation mode to view retrieved document sources.
    """)

# -----------------------------
# Input Form
# -----------------------------
st.subheader("1️⃣ Teacher Inputs")
with st.form("input_form"):
    topic = st.text_input("Unit Topic or Learning Objectives", "Climate Justice")
    grade_level = st.selectbox("Grade Level", ["6th", "7th", "8th", "9th", "10th", "11th", "12th"])
    student_context = st.text_area("Describe Your Subject, Student/Community, or additional context", "Biology class. Black and Latinx students living in Chicago")
    submitted = st.form_submit_button("✨ Generate Unit Outline")

if submitted:
    data = {
        "topic": topic,
        "grade_level": grade_level,
        "student_context": student_context
    }

    st.info("🔄 Generating outline using your topic and context...")
    rag_chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["topic"]),
            "topic": lambda x: x["topic"],
            "student_context": lambda x: x["student_context"],
            "grade_level": lambda x: x.get("grade_level")
        })
        | unit_prompt
        | llm
        | StrOutputParser()
    )

    outline_response = rag_chain.invoke(data)
    st.subheader("📘 Generated Unit Plan")
    st.markdown(outline_response)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_slug = slugify(f"{topic}_{grade_level}")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save markdown
    md_path = output_dir / f"unit_outline_{filename_slug}_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(outline_response)

    # Save JSON
    json_data = extract_sections(outline_response)
    json_data.update(data)
    json_path = output_dir / f"unit_outline_{filename_slug}_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    st.success(f"✅ Saved markdown: {md_path.name}")
    st.success(f"✅ Saved JSON: {json_path.name}")

    if eval_mode:
        st.subheader("🔍 Evaluation Mode – Retrieved Chunks")
        docs = retriever.get_relevant_documents(topic)
        log_path = output_dir / f"retrieval_log_{filename_slug}_{timestamp}.csv"
        with open(log_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Chunk #", "Source Folder", "Filename", "Preview"])
            for i, doc in enumerate(docs):
                source_folder = doc.metadata.get("source_folder", "unknown")
                filename = doc.metadata.get("filename", "unknown")
                preview = doc.page_content[:500].replace("\n", " ")
                st.markdown(f"**{i+1}. {filename}**\n\n{preview}")
                writer.writerow([i+1, source_folder, filename, preview])
        st.success(f"✅ Logged document traces: {log_path.name}")
