import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI HR Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.sidebar.title("AI Resume Screener")
st.sidebar.write("LangChain + RAG + ChromaDB + Gemini")
st.sidebar.write("Use Case: HRMS / ATS Resume Screening")

st.title("📄 AI HR Resume Screening System")
st.write("Upload a resume and compare it with a job description using RAG.")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=250)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def build_resume_vector_db(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
    

    return vector_db

def analyze_resume_with_rag(vector_db, jd_text):
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    query = f"""
Find the most relevant resume details for this job description:
{jd_text}
"""

    relevant_docs = retriever.invoke(query)

    resume_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are an experienced HR recruiter and ATS screening expert.

Use only the resume context given below and compare it with the job description.

Give output in this format:

1. ATS Match Score: __%

2. Candidate Summary:
Write 4-5 lines.

3. Matching Skills:
List skills from resume that match the job description.

4. Missing Skills:
List important skills missing from resume.

5. Resume Improvement Suggestions:
Give practical suggestions.

6. Technical Interview Questions:
Generate 7 technical questions.

7. HR Interview Questions:
Generate 5 HR questions.

Resume Context:
{resume_context}

Job Description:
{jd_text}
"""

    response = llm.invoke(prompt)
    return response.content

if st.button("Analyze Resume"):
    if resume_file is None:
        st.warning("Please upload a resume PDF.")
    elif job_description.strip() == "":
        st.warning("Please paste the job description.")
    else:
        with st.spinner("Creating resume vector database..."):
            vector_db = build_resume_vector_db(resume_file)

        with st.spinner("Analyzing resume using RAG..."):
            result = analyze_resume_with_rag(vector_db, job_description[:2000])

        st.success("Analysis Completed!")

        st.subheader("AI Analysis Result")
        st.write(result)

        st.download_button(
            label="Download Analysis Report",
            data=result,
            file_name="resume_analysis_report.txt",
            mime="text/plain"
        )

st.divider()

st.subheader("Project Architecture")
st.write("""
Resume PDF → PyPDFLoader → Text Splitting → Gemini Embeddings → ChromaDB → Retriever → Gemini LLM → ATS Report
""")