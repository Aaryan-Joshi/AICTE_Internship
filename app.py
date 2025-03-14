import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Extract text from each resume
    resume_texts = [extract_text_from_pdf(resume) for resume in resumes]

    # Include job description in the list for comparison
    texts = [job_description] + resume_texts

    # Convert text data into TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity between job description and resumes
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])

    # Rank resumes based on similarity scores
    ranked_resumes = sorted(
        enumerate(resumes), key=lambda x: similarity_scores[0][x[0]], reverse=True
    )

    return ranked_resumes

# Streamlit App
st.title("AI Resume Screening & Candidate Ranking System")
st.write("Upload resumes and provide a job description to rank candidates.")

# Upload job description
job_description = st.text_area("Enter Job Description")

# Upload resumes (multiple files)
uploaded_files = st.file_uploader("Upload Resumes (PDF format)", accept_multiple_files=True, type=["pdf"])

if st.button("Rank Resumes"):
    if job_description and uploaded_files:
        ranked_resumes = rank_resumes(job_description, uploaded_files)

        # Display results
        st.subheader("Ranked Resumes:")
        for rank, (index, resume) in enumerate(ranked_resumes, start=1):
            st.write(f"{rank}. {resume.name}")
    else:
        st.warning("Please enter a job description and upload resumes.")
