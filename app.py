
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

skills_database = [
    "python",
    "machine learning",
    "deep learning",
    "pandas",
    "numpy",
    "scikit-learn",
    "sql",
    "tensorflow"
]
st.title("🤖 AI Resume Screening System")
st.write("Analyze resume-job fit using NLP and Machine Learning")

resume = st.text_area("Paste Resume")

job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume, job_description]
    )

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )

    st.success(
        f"Match Score: {round(score[0][0]*100,2)}%"
    )
    st.progress(float(score[0][0]))
    if score[0][0] >= 0.8:
       Sst.success("🚀 Excellent Match")

    elif score[0][0] >= 0.6:
         st.warning("👍 Good Match")

    else:
         st.error("📚 Needs Improvement")

    resume_lower = resume.lower()
    jd_lower = job_description.lower()

    resume_skills = [
        skill for skill in skills_database
        if skill in resume_lower
    ]

    jd_skills = [
        skill for skill in skills_database
        if skill in jd_lower
    ]

    missing_skills = list(
        set(jd_skills) - set(resume_skills)
    )

    st.subheader("Missing Skills")

    if missing_skills:
        for skill in missing_skills:
            st.error(skill)
    else:
        st.success("No Missing Skills 🎉")