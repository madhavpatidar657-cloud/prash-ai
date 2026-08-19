import os
import glob
import streamlit as st
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PRASH-AI", page_icon="🎓")
st.title("🎓 PRASH-AI — Study Assistant")

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_notes():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        return [], None, None
    reader = PdfReader(pdf_files[0])
    text = "".join([page.extract_text() or "" for page in reader.pages])
    
    chunk_size, overlap = 600, 100
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, chunk_vectors

chunks, vectorizer, chunk_vectors = load_notes()

tab1, tab2 = st.tabs(["💬 Ask Notes", "📝 Study Tools"])

with tab1:
    user_query = st.text_input("Ask any question from your notes:")
    if st.button("Ask PRASH"):
        if user_query and vectorizer is not None:
            query_vec = vectorizer.transform([user_query])
            scores = cosine_similarity(query_vec, chunk_vectors)[0]
            top_indices = scores.argsort()[-3:][::-1]
            context = "\n\n".join([chunks[i] for i in top_indices])
            
            prompt = f"""You are PRASH-AI, a supportive study assistant. Use ONLY this context to answer clearly:\n{context}\n\nQuestion: {user_query}"""
            res = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            st.write(res.text)

with tab2:
    mode = st.selectbox("Choose Study Tool", ["quiz", "flashcards", "summary"])
    if st.button("Generate Deck"):
        if chunks:
            full_doc = "\n".join(chunks)
            prompts = {
                "quiz": f"Create a 5-question MCQ quiz with answers at the end based on:\n{full_doc}",
                "flashcards": f"Create 5 flashcards from:\n{full_doc}",
                "summary": f"Create an exam revision cheat sheet from:\n{full_doc}"
            }
            res = client.models.generate_content(model="gemini-3.6-flash", contents=prompts[mode])
            st.write(res.text)
