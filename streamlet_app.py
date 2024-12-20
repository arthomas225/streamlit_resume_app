import streamlit as st
import os
import zipfile
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

##############################
# Helper Functions
##############################

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pass
    return text

def read_docx_file(file_path):
    try:
        document = docx.Document(file_path)
        return "\n".join([para.text for para in document.paragraphs])
    except:
        return ""

def read_resume(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".txt":
        return read_txt(file_path)
    elif extension == ".pdf":
        return read_pdf(file_path)
    elif extension == ".docx":
        return read_docx_file(file_path)
    else:
        return ""

def extract_snippet(content, query):
    query_lower = query.lower()
    index = content.lower().find(query_lower)
    if index == -1:
        # If query isn't found directly (it should be, but just in case)
        # return the first 200 chars as snippet
        snippet = content[:200].replace('\n', ' ')
    else:
        start = max(0, index - 50)
        end = min(len(content), index + len(query) + 50)
        snippet = content[start:end].replace('\n', ' ')
    return snippet.strip()

def preprocess_text_for_keywords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return set(filtered_tokens)

def highlight_query(snippet, query):
    # Simple highlight: replace exact query matches with bold
    return snippet.replace(query, f"**{query}**")

##############################
# Streamlit App
##############################

st.title("Resume Search with Semantic Embeddings and Keywords")

st.write("Upload a ZIP of resumes, enter a query, and select a similarity threshold. Only resumes above the threshold are shown.")

uploaded_zip = st.file_uploader("Upload a ZIP file of resumes (PDF, DOCX, TXT):", type=['zip'])

resumes = {}
if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith(('.pdf','.docx','.txt')):
                with open(name, 'wb') as f:
                    f.write(z.read(name))
                content = read_resume(name)
                if content.strip():
                    resumes[name] = content
    st.success(f"Loaded {len(resumes)} resumes from the ZIP file.")

query = st.text_input("Enter your search query:")
score_threshold = st.slider("Semantic Similarity Threshold", 0.0, 1.0, 0.40, 0.05)

if st.button("Search"):
    if not resumes:
        st.warning("No resumes loaded. Please upload a ZIP file first.")
    else:
        # Load model once
        @st.cache_resource
        def load_model():
            return SentenceTransformer('all-MiniLM-L6-v2')
        model = load_model()

        query_embedding = model.encode(query)
        resume_embeddings = {}
        for path, content in resumes.items():
            resume_embeddings[path] = model.encode(content)

        results = []
        # We use threshold strictly: no resumes below threshold
        for path, content in resumes.items():
            embedding = resume_embeddings[path]
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]

            if similarity >= score_threshold:
                snippet = extract_snippet(content, query)
                # Highlight the query in snippet
                highlighted_snippet = highlight_query(snippet, query)
                results.append((os.path.basename(path), similarity, highlighted_snippet))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        if results:
            st.write(f"Found {len(results)} matching resume(s):")
            for filename, sim, snippet in results:
                st.write(f"**{filename}** - Similarity: {sim:.2f}")
                st.write(f"> {snippet}")
        else:
            st.write("No resumes matched the query with the given threshold.")
