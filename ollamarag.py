import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_folder):
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            with open(os.path.join(pdf_folder, file), "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                documents.append(text)
    return documents

# Function to create FAISS index
def create_faiss_index(texts):
    vectors = embed_model.encode(texts)
    d = vectors.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.array(vectors, dtype=np.float32))
    return index, texts

# Function to retrieve relevant text
def retrieve_relevant_text(query, index, texts, k=2):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return "\n".join([texts[i] for i in indices[0]])

# Function to generate response with Ollama
def generate_response(prompt, context):
    full_prompt = f"Context: {context}\n\nUser query: {prompt}\n\nAnswer:"
    response = chat(model='deepseek-r1:8b', messages=[{"role": "user", "content": full_prompt}])
    return response['message']['content']

# Main execution
# pdf_folder = r"C:\Users\Eric\Documents\Projects"
pdf_folder = r"C:\Users\Eric\Downloads\mancom_minutes_01"
documents = extract_text_from_pdfs(pdf_folder)
faiss_index, stored_texts = create_faiss_index(documents)

# Example query
query = "What are the deliverables of Mr. Emberda?"
context = retrieve_relevant_text(query, faiss_index, stored_texts)
response = generate_response(query, context)

print(response)
