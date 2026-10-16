import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer
import streamlit as st
import textwrap

st.title("GenAI common project - 2: RAG based chat bot (Local Gemma)")
st.sidebar.title("About")

uploaded_files = st.file_uploader(
    "Upload PDF files", accept_multiple_files=True
)

if uploaded_files:
  overall_content = ""
  # Step 1: Ingestion (Retrieve and extract text from PDFs)
  for uploaded_file in uploaded_files:
    pdf_file = PdfReader(uploaded_file)

    text = ""
    for page in pdf_file.pages:
      extracted = page.extract_text()
      if extracted:
        text += extracted

    overall_content += text + "\n"

  # Step 2: Chunking (Split text into manageable pieces)
  chunks = textwrap.wrap(overall_content, width=400)

  if chunks:
    # Step 3: Embedding (Convert text chunks into vector representations)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks)

    # Step 4: Store embeddings (in Vector Database)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))


    # Step 5: Retrieval from Vector DB based on user query
    def retrieve_similar_chunks(user_query, top_k=3):
      query_embedding = embedding_model.encode([user_query])
      distances, indices = index.search(np.array(query_embedding), top_k)
      retrieved_chunks = [chunks[i] for i in indices[0]]
      return retrieved_chunks


    # Step 6: Generate response using retrieved chunks via local Ollama API
    def generate_response(query, context_chunks):
      context = "\n".join(context_chunks)
      prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

      url = "http://localhost:11434/api/generate"
      payload = {"model": "llama3.2", "prompt": prompt}

      response_text = ""
      with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
          if line:
            chunk = json.loads(line)
            response_text += chunk.get("response", "")
            if chunk.get("done"):
              break
      return response_text


    user_query = st.chat_input("Enter your query here!!")

    if user_query:
      related_info_chunks = retrieve_similar_chunks(user_query)
      answer = generate_response(user_query, related_info_chunks)
      st.markdown(f"Bot: {answer}")