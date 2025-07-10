import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load cleaned data
df = pd.read_csv('./data/filtered_complaints.csv')

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

# Track chunk metadata
chunk_texts = []
chunk_metadata = []

for _, row in df.iterrows():
    text = row['cleaned_narrative']
    
    # Skip if text is missing or not a string
    if not isinstance(text, str) or not text.strip():
        continue

    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        chunk_texts.append(chunk)
        chunk_metadata.append({
            "complaint_id": row['Complaint ID'],
            "product": row['Product']
        })

print(f"Total chunks created: {len(chunk_texts)}")


from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(chunk_texts, show_progress_bar=True)

# Create FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.idx")

with open("vector_store/chunk_metadata.pkl", "wb") as f:
    pickle.dump(chunk_metadata, f)

print("✅ Vector store saved in vector_store/")


import numpy as np

def retrieve_relevant_chunks(question: str, k: int = 5):
    # Embed the question
    question_embedding = model.encode([question])
    
    # Search for top-k similar chunks
    D, I = index.search(np.array(question_embedding), k)
    
    # Fetch results and metadata
    retrieved_chunks = []
    for idx in I[0]:
        retrieved_chunks.append({
            "text": chunk_texts[idx],
            "metadata": metadata[idx]
        })
    
    return retrieved_chunks


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.

If the context doesn't contain the answer, say:
"I don't have enough information to answer that based on the provided complaints."

Context:
{context}

Question:
{question}

Answer:"""

    return prompt.strip()



sample_questions = [
    "How do customers describe issues with credit card fraud?",
    "Are there frequent complaints about loan approval delays?",
    "What are common problems in money transfers?",
    "Do customers mention problems with savings account access?",
    "How satisfied are users with BNPL services?",
]


evaluation_results = []

for question in sample_questions:
    results = retrieve_relevant_chunks(question, k=5)
    context = [r['text'] for r in results]
    prompt = build_prompt(question, context)
    answer = generate_answer(prompt)

    # Add first 1–2 chunks for reference
    retrieved_sources = "\n---\n".join(context[:2])

    evaluation_results.append({
        "question": question,
        "generated_answer": answer,
        "retrieved_sources": retrieved_sources,
        "quality_score": "",  # fill in manually: 1 (poor) to 5 (excellent)
        "comments": ""         # write a quick analysis
    })


