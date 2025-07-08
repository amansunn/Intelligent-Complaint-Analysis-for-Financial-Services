import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load cleaned data
df = pd.read_csv('../data/filtered_complaints.csv')

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
    chunks = text_splitter.split_text(row['cleaned_narrative'])
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
