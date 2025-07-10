import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and metadata
index = faiss.read_index("vector_store/faiss_index.idx")

with open("vector_store/chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load same embedding model used in Task 2
model = SentenceTransformer('all-MiniLM-L6-v2')
