import json
import faiss
import numpy as np
import os
import sys

# Ensure the parent directory is in sys.path to import config if needed, 
# but here we rely on local imports or relative paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text2vec import get_vector

class Retriever:
    def __init__(self, index_dir: str = "kb_index"):
        self.index_path = os.path.join(index_dir, "medical.index")
        self.meta_path = os.path.join(index_dir, "medical.json")
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Index or metadata not found in {index_dir}")
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        print(f"[Retriever] Loaded index with {self.index.ntotal} vectors.")
        
        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[Retriever] Loaded metadata for {len(self.metadata)} chunks.")

    def search(self, query: str, top_k: int = 2):
        print(f"\n[Search] Query: {query}")
        
        # 1. Convert query to vector
        # get_vector returns a list of vectors, we take the first one for the single query
        query_vec = get_vector([query])[0].astype(np.float32)
        
        # 2. Reshape for FAISS (1, dimension)
        query_vec = query_vec.reshape(1, -1)
        
        # 3. Search
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue # No match found
            
            chunk_text = self.metadata[idx]["chunk"]
            score = distances[0][i]
            
            result = {
                "rank": i + 1,
                "score": float(score),
                "text": chunk_text
            }
            results.append(result)
            
        return results

if __name__ == "__main__":
    # Test the retriever
    try:
        retriever = Retriever()
        
        # Test query relevant to the dummy medical text
        query = "糖尿病的症状有哪些？"
        results = retriever.search(query)
        
        print(f"\nTop {len(results)} Results:")
        for res in results:
            print(f"Rank {res['rank']} (Score: {res['score']:.4f}):")
            print(f"Content: {res['text']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
