import os
import json
import numpy as np
import faiss
from typing import List, Dict
from config import Config
from text2vec import get_vector

try:
    from llama_index.core.node_parser import SentenceSplitter
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    print("Warning: llama-index not found. Using simple chunking.")

class KnowledgeIndexer:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.dimension = 1536 # text-embedding-3-small dimension
        
    def load_file(self, file_path: str) -> str:
        """读取文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """将文本切分为块"""
        if HAS_LLAMA_INDEX:
            splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
            return splitter.split_text(text)
        else:
            # Simple fallback chunking
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks

    def build_index(self, chunks: List[str], save_dir: str = "kb_index"):
        """构建索引并保存"""
        if not chunks:
            print("No chunks to index.")
            return

        print(f"Vectorizing {len(chunks)} chunks...")
        # 批量获取向量
        # text2vec.get_vector returns (N, 1536) for list input
        # We process in batches to be safe, though text2vec already handles batches
        vectors = get_vector(chunks)
        
        # 确保 vectors 是 float32 类型的 numpy 数组，这是 FAISS 要求的
        vectors = vectors.astype(np.float32)
        
        # 创建索引
        # 使用简单的 IndexFlatIP (Inner Product) 
        # 注意：如果向量未归一化，IP = Dot Product。OpenAI 向量通常是归一化的，所以 IP = Cosine Similarity
        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors)
        
        # 准备元数据
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "id": i,
                "chunk": chunk,
                "source": "dummy_medical.txt"
            })

        # 保存
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        index_path = os.path.join(save_dir, "medical.index")
        metadata_path = os.path.join(save_dir, "medical.json")
        
        faiss.write_index(index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Index built successfully!")
        print(f"Saved index to: {index_path}")
        print(f"Saved metadata to: {metadata_path}")
        print(f"Total vectors: {index.ntotal}")

if __name__ == "__main__":
    # 测试代码
    indexer = KnowledgeIndexer()
    
    # 1. Load
    print("Loading data...")
    try:
        content = indexer.load_file("dummy_medical.txt")
        
        # 2. Chunk
        print("Chunking text...")
        chunks = indexer.chunk_text(content)
        print(f"Generated {len(chunks)} chunks.")
        
        # 3. Index & Save
        print("Building index...")
        indexer.build_index(chunks)
        
    except FileNotFoundError:
        print("Please create 'dummy_medical.txt' first.")
    except Exception as e:
        print(f"An error occurred: {e}")
