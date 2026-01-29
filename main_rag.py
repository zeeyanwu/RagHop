import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.retriever import Retriever
from reproduction.generator import Generator

def main():
    print("Initializing RAG System...")
    try:
        retriever = Retriever(index_dir="kb_index")
        generator = Generator()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\n=== RAG System Ready (Type 'exit' to quit) ===")
    
    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        if not query:
            continue
            
        print(f"\n[1] Retrieving knowledge...")
        try:
            results = retriever.search(query, top_k=3)
            
            contexts = []
            print("\n--- Retrieved Contexts ---")
            for res in results:
                print(f"[{res['score']:.4f}] {res['text']}")
                contexts.append(res['text'])
            print("--------------------------")
            
            if not contexts:
                print("No relevant context found.")
                continue
                
            print(f"\n[2] Generating answer...")
            answer = generator.generate(query, contexts)
            
            print(f"\n=== Answer ===\n{answer}\n==============")
            
        except Exception as e:
            print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
