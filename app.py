import gradio as gr
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.retriever import Retriever
from reproduction.generator import Generator

# Initialize system
print("Initializing RAG System...")
try:
    # Use absolute path for index_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_dir = os.path.join(current_dir, "kb_index")
    
    retriever = Retriever(index_dir=index_dir)
    generator = Generator()
    print("System initialized successfully.")
except Exception as e:
    print(f"Initialization failed: {e}")
    retriever = None
    generator = None

def chat_response(message, history):
    if not retriever or not generator:
        return "System not initialized correctly. Please check logs."
    
    # 1. Retrieve
    try:
        results = retriever.search(message, top_k=3)
        contexts = [res['text'] for res in results]
        
        # Format references for display
        refs = "\n\n**参考信息:**\n"
        for i, res in enumerate(results):
            refs += f"{i+1}. {res['text'][:100]}... (Score: {res['score']:.4f})\n"
            
    except Exception as e:
        print(f"Retrieval error: {e}")
        return f"Retrieval failed: {e}"

    # 2. Generate
    try:
        answer = generator.generate(message, contexts)
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Generation failed: {e}"
        
    return answer + refs

# Define Gradio Interface
with gr.Blocks(title="Medical RAG System") as demo:
    gr.Markdown("# 🏥 Medical Knowledge RAG System")
    gr.Markdown("这是一个基于 RAG (Retrieval-Augmented Generation) 的医疗问答演示系统。")
    
    chat_interface = gr.ChatInterface(
        fn=chat_response,
        # type="messages", # Removed as it causes TypeError in this version
        examples=["糖尿病的症状有哪些？", "如何预防二型糖尿病？", "妊娠期糖尿病对胎儿有什么影响？"],
        title="医疗助手",
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
