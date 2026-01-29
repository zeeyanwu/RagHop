import gradio as gr
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduction.retriever import Retriever
from reproduction.generator import Generator
from reproduction.indexer import KnowledgeBaseManager

# --- Global State ---
kb_manager = KnowledgeBaseManager()
current_retriever = None
current_generator = Generator()
current_kb = "kb_index" # Default KB name from previous steps (which was just the folder name)

# Check if default KB exists in the new structure
# The previous step created 'kb_index' folder directly in reproduction.
# The new manager expects KBs inside 'knowledge_bases' folder.
# We should probably migrate or just create a new default KB.
if not os.path.exists(kb_manager.get_kb_path("default")):
    kb_manager.create_kb("default")
    # If we have the old medical.index, maybe move it?
    # For simplicity, let's start fresh or user can upload files.

def load_retriever(kb_name):
    global current_retriever
    try:
        kb_path = kb_manager.get_kb_path(kb_name)
        # Check if index exists
        if os.path.exists(os.path.join(kb_path, "medical.index")):
            current_retriever = Retriever(index_dir=kb_path)
            return f"Successfully loaded KB: {kb_name}"
        else:
            current_retriever = None
            return f"KB '{kb_name}' selected but no index found. Please upload files."
    except Exception as e:
        current_retriever = None
        return f"Error loading KB '{kb_name}': {e}"

# Initial Load
load_retriever("default")

# --- Chat Functions ---

def chat_response(message, history, kb_name):
    # Reload retriever if KB changed (though we use dropdown change event usually)
    # But to be safe, we can check if current_retriever matches? 
    # For now, rely on global state or reload here if needed.
    global current_retriever
    
    if not current_retriever:
        msg = load_retriever(kb_name)
        if not current_retriever:
            return msg
    
    # 1. Retrieve
    try:
        results = current_retriever.search(message, top_k=3)
        contexts = [res['text'] for res in results]
        
        if not contexts:
            return "No relevant information found in the knowledge base."

        # Format references
        refs = "\n\n**参考信息:**\n"
        for i, res in enumerate(results):
            refs += f"{i+1}. {res['text'][:100]}... (Score: {res['score']:.4f})\n"
            
    except Exception as e:
        return f"Retrieval failed: {e}"

    # 2. Generate
    try:
        answer = current_generator.generate(message, contexts)
    except Exception as e:
        return f"Generation failed: {e}"
        
    return answer + refs

# --- Management Functions ---

def refresh_kb_list():
    kbs = kb_manager.list_kbs()
    return gr.Dropdown(choices=kbs), gr.Dropdown(choices=kbs)

def create_new_kb(name):
    msg = kb_manager.create_kb(name)
    return msg, *refresh_kb_list()

def upload_and_index(kb_name, files):
    if not files:
        return "No files selected."
    
    results = []
    for file in files:
        res = kb_manager.add_file_to_kb(kb_name, file)
        results.append(f"{os.path.basename(file.name)}: {res}")
    
    # Reload retriever if we updated the current KB
    if kb_name == current_kb:
        load_retriever(kb_name)
        
    return "\n".join(results)

# --- UI ---

with gr.Blocks(title="RagHop Reproduction") as demo:
    gr.Markdown("# 🏥 Medical RAG System (Reproduction)")
    
    with gr.Tabs():
        # Tab 1: Chat
        with gr.TabItem("💬 Chat"):
            with gr.Row():
                kb_selector = gr.Dropdown(
                    label="Select Knowledge Base",
                    choices=kb_manager.list_kbs(),
                    value="default",
                    interactive=True
                )
                status_box = gr.Textbox(label="System Status", value="Ready", interactive=False)
            
            chat_interface = gr.ChatInterface(
                fn=chat_response,
                additional_inputs=[kb_selector],
                examples=["糖尿病的症状有哪些？", "如何预防二型糖尿病？"],
                title="Medical Assistant",
            )
            
            # Event: Change KB
            def on_kb_change(kb_name):
                global current_kb
                current_kb = kb_name
                msg = load_retriever(kb_name)
                return msg
                
            kb_selector.change(on_kb_change, inputs=[kb_selector], outputs=[status_box])

        # Tab 2: Management
        with gr.TabItem("⚙️ Knowledge Base Management"):
            gr.Markdown("### Manage Knowledge Bases")
            
            with gr.Row():
                new_kb_name = gr.Textbox(label="New KB Name", placeholder="e.g., cardiology")
                create_btn = gr.Button("Create KB")
            
            create_msg = gr.Textbox(label="Output")
            
            gr.Markdown("### Upload Documents")
            with gr.Row():
                upload_kb_select = gr.Dropdown(
                    label="Target Knowledge Base",
                    choices=kb_manager.list_kbs(),
                    interactive=True
                )
                file_uploader = gr.File(
                    label="Upload Files (PDF, TXT, MD, Excel)",
                    file_count="multiple"
                )
                upload_btn = gr.Button("Upload & Index")
            
            upload_msg = gr.Textbox(label="Index Status")
            
            # Wiring
            create_btn.click(
                create_new_kb,
                inputs=[new_kb_name],
                outputs=[create_msg, kb_selector, upload_kb_select]
            )
            
            upload_btn.click(
                upload_and_index,
                inputs=[upload_kb_select, file_uploader],
                outputs=[upload_msg]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
