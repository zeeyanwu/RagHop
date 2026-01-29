# RagHop Reproduction Project

This project is a modular reproduction of a Medical RAG (Retrieval-Augmented Generation) system. It is designed to be clean, secure, and stable, leveraging OpenAI's state-of-the-art models and FAISS for efficient vector retrieval.

## 🚀 Phase I: Core RAG Implementation (Completed)

In Phase I, we have successfully established the foundational architecture for the RAG system.

### Key Features
*   **Modular Architecture**:
    *   `indexer.py`: Basic text loading and vectorization logic.
    *   `retriever.py`: Manages FAISS vector index loading and similarity search.
    *   `generator.py`: Interfaces with OpenAI GPT models for context-aware answer generation.
    *   `text2vec.py`: Unified wrapper for OpenAI Embeddings (`text-embedding-3-small`).
*   **Core RAG Functionality**:
    *   Supports loading a pre-built knowledge base (`kb_index`).
    *   Implements the complete Retrieve-then-Generate pipeline.
*   **Modern UI**: 
    *   Built with **Gradio 6.x**.
    *   Features a clean, dictionary-based Chat Interface.
*   **Security**: 
    *   All API keys and sensitive configurations are managed via `.env` file.
    *   No hardcoded credentials in the source code.

### Installation & Usage

1.  **Prerequisites**:
    *   Python 3.10+
    *   OpenAI API Key

2.  **Setup**:
    ```bash
    cd reproduction
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    *   Create a `.env` file in the `reproduction` directory:
        ```env
        OPENAI_API_KEY=sk-your-key-here
        OPENAI_BASE_URL=https://api.openai.com/v1
        LLM_MODEL=gpt-4o
        ```

4.  **Run**:
    ```bash
    python app.py
    ```

---

## 🔮 Phase II: Knowledge Management & Advanced Intelligence (Roadmap)

The next phase focuses on making the system dynamic and smarter.

### Planned Features

#### 1. 📂 Dynamic Knowledge Base Management
*   **Goal**: Allow users to create, switch, and manage multiple knowledge bases via the UI.
*   **Features**:
    *   **UI Integration**: A dedicated "Knowledge Management" tab in Gradio.
    *   **Multi-format Support**: Support for parsing PDF, Excel (.xlsx), CSV, and Markdown files.
    *   **Semantic Chunking**: Integration of `llama-index` for smarter text segmentation.

#### 2. 🌐 Web Search Integration (联网搜索)
*   **Goal**: Supplement local knowledge base with real-time information from the internet.
*   **Implementation**:
    *   Integrate search APIs (e.g., Bing Search API, Google Custom Search).
    *   **Hybrid Retrieval**: Intelligently combine local KB results with web search results.

#### 3. 🧠 Multi-hop Reasoning (多跳推理)
*   **Goal**: Solve complex questions that require connecting information from multiple documents.
*   **Implementation**:
    *   **Query Decomposition**: Break down complex questions into sub-queries.
    *   **Iterative Retrieval**: Use intermediate answers to guide further retrieval.

#### 4. 📊 Advanced Data Processing
*   **Table Understanding**: Enhanced parsing for complex Excel structures.
*   **Multimodal RAG**: Support for interpreting images and charts in medical documents.
