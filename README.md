# RagHop Reproduction Project

This project is a modular reproduction and enhancement of a Medical RAG (Retrieval-Augmented Generation) system. It is designed to be clean, secure, and easily extensible, leveraging OpenAI's state-of-the-art models and FAISS for efficient vector retrieval.

## 🚀 Phase I: Core RAG & Knowledge Management (Completed)

In Phase I, we have successfully rebuilt the core infrastructure and implemented essential knowledge base management features.

### Key Features
*   **Modular Architecture**:
    *   `indexer.py`: Handles file parsing (PDF, Excel, CSV, TXT), chunking, and vectorization.
    *   `retriever.py`: Manages FAISS vector index loading and similarity search.
    *   `generator.py`: Interfaces with OpenAI GPT models for context-aware answer generation.
    *   `text2vec.py`: Unified wrapper for OpenAI Embeddings (`text-embedding-3-small`).
*   **Knowledge Base Management**:
    *   **Dynamic KB Creation**: Users can create multiple isolated knowledge bases via the UI.
    *   **Multi-format Support**: 
        *   **PDF**: Text extraction using `PyMuPDF`.
        *   **Excel/CSV**: Row-based parsing for structured medical data.
        *   **Markdown/Text**: Standard text processing.
    *   **Semantic Chunking**: Utilizes `llama-index`'s `SentenceSplitter` for context-preserving text segmentation.
*   **Modern UI**: 
    *   Built with **Gradio 6.x**.
    *   Features a Chat Interface and a dedicated Knowledge Base Management tab.
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

## 🔮 Phase II: Advanced Reasoning & Web Integration (Roadmap)

The next phase focuses on enhancing the system's intelligence and breadth of information.

### planned Features

#### 1. 🌐 Web Search Integration (联网搜索)
*   **Goal**: Supplement local knowledge base with real-time information from the internet.
*   **Implementation**:
    *   Integrate search APIs (e.g., Bing Search API, Google Custom Search, or Serper).
    *   **Hybrid Retrieval Strategy**: Automatically decide whether to use local KB, Web Search, or both based on query intent and confidence scores.
    *   **Citation Source**: Clearly distinguish between "Internal Knowledge" and "Web Source" in the final answer.

#### 2. 🧠 Multi-hop Reasoning (多跳推理)
*   **Goal**: Solve complex questions that require connecting information from multiple documents or chunks.
*   **Implementation**:
    *   **Query Decomposition**: Break down complex questions into sub-queries (e.g., "What is the treatment for the disease caused by X?" -> "What disease is caused by X?" + "What is the treatment for [Disease]?").
    *   **Iterative Retrieval**: Use the answer from the first retrieval step as the context for the second step.
    *   **Chain-of-Thought (CoT)**: Guide the LLM to explicitly reason through the steps before generating the final answer.

#### 3. 📊 Advanced Data Processing
*   **Enhanced Table Understanding**: Improved parsing logic for complex Excel headers and merged cells.
*   **Image/Chart Analysis**: Integration of Vision-Language Models (e.g., GPT-4o Vision) to interpret charts and images within PDFs.

#### 4. 🧪 Evaluation & Optimization
*   **RAGAS Integration**: Automated evaluation of retrieval precision and generation faithfulness.
*   **Hyperparameter Tuning**: Optimization of chunk sizes and overlap based on evaluation metrics.
