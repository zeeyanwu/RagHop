import os
from dotenv import load_dotenv

# 加载当前目录下的 .env 文件
# override=True 确保 .env 中的变量覆盖系统环境变量（如果有冲突）
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class Config:
    """
    项目基础配置文件
    包含路径设置、模型参数和API凭证
    """
    # 项目根目录 (当前文件所在目录)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 知识库存储目录
    KB_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_bases")
    
    # 确保知识库目录存在
    os.makedirs(KB_BASE_DIR, exist_ok=True)
    
    # API 配置 (使用 OpenAI)
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not API_KEY or API_KEY == "your_openai_api_key_here":
        print("警告: 未找到有效的 OPENAI_API_KEY，请在 reproduction/.env 文件中设置。")

    # Embedding 模型配置 (OpenAI)
    # 推荐使用 text-embedding-3-small 或 text-embedding-3-large
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536 # text-embedding-3-small 的默认维度
    BATCH_SIZE = 10
    
    # LLM 模型配置 (OpenAI)
    llm_api_key = os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")
    
    # Keep uppercase for compatibility if needed
    API_KEY = llm_api_key
    BASE_URL = llm_base_url
    LLM_MODEL = llm_model
    
    # 检索与分块配置
    CHUNK_SIZE = 512       # 文本块大小
    CHUNK_OVERLAP = 50     # 重叠大小，防止切断上下文
    TOP_K = 3             # 默认检索召回数量
