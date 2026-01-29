import numpy as np
from openai import OpenAI
from config import Config

class TextVector:
    def __init__(self):
        """初始化 TextVector，加载 API 配置"""
        self.api_key = Config.API_KEY
        self.base_url = Config.BASE_URL
        self.model_name = Config.EMBEDDING_MODEL
        self.batch_size = Config.BATCH_SIZE
        
        # 实例化 OpenAI 客户端
        if not self.api_key:
            # 仅打印警告，允许实例化，但在调用时可能会失败
            print("Warning: API Key not set. Embeddings will fail.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def get_vec_batch(self, texts):
        """
        获取文本列表的向量
        :param texts: 文本列表 [str]
        :return: numpy array of shape (len(texts), dim)
        """
        if not texts:
            return np.array([])
            
        # 过滤无效文本
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid_texts:
            print("Warning: No valid text to vectorize")
            return np.array([])

        if not self.client:
            raise ValueError("OpenAI Client not initialized. Please set API Key.")

        all_vectors = []
        
        # 分批处理以避免超过 API 限制
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            try:
                # 移除换行符，通常能提升 Embedding 效果
                batch = [t.replace("\n", " ") for t in batch]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                # 提取向量
                vectors = [data.embedding for data in response.data]
                all_vectors.extend(vectors)
                
            except Exception as e:
                print(f"Error getting embeddings for batch {i}: {e}")
                raise e

        return np.array(all_vectors)

# 创建单例实例方便调用
text_vectorizer = TextVector()

# 导出便捷函数，保持与原项目类似的接口风格
def get_vector(texts):
    if isinstance(texts, str):
        texts = [texts]
    return text_vectorizer.get_vec_batch(texts)

if __name__ == "__main__":
    # 简单的测试代码
    try:
        test_text = ["Hello, world!", "这是一个测试"]
        vectors = get_vector(test_text)
        print(f"Test successful. Vector shape: {vectors.shape}")
        print(f"First vector sample: {vectors[0][:5]}...")
    except Exception as e:
        print(f"Test failed: {e}")
