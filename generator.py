import os
import sys
from openai import OpenAI
from typing import List, Dict

# Add project root to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=Config.llm_api_key,
            base_url=Config.llm_base_url
        )
        self.model = Config.llm_model

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer based on the query and retrieved contexts.
        """
        # Construct the system prompt and user prompt
        system_prompt = "你是一个专业的医疗助手。请基于提供的上下文回答用户的问题。如果上下文中没有相关信息，请诚实地说明。"
        
        # Combine contexts
        context_str = "\n\n".join([f"上下文 {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        user_content = f"问题: {query}\n\n参考信息:\n{context_str}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3, # Lower temperature for more factual answers
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {e}"

if __name__ == "__main__":
    # Simple test
    generator = Generator()
    test_query = "什么是糖尿病？"
    test_contexts = [
        "糖尿病是一组以高血糖为特征的代谢性疾病。",
        "高血糖则是由于胰岛素分泌缺陷或其生物作用受损，或两者兼有引起。"
    ]
    
    print(f"Query: {test_query}")
    print("Generating answer...")
    answer = generator.generate(test_query, test_contexts)
    print(f"Answer: {answer}")
