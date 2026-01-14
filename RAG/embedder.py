# embedder
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
import os


class BGEEmbedder:
    def __init__(self, model_name=None):
        """
        初始化 BGE 嵌入模型。
        
        Args:
            model_name: 模型路径或名称。支持：
                - None: 使用默认的本地路径 (RAG/sentenceModel/bge-m3)
                - 相对路径: 相对于 RAG 目录
                - 绝对路径: 直接使用
                - HuggingFace 模型名: 如 "BAAI/bge-m3"
        """
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        if model_name is None:
            # 默认使用本地模型
            model_name = os.environ.get("BGE_MODEL_PATH", "sentenceModel/bge-m3")
        
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(model_name) and not model_name.startswith("BAAI/"):
            model_path = os.path.join(CURRENT_DIR, model_name)
        else:
            model_path = model_name
        
        # 检查本地路径是否存在
        if os.path.isdir(model_path):
            print(f"[BGEEmbedder] Loading model from local path: {model_path}")
        else:
            print(f"[BGEEmbedder] Local path not found, trying as HuggingFace model: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.eval()
            print(f"[BGEEmbedder] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model from '{model_path}'. "
                f"Please ensure the model exists at the specified path or is a valid HuggingFace model name. "
                f"Error: {e}"
            )

    @torch.no_grad()
    def encode(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.cpu().tolist()