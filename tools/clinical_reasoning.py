import os
import sys
import logging
import json

# 确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from .register import function_register

logger = logging.getLogger("ClinicalReasoning")

@function_register.register(
    name="clinical_reasoning_synthesis",
    description=(
        "Synthesize results from arrhythmia analysis and stress assessment into a unified clinical hypothesis. "
        "Explores the physiological relationship between autonomic tone (stress) and cardiac rhythm."
    ),
    parameters=[
        {"name": "arrhythmia_results", "type": "object", "description": "Output from arrhythmiaAnalysis tool", "required": True},
        {"name": "stress_results", "type": "object", "description": "Output from analyze_stress tool", "required": True}
    ]
)
def clinical_reasoning_synthesis(arrhythmia_results, stress_results):
    """
    生理学逻辑综合分析：
    1. 提取 Stress 中的 HRV 指标 (LF/HF, RMSSD)
    2. 提取 Arrhythmia 中的主要发现 (MI, CD, etc.)
    3. 根据生理学常识提供临床假设 (Hypothesis)
    """
    try:
        # 解析输入 (兼容列表或字典嵌套结果)
        arr_data = arrhythmia_results[0] if isinstance(arrhythmia_results, list) else arrhythmia_results
        str_data = stress_results[0] if isinstance(stress_results, list) else stress_results
        
        findings = {
            "rhythm_status": arr_data.get('primary_finding', 'Unknown'),
            "stress_level": str_data.get('stress_assessment', 'Unknown'),
            "key_metrics": {
                "LF/HF": str_data.get('hrv_metrics', {}).get('LF/HF', 0),
                "Arrhythmia_Probs": arr_data.get('findings', {})
            }
        }
        
        # 推理逻辑 (Physiological Reasoning)
        hypothesis = "Baseline Analysis: No significant acute correlation detected."
        
        lf_hf = findings['key_metrics']['LF/HF']
        rhythm = findings['rhythm_status']
        
        if lf_hf > 2.5 and rhythm != 'Normal':
            hypothesis = (
                "High sympathetic dominance (LF/HF > 2.5) is observed concurrently with rhythm abnormalities. "
                "This suggests that autonomic arousal may be triggering or exacerbating ectopic activity."
            )
        elif lf_hf < 0.5 and rhythm == 'Normal':
            hypothesis = "Strong parasympathetic tone and stable rhythm indicate a healthy recovery or resting state."
        elif rhythm == 'Myocardial Infarction' and findings['key_metrics']['Arrhythmia_Probs'].get('Myocardial Infarction', 0) > 0.8:
            hypothesis = "Critical Observation: High probability of MI detected. Physiological data suggests acute cardiac distress."

        return {
            "synthesis_summary": f"Target is currently {findings['stress_level']} with a rhythm finding of '{rhythm}'.",
            "physiological_hypothesis": hypothesis,
            "reasoning_chain": [
                f"Step 1: Analyzed HRV - Sympathovagal balance (LF/HF) is {lf_hf}.",
                f"Step 2: Cross-referenced with Arrhythmia probability mapping.",
                f"Step 3: Applied clinical correlation rules."
            ]
        }
    except Exception as e:
        logger.error(f"Reasoning synthesis failed: {e}")
        return {"error": "Could not synthesize data. Ensure both tool outputs are provided."}

@function_register.register(
    name="medical_knowledge_lookup",
    description="Search the local clinical knowledge base for diagnostic guidelines, physiological definitions, and treatment protocols.",
    parameters=[
        {"name": "query", "type": "string", "description": "Medical term or clinical question (e.g., 'What is the diagnostic criteria for STTC?')", "required": True}
    ]
)
def medical_knowledge_lookup(query, config=None):
    """
    RAG 接口封装
    """
    try:
        from RAG.embedder import BGEEmbedder
        from RAG.searcher import FaissSearcher
        
        # 路径解析
        rag_dir = os.path.join(project_root, 'RAG')
        index_path = os.path.join(rag_dir, 'faiss.index')
        text_path = os.path.join(rag_dir, 'chunks.pkl')
        
        # 初始化 (理想情况下这里应该使用单例模式优化内存)
        embedder = BGEEmbedder()
        searcher = FaissSearcher(index_path, text_path)
        
        # 向量化查询
        query_vec = embedder.encode([query])[0]
        # 检索
        results = searcher.search(query_vec, top_k=3)
        
        formatted_results = []
        for text, score in results:
            formatted_results.append({"knowledge_chunk": text, "relevance_score": round(score, 4)})
            
        return formatted_results
    except Exception as e:
        logger.error(f"RAG Lookup failed: {e}")
        return {"error": "Knowledge base unavailable. Falling back to internal LLM training data."}
