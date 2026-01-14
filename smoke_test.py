import os
import sys
import torch

# Ensure current directory is in path
sys.path.append(os.getcwd())

def diagnostic_test():
    print("--- 🏥 ECGAgent System Diagnostic (全量体检) ---")
    
    # 1. 模块导入与注册检查
    try:
        from tools.register import registry
        import tools.arrhythmia
        import tools.stress_tool
        import tools.clinical_reasoning
        
        tools_list = registry.get_all_metadata()
        # 修复 None 处理，确保打印可读
        tool_names = [str(t.get('name', 'Unnamed')) for t in tools_list]
        print(f"✅ Module Imports: OK")
        print(f"✅ Registered Tools: {len(tool_names)} found ({', '.join(tool_names)})")
    except Exception as e:
        print(f"❌ Module Registration Error: {e}")

    # 2. 核心模型权重检查
    print("\n--- 🧠 Model Weights Check ---")
    weights = {
        "Arrhythmia (ResNet)": "eval/arrhythmia/checkpoints/best_model.pth",
        "Stress (MLP)": "tools/localModels/stress_mlp.pth",
        "Scaler (Pickle)": "tools/localModels/stress_scaler.pkl"
    }
    for name, path in weights.items():
        if os.path.exists(path):
            print(f"✅ {name}: Found ({os.path.getsize(path)/1024:.1f} KB)")
        else:
            print(f"⚠️ {name}: MISSING (Path: {path})")

    # 3. RAG 知识库与本地 LLM 模型检查
    print("\n--- 📚 Knowledge Base (RAG) Check ---")
    rag_files = {
        "FAISS Index": "RAG/faiss.index",
        "Text Chunks": "RAG/chunks.pkl",
        "Embedding Model": "RAG/sentenceModel/bge-m3/config.json"
    }
    for name, path in rag_files.items():
        if os.path.exists(path):
            print(f"✅ {name}: Found")
        else:
            print(f"⚠️ {name}: MISSING")

    # 4. 运行时设备检查
    print("\n--- ⚡ Runtime Check ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💡 Execution Device: {device}")
    
    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    diagnostic_test()
