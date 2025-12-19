# ask_pdf.py (增强版 - 显示完整 output)

import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
import os

# === 配置 ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 中文建议换 "BAAI/bge-small-zh-v1.5"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
RAG_DATA_DIR = "rag_data"

# === 加载向量库和元数据 ===
print("🔍 正在加载向量库和元数据...")
model_emb = SentenceTransformer(EMBEDDING_MODEL_NAME)
index = faiss.read_index(os.path.join(RAG_DATA_DIR, "faiss.index"))
with open(os.path.join(RAG_DATA_DIR, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)
with open(os.path.join(RAG_DATA_DIR, "metadatas.pkl"), "rb") as f:
    metadatas = pickle.load(f)

def retrieve_with_metadata(query: str, top_k: int = 3):
    query_vec = model_emb.encode([query])
    D, I = index.search(np.array(query_vec, dtype=np.float32), top_k)
    results = []
    for idx in I[0]:
        results.append({
            "text": texts[idx],
            "source": metadatas[idx]["source"],
            "page": metadatas[idx]["page"]
        })
    return results

def ask_pdf(question: str):
    print("\n" + "="*60)
    print(f"❓ 问题: {question}")
    print("="*60)

    # 1. 检索相关段落
    retrieved = retrieve_with_metadata(question, top_k=3)
    context_text = "\n\n".join([r["text"] for r in retrieved])

    print("📄 检索到的上下文（来自 PDF）:")
    print("-" * 40)
    for i, r in enumerate(retrieved, 1):
        print(f"[{i}] 来源: {r['source']} (第 {r['page']} 页)")
        print(f"    内容: {r['text'][:200]}...\n")

    # 2. 构造 prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based ONLY on the provided document context. "
                "Cite the source if possible. If the answer is not in the context, say 'I don't know based on the provided documents.'"
            )
        },
        {
            "role": "user",
            "content": f"Document context:\n{context_text}\n\nQuestion: {question}"
        }
    ]

    # 3. 调用 vLLM
    print("🧠 正在调用 Qwen2.5-3B-Instruct...")
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.1
            },
            timeout=120
        )
        response.raise_for_status()
    except Exception as e:
        print(f"❌ 调用 vLLM 失败: {e}")
        return

    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()

    # 4. 显示完整输出
    print("✅ 模型输出:")
    print("-" * 40)
    print(answer)
    print("-" * 40)

    # 5. 显示来源
    sources = list(set([f"{r['source']} (p.{r['page']})" for r in retrieved]))
    print(f"📚 来源: {', '.join(sources)}")
    print("="*60 + "\n")

# === 交互式问答 ===
if __name__ == "__main__":
    print("🤖 PDF RAG 问答系统已启动！输入 'quit' 退出。")
    while True:
        question = input("❓ 请输入你的问题: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 再见！")
            break
        if not question:
            continue
        ask_pdf(question)