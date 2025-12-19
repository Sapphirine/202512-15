# build_multi_pdf_rag.py

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# === 配置 ===
PDF_DIR = "pdfs"  # ← 放所有 PDF 的文件夹
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 中文建议换 "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OUTPUT_DIR = "rag_data"

# === 1. 加载所有 PDF ===
print("📄 正在加载所有 PDF 文件...")
loader = DirectoryLoader(
    PDF_DIR,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True  # 加速加载
)
documents = loader.load()

print(f"✅ 共加载 {len(documents)} 页（来自多个 PDF）")

# === 2. 切分文本（保留元数据：source, page）===
print("✂️ 正在切分文本...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ". ", "? ", "! ", " ", ""]
)
chunks_with_metadata = text_splitter.split_documents(documents)

# 提取纯文本和元数据
texts = [doc.page_content for doc in chunks_with_metadata]
metadatas = [
    {
        "source": os.path.basename(doc.metadata.get("source", "unknown")),
        "page": doc.metadata.get("page", "N/A")
    }
    for doc in chunks_with_metadata
]

print(f"✅ 共切分 {len(texts)} 个段落")

# === 3. 生成 Embedding ===
print("🧠 正在生成 embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = model.encode(texts, show_progress_bar=True)

# === 4. 构建 FAISS 索引 ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# === 5. 保存所有数据 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{OUTPUT_DIR}/texts.pkl", "wb") as f:
    pickle.dump(texts, f)
with open(f"{OUTPUT_DIR}/metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
faiss.write_index(index, f"{OUTPUT_DIR}/faiss.index")

print(f"✅ 多 PDF 向量库已保存到 {OUTPUT_DIR}/")