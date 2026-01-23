"""Quantum I-Ching 專案向量資料庫模組.

此模組負責將易經文件儲存到向量資料庫（ChromaDB），並提供語義搜尋功能。
用於 RAG 系統的檢索增強生成。
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from knowledge_loader import IChingDocument, IChingKnowledgeLoader


class IChingVectorStore:
    """易經向量資料庫類別.

    使用 ChromaDB 進行本地持久化儲存，並使用 SentenceTransformers 進行嵌入。
    提供文件的儲存和語義搜尋功能。
    """

    def __init__(self, persist_directory: str = "data/chroma_db") -> None:
        """初始化向量資料庫.

        Args:
            persist_directory: ChromaDB 持久化目錄路徑，預設為 "data/chroma_db"
        """
        # 初始化嵌入函數（使用 SentenceTransformers）
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 確保目錄存在
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客戶端（本地持久化）
        self.client = chromadb.PersistentClient(path=str(persist_path))

        # 取得或建立集合
        self.collection = self.client.get_or_create_collection(
            name="iching_knowledge",
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents: List[IChingDocument]) -> None:
        """將文件加入向量資料庫.

        將 IChingDocument 物件轉換為 ChromaDB 格式並儲存。
        使用 upsert 操作，如果文件已存在則更新。

        Args:
            documents: IChingDocument 物件列表

        Note:
            - 使用 hexagram_id 作為文件 ID（轉換為字串）
            - content 欄位用於嵌入和檢索
            - metadata 保留結構化資訊供後續使用
        """
        # 準備資料列表
        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            # ID 轉換為字串
            ids.append(str(doc.id))
            
            # 文件內容（用於嵌入）
            contents.append(doc.content)
            
            # 元資料（保留結構化資訊）
            metadatas.append(doc.metadata)

        # 使用 upsert 加入或更新文件
        self.collection.upsert(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        print(f"Upserted {len(documents)} documents to VectorDB.")

    def query(
        self,
        query_text: str,
        n_results: int = 1,
        hex_id: Optional[int] = None,
        doc_type: Optional[str] = None,
        line_numbers: Optional[Sequence[int]] = None,
    ) -> List[str]:
        """查詢向量資料庫（可根據 metadata 嚴格過濾）.

        使用語義搜尋找出與查詢文字最相關的文件，同時支援以
        `hex_id`、`type`（main/line）與 `line_number` 等 metadata
        進行精確過濾，確保只在指定卦象／爻位範圍內檢索。

        Args:
            query_text: 查詢文字
            n_results: 返回結果數量，預設為 1
            hex_id: 若提供，僅在指定卦（number）內搜尋
            doc_type: 若提供，限制為 "main" 或 "line"
            line_numbers: 若提供，限制為指定爻位列表（1-6 或 7=用九/用六）

        Returns:
            檢索到的文件內容列表（按相關性排序）

        Note:
            - ChromaDB 返回的結構為字典
            - 需要提取 `['documents'][0]` 取得結果列表
            - 結果按相似度排序（最相關的在前面）
        """
        # 構造 metadata 過濾條件，確保只在目標卦象／爻位中搜尋
        where: dict = {}
        if hex_id is not None:
            where["hex_id"] = hex_id
        if doc_type is not None:
            where["type"] = doc_type
        if line_numbers:
            # 使用 $in 過濾多個爻位
            where["line_number"] = {"$in": list(line_numbers)}

        # 執行查詢
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where or None,
        )

        # 提取文件內容
        # ChromaDB 返回格式: {'ids': [...], 'documents': [[...]], 'metadatas': [[...]], ...}
        if results and 'documents' in results and len(results['documents']) > 0:
            return results['documents'][0]  # 返回第一個查詢的結果列表
        else:
            return []


def build_vector_db() -> None:
    """建立向量資料庫的輔助函數.

    執行完整的資料載入和向量化流程：
    1. 載入易經文件
    2. 初始化向量資料庫
    3. 將文件加入向量資料庫

    Note:
        此函數可以在初始化時執行，將所有易經知識載入向量資料庫。
    """
    print("開始建立向量資料庫...")
    
    # 步驟 1: 載入文件
    print("[步驟 1] 載入易經知識庫文件...")
    loader = IChingKnowledgeLoader()
    documents = loader.load_documents()
    print(f"[成功] 載入 {len(documents)} 個文件\n")

    # 步驟 2: 初始化向量資料庫
    print("[步驟 2] 初始化向量資料庫...")
    vector_store = IChingVectorStore()
    print(f"[成功] 向量資料庫已初始化\n")

    # 步驟 3: 加入文件
    print("[步驟 3] 將文件加入向量資料庫...")
    vector_store.add_documents(documents)
    print(f"[成功] 向量資料庫建立完成！\n")


if __name__ == "__main__":
    build_vector_db()