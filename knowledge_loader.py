"""Quantum I-Ching 專案知識庫載入模組.

從 data/iching_complete.json 載入易經資料（由 setup_iching_db 自
john-walks-slow/open-iching 下載並轉為統一格式），切成主卦與六爻文件，
並可重建 ChromaDB 向量庫。僅解析既有的 JSON，不進行 AI 生成。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


@dataclass
class IChingDocument:
    """易經文件資料類別（供 vector_store 等模組相容使用）.

    Attributes:
        id: 文件 ID，可為 int 或 str（如 "hex_1_main", "hex_1_line_1"）
        content: 用於嵌入與檢索的內文
        metadata: 結構化元資料（hex_id, type, name, line_number 等）
    """
    id: Union[int, str]
    content: str
    metadata: Dict[str, Any]


class IChingKnowledgeLoader:
    """易經知識載入器（統一格式：number, name, judgment, image, lines）.

    讀取 iching_complete.json，產生主卦＋六爻文件，並可重建 ChromaDB 集合 "iching_knowledge"。
    """

    def __init__(self, file_path: str = "data/iching_complete.json") -> None:
        self.file_path = Path(file_path)

    def load_documents(self) -> List[IChingDocument]:
        """從 iching_complete.json 載入並切成主卦＋六爻之 IChingDocument 列表."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"知識庫檔案不存在: {self.file_path}\n"
                f"請先執行: python setup_iching_db.py"
            )
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) != 64:
            raise ValueError(f"iching_complete.json 須為 64 卦之 list，目前 len={len(data) if isinstance(data, list) else 'N/A'}")

        out: List[IChingDocument] = []
        for item in data:
            number = item.get("number", 0)
            name = item.get("name") or item.get("hexagram_name") or item.get("title") or "?"
            judgment = item.get("judgment") or item.get("judgement") or ""
            image = item.get("image") or item.get("img") or ""

            # 主卦
            main_content = f"【{number}. {name}卦】\n卦辭：{judgment}\n象曰：{image}"
            main_meta: Dict[str, Any] = {"hex_id": number, "type": "main", "name": name, "line_number": 0}
            out.append(IChingDocument(id=f"hex_{number}_main", content=main_content, metadata=main_meta))

            # 六爻（依 lines 迭代，支援 6 或 7 條，如乾卦 Use Nine）
            for i, line in enumerate(item.get("lines") or []):
                if not isinstance(line, dict):
                    continue
                pos = line.get("position")
                if pos is None:
                    pos = i + 1
                try:
                    pos = int(pos)
                except (TypeError, ValueError):
                    pos = i + 1
                meaning = line.get("meaning") or line.get("text") or ""
                xiang = line.get("xiang") or ""
                line_content = f"【{name}卦】 {meaning}\n象曰：{xiang}"
                line_meta: Dict[str, Any] = {"hex_id": number, "type": "line", "line_number": pos, "name": name}
                out.append(IChingDocument(id=f"hex_{number}_line_{pos}", content=line_content, metadata=line_meta))

        return out

    def build_vector_db(self, persist_path: str = "data/chroma_db") -> None:
        """清除並重建 ChromaDB 集合 iching_knowledge，寫入主卦＋六爻文件。"""
        docs = self.load_documents()
        ids = [str(d.id) for d in docs]
        documents = [d.content for d in docs]
        metadatas = [d.metadata for d in docs]

        Path(persist_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        try:
            client.delete_collection("iching_knowledge")
        except Exception:
            pass
        coll = client.create_collection("iching_knowledge", embedding_function=embedding_fn)
        coll.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[OK] Knowledge Base Rebuilt: {len(docs)} documents indexed.")


if __name__ == "__main__":
    loader = IChingKnowledgeLoader()
    loader.build_vector_db()
