import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from src.models.schemas import Document

class DataLoader:
    @staticmethod
    def load_from_directory(dir_path: str) -> List[Document]:
        docs = []
        path = Path(dir_path)
        for file in path.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            docs.append(Document(doc_id=file.stem, content=content, source="工程归档卷宗"))
        return docs

    @staticmethod
    def load_test_case(json_path: str) -> Tuple[str, List[Document]]:
        """
        兼容加载单个字典用例或评测列表文件中的第 1 条用例
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            if len(data) == 0:
                return "", []
            case_data = data[0]
        else:
            case_data = data

        query = case_data.get("query", "")
        raw_candidates = case_data.get("candidates", [])
        
        docs = []
        for item in raw_candidates:
            doc_id = item.get("doc_id", "UNKNOWN_DOC")
            content = item.get("content", "")
            source = item.get("source", "未知来源")
            docs.append(Document(doc_id=doc_id, content=content, source=source))

        return query, docs

    @staticmethod
    def load_benchmark_dataset(json_path: str) -> List[Dict[str, Any]]:
        """
        加载完整的多案例评测基准列表，兼容 list 与 dict 格式
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
