import torch
import json
import re
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Reranker:
    def __init__(
            self,
            model_name: str = "BAAI/bge-reranker-base",
            device: str = "cpu",
            table_boost: float = 0.5,  # 【核心】大幅提高到0.5
            table_max_length: int = 512,
            debug: bool = False
    ):
        self.device = device
        self.table_boost = table_boost
        self.table_max_length = table_max_length
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:450]

    def _format_insurance_table(self, table_data: Dict, query: str) -> str:
        if not table_data or "header" not in table_data or "rows" not in table_data:
            return ""

        header = [str(cell) for cell in table_data["header"]]
        rows = [[str(cell) for cell in row] for row in table_data["rows"][:5]]

        table_text = "表格内容："
        for row in rows:
            table_text += " | ".join(row) + "；"
        return table_text

    def rerank(self, query: str, candidates: List[Dict[str, Any]], modalities: List[str], query_type: str = "text") -> \
    List[float]:
        if not candidates or len(candidates) != len(modalities):
            return []

        processed_texts = []
        is_table_flags = []

        for cand, mod in zip(candidates, modalities):
            if mod in ["table", "column", "row"]:
                table_data = cand.get("table")
                if table_data:
                    processed = self._format_insurance_table(table_data, query)
                    processed = self._clean_text(processed)
                    processed_texts.append(processed)
                    is_table_flags.append(True)
                else:
                    processed = self._clean_text(cand.get("text", ""))
                    processed_texts.append(processed)
                    is_table_flags.append(False)
            else:
                processed = self._clean_text(cand.get("text", ""))
                processed_texts.append(processed)
                is_table_flags.append(False)

        pairs = [[query, t] for t in processed_texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.table_max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        # 【核心】动态加成再提高
        boost_map = {
            "text": 0.3,  # 文本查询：表格加0.3
            "column": 0.6,  # 数值查询：表格加0.6
            "row": 0.5,
            "table": 0.7
        }
        current_boost = boost_map.get(query_type, 0.4)

        final_scores = []
        for score, is_table in zip(scores, is_table_flags):
            if is_table:
                final_scores.append(score + current_boost)
            else:
                final_scores.append(score)

        return final_scores