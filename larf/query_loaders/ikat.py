import json

from ..data_models.query import Query
from .base import BaseQueryLoader


class iKAT2023(BaseQueryLoader):
    def get_queries(self) -> list[Query]:
        queries = []
        with open(self.queries_path, "r", encoding="utf‑8") as f:
            conversations = json.load(f)

        for conv in conversations:
            for turn in conv["turns"]:
                query_id = f"{conv['number']}_{turn['turn_id']}"
                query = turn["resolved_utterance"]
                q_obj = Query(
                    id=query_id,
                    text=query,
                )
                if self.resume_path:
                    q_obj.set_passages(self._load_passages(query_id))
                queries.append(q_obj)

        return queries
