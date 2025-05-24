from collections import defaultdict
from logging import getLogger
from typing import Callable

from pyserini.search.lucene import LuceneSearcher  # type: ignore[import-untyped]

from larf.data_models.query import Passage, Query
from larf.retrieval_system.base import Component

from .query_by_document.pyserini import (
    EXPANSION_METHOD_TO_FN,
    ExpansionMethod,
    PyseriniBasedQBDExpander,
)
from .query_reformulation.pyserini import (
    REFORMULATION_METHOD_TO_FN,
    PyseriniBasedQRExpander,
    ReformulationMethod,
)

logger = getLogger(__name__)


class FusionExpander(Component):
    def __init__(
        self, qbd_component: PyseriniBasedQBDExpander, qr_component: PyseriniBasedQRExpander
    ) -> None:
        self.qbd_component = qbd_component
        self.qr_component = qr_component


class QBDThenQR(FusionExpander):
    def run(self, queries: list[Query]) -> list[Query]:
        queries = self.qbd_component.run(queries)
        return self.qr_component.run(queries)


class QRThenQBD(FusionExpander):
    def run(self, queries: list[Query]) -> list[Query]:
        queries = self.qr_component.run(queries)
        return self.qbd_component.run(queries)


class QRQBDRRF(Component):
    def __init__(
        self,
        search_engine: LuceneSearcher,
        rerank_budget: int = 1000,
        num_threads: int = 1024,
        top_n_qr: int = 1,
        result_parser_fn: Callable[[str], str] | None = None,
    ):
        self.search_engine = search_engine
        self.rerank_budget = rerank_budget
        self.num_threads = num_threads
        self.reformulation_fn = lambda q: PyseriniBasedQRExpander.get_new_queries(
            q, top_n_qr, REFORMULATION_METHOD_TO_FN[ReformulationMethod.FULL_TEXT]
        )
        self.expansion_fn = lambda q, p: EXPANSION_METHOD_TO_FN[ExpansionMethod.FULL_TEXT](q, p)

        self._result_parser_fn: Callable[[str], str] = result_parser_fn or (lambda raw: raw)

    def fuse_results(self, ranked_lists: list[list]) -> list[Passage]:
        score_accumulator: dict[str, float] = defaultdict(float)
        id_to_passage: dict[str, Passage] = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):  # 1-based rank
                print(hit)
                score_accumulator[hit.docid] += 1.0 / (60 + rank)
                if hit.docid not in id_to_passage:
                    text = hit.lucene_document.get("raw", "")
                    parsed = self._result_parser_fn(text)
                    id_to_passage[hit.docid] = Passage(id=hit.docid, text=parsed, score=hit.score)

        # sort by descending cumulative score, then docid for tie-breaker
        fused_ids = sorted(score_accumulator, key=lambda docid: (-score_accumulator[docid], docid))
        return [id_to_passage[docid] for docid in fused_ids]

    def run(self, queries: list[Query]) -> list[Query]:
        # Code replication is intentional.
        qbd_queries: dict[str | int, str] = {
            f"{q.id}:::{p.id}": self.expansion_fn(q.text, p.text)
            for q in queries
            for p in q.passages
        }
        qr_queries = {q.id: self.reformulation_fn(q) for q in queries}

        # Overwriting should never happen
        all_queries: dict[str | int, str] = qbd_queries | qr_queries

        # Batch search
        results = self.search_engine.batch_search(
            queries=list(all_queries.values()),
            qids=list(all_queries.keys()),
            k=self.rerank_budget,
            threads=self.num_threads,
        )

        for query in queries:
            query_results = [res for key, res in results.items() if key.startswith(query.id)]
            fused = self.fuse_results(query_results)

            existing_ids = {p.id for p in query.passages}
            new_passages = [p for p in fused if p.id not in existing_ids]

            query.set_passages((query.passages + new_passages)[: self.rerank_budget])

        return queries


class QRQBDWeightedRRF(QRQBDRRF):
    def __init__(
        self,
        search_engine: LuceneSearcher,
        rerank_budget: int = 1000,
        num_threads: int = 1024,
        top_n_qr: int = 1,
        result_parser_fn: Callable[[str], str] | None = None,
        qbd_weight: float = 0.5,
        qr_weight: float = 0.5,
    ):
        super().__init__(
            search_engine, rerank_budget, num_threads, top_n_qr, result_parser_fn=result_parser_fn
        )
        self.qbd_weight = qbd_weight
        self.qr_weight = qr_weight

    def run(self, queries: list[Query]) -> list[Query]:
        qbd_queries: dict[str | int, str] = {
            f"{q.id}:::{p.id}": self.expansion_fn(q.text, p.text)
            for q in queries
            for p in q.passages
        }
        qr_queries = {q.id: self.reformulation_fn(q) for q in queries}

        all_queries: dict[str | int, str] = qbd_queries | qr_queries

        # Batch search
        results = self.search_engine.batch_search(
            queries=list(all_queries.values()),
            qids=list(all_queries.keys()),
            k=self.rerank_budget,
            threads=self.num_threads,
        )

        for query in queries:
            base_seen = {p.id for p in query.passages}
            remaining = self.rerank_budget - len(base_seen)
            if remaining <= 0:
                continue

            qr_lim = int(remaining * self.qr_weight)
            qbd_lim = remaining - qr_lim

            qr_hits = []
            if qr_lim:
                qr_hits = results.get(query.id, [])[:qr_lim]
            qbd_hits = [
                res[:qbd_lim] for qid, res in results.items() if qid.startswith(f"{query.id}::")
            ]

            fused = self.fuse_results([qr_hits] + qbd_hits)
            additions = [p for p in fused if p.id not in base_seen]
            query.set_passages((query.passages + additions)[: self.rerank_budget])

        return queries
