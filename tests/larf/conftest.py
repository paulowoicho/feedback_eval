from typing import Any

import pytest

from larf.data_models.query import Passage, Query


class DummyHit:
    def __init__(self, docid: str, score: float, raw: str):
        self.docid = docid
        self.score = score
        self.lucene_document = {"raw": raw}


class DummySearcher:
    def __init__(self, hits_map: dict[Any, list[DummyHit]], using_rm3=False):
        self.hits_map = hits_map
        self._using_rm3 = using_rm3
        self.set_rm3_called = False
        self.unset_rm3_called = False
        self.last_batch_args = None

    def is_using_rm3(self) -> bool:
        return self._using_rm3

    def set_rm3(self):
        self._using_rm3 = True
        self.set_rm3_called = True

    def unset_rm3(self):
        self._using_rm3 = False
        self.unset_rm3_called = True

    def batch_search(self, *, queries, qids, k, threads):
        self.last_batch_args = {"queries": queries, "qids": qids, "k": k, "threads": threads}
        return {qid: self.hits_map.get(qid, []) for qid in qids}


@pytest.fixture
def make_hit():
    def _make(docid, score, raw):
        return DummyHit(docid=docid, score=score, raw=raw)

    return _make


@pytest.fixture
def make_search_engine():
    def _make(hits_map, using_rm3=False):
        return DummySearcher(hits_map=hits_map, using_rm3=using_rm3)

    return _make


@pytest.fixture
def make_passage():
    """Factory to create a Passage with a given id, score and relevance_assessment."""

    def _make(pid, text="txt", score=0.0, relevance=None):
        p = Passage(id=pid, text=text, score=score)
        p.set_relevance_assessment(relevance)
        return p

    return _make


@pytest.fixture
def make_query(make_passage):
    """Factory to create a Query with an initial set of passages."""

    def _make(qid, passages):
        q = Query(id=qid, text=f"query-{qid}")
        q.set_passages(passages)
        return q

    return _make
