from larf.data_models.query import Passage
from larf.retrieval_system.searchers.pyserini import PyseriniLuceneSearcher


def test_search(make_query, make_hit, make_search_engine):
    # Initial passages has D1
    existing = Passage(id="D1", text="old", score=1.0)
    q = make_query("QF", passages=[existing])

    # Dummy hits include D1 (should be filtered) and D2, D3, D4
    hits_map = {
        "QF": [
            make_hit("D1", 0.9, "new1"),
            make_hit("D2", 0.8, "new2"),
            make_hit("D3", 0.7, "new3"),
            make_hit("D4", 0.6, "new4"),
        ]
    }
    engine = make_search_engine(hits_map)
    # num_results=3 => after combining existing + new [D2, D3, D4], take first 3
    searcher = PyseriniLuceneSearcher(search_engine=engine, num_results=3)
    out = searcher.run([q])
    passages = out[0].passages

    # D1 should remain first, then top 2 new (D2, D3) to fill up to 3 total
    assert [p.id for p in passages] == ["D1", "D2", "D3"]
    # Scores correspond to the hits
    assert [p.score for p in passages] == [1.0, 0.8, 0.7]
