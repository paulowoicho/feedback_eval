from larf.retrieval_system.expanders.fusion import QRQBDRRF, QRQBDWeightedRRF


def test_qrqbdr(make_hit, make_search_engine, make_passage, make_query):
    q = make_query("q1", [])

    rf = QRQBDRRF(search_engine=None, top_n_qr=1)
    rf.reformulation_fn = lambda q: [q]
    rf.expansion_fn = lambda q, p: q.text

    hit = make_hit("d1", 1.0, "parsed")
    dummy_searcher = make_search_engine({"q1": [hit]})
    rf.search_engine = dummy_searcher

    rf.run([q])
    assert len(q.passages) == 1
    p = q.passages[0]
    assert p.id == "d1"
    assert p.text == "parsed"


def test_weighted_rrf(make_hit, make_search_engine, make_passage, make_query):
    existing = make_passage("p1", text="t1", score=1)
    q = make_query("q1", [existing])

    ws = QRQBDWeightedRRF(search_engine=None, rerank_budget=5, qr_weight=0.4, qbd_weight=0.6)
    ws.reformulation_fn = lambda q: "reformulated"
    ws.expansion_fn = lambda q, p: "expanded"

    qr_hits = [make_hit(f"d{i}", float(i), f"raw{i}") for i in range(1, 5)]
    qbd_hits = [make_hit(f"e{i}", float(i), f"rawE{i}") for i in range(1, 5)]
    results_map = {"q1": qr_hits, "q1:::p1": qbd_hits}
    ws.search_engine = make_search_engine(results_map)

    ws.run([q])
    ids = [p.id for p in q.passages]
    assert ids == ["p1", "d1", "e1", "e2", "e3"]
