import json

import jsonlines

from larf.query_loaders.base import Query
from larf.query_loaders.cast import CAsT2022


def test_cast2022(tmp_path):
    conversations = [
        {
            "number": 1,
            "turn": [
                {"number": 1, "manual_rewritten_utterance": "query1"},
                {"number": 2, "manual_rewritten_utterance": "query2"},
            ],
        },
        {"number": 2, "turn": [{"number": 1, "manual_rewritten_utterance": "query3"}]},
    ]
    queries_file = tmp_path / "cast_queries.json"
    queries_file.write_text(json.dumps(conversations), encoding="utf-8")

    loader = CAsT2022(queries_path=queries_file)
    queries = loader.get_queries()
    assert len(queries) == 3
    expected_ids = ["1_1", "1_2", "2_1"]
    expected_texts = ["query1", "query2", "query3"]
    for q, exp_id, exp_text in zip(queries, expected_ids, expected_texts):
        assert isinstance(q, Query)
        assert q.id == exp_id
        assert q.text == exp_text
        assert q.passages == []


def test_cast2022_resume(tmp_path):
    # Fake topics
    conversations = [
        {"number": 1, "turn": [{"number": 1, "manual_rewritten_utterance": "query1"}]}
    ]
    queries_file = tmp_path / "cast_queries.json"
    queries_file.write_text(json.dumps(conversations), encoding="utf-8")

    # Resume directory.
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    qid = "1_1"
    record = {"id": "pid1", "text": "passage1", "score": 1.0, "relevance_assessment": False}
    jsonl_path = resume_dir / f"{qid}.jsonl"
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write(record)

    loader = CAsT2022(queries_path=queries_file, resume_path=resume_dir)
    queries = loader.get_queries()
    assert len(queries) == 1
    q = queries[0]

    assert hasattr(q, "passages")
    assert len(q.passages) == 1
    p = q.passages[0]
    assert p.id == record["id"]
    assert p.text == record["text"]
    assert p.score == record["score"]
    assert p.relevance_assessment == record["relevance_assessment"]
