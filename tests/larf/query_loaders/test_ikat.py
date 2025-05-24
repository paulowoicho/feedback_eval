import json

import jsonlines

from larf.query_loaders.base import Query
from larf.query_loaders.ikat import iKAT2023


def test_ikat2023(tmp_path):
    conversations = [
        {
            "number": 1,
            "turns": [
                {"turn_id": 1, "resolved_utterance": "qA"},
                {"turn_id": 2, "resolved_utterance": "qB"},
            ],
        },
        {"number": 2, "turns": [{"turn_id": 1, "resolved_utterance": "qC"}]},
    ]
    queries_file = tmp_path / "ikat_queries.json"
    queries_file.write_text(json.dumps(conversations), encoding="utf-8")

    loader = iKAT2023(queries_path=queries_file)
    queries = loader.get_queries()
    # Total queries = 3
    assert len(queries) == 3
    expected_ids = ["1_1", "1_2", "2_1"]
    expected_texts = ["qA", "qB", "qC"]
    for q, exp_id, exp_text in zip(queries, expected_ids, expected_texts):
        assert isinstance(q, Query)
        assert q.id == exp_id
        assert q.text == exp_text
        assert q.passages == []


def test_ikat2023_with_resume(tmp_path):
    # Prepare queries JSON
    conversations = [{"number": 3, "turns": [{"turn_id": 1, "resolved_utterance": "qD"}]}]
    queries_file = tmp_path / "ikat_queries.json"
    queries_file.write_text(json.dumps(conversations), encoding="utf-8")

    # Prepare resume directory and passages
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    qid = "3_1"
    record = {"id": "pidX", "text": "passX", "score": 2.0, "relevance_assessment": True}
    jsonl_path = resume_dir / f"{qid}.jsonl"
    with jsonlines.open(jsonl_path, mode="w") as writer:
        writer.write(record)

    loader = iKAT2023(queries_path=queries_file, resume_path=resume_dir)
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
