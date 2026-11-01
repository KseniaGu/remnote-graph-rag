def make_case_payload(**overrides):
    payload = {
        "schema_version": "retrieval-benchmark-v1",
        "id": "case_1",
        "review_status": "candidate_needs_review",
        "mode": "analyst",
        "queries": ["query"],
        "seeded_from": [],
        "expected_evidence": {
            "source_chunk_ids": {
                "required_all": ["chunk_1"],
                "required_any": ["chunk_2"],
            },
            "concept_labels": {"required_any": ["CLIP"]},
            "required_answer_points": ["not scored"],
        },
        "forbidden_evidence": {
            "source_chunk_ids": [],
            "concept_ids": [],
            "concept_labels": [],
            "relations": [{"predicate": "MENTIONS"}],
        },
        "thresholds": {
            "evidence_chunk_recall": 0.5,
            "concept_recall": 1.0,
            "forbidden_evidence_count": 0,
        },
        "review_notes": None,
    }
    payload.update(overrides)
    return payload
