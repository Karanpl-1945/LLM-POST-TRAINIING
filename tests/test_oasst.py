from posttrainx.data.oasst_preferences import build_oasst1_pairs


def test_oasst_pair_builder():
    rows = [
        {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Q", "lang": "en", "rank": None},
        {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "good", "lang": "en", "rank": 0},
        {"message_id": "a2", "parent_id": "u1", "role": "assistant", "text": "bad", "lang": "en", "rank": 1},
    ]
    pairs = build_oasst1_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good"
    assert pairs[0]["rejected"] == "bad"
