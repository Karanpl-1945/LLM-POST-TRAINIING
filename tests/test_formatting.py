from posttrainx.data.formatting import normalize_messages, prompt_chosen_rejected_to_conversational


def test_normalize_messages():
    x = {"dialog": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}
    out = normalize_messages(x, "dialog")
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][1]["role"] == "assistant"


def test_dpo_conversion():
    out = prompt_chosen_rejected_to_conversational({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert out["prompt"][0]["role"] == "user"
    assert out["chosen"][0]["role"] == "assistant"
