from helpers import chatml


def test_pair_to_chatml_basic():
    conv = chatml.pair_to_chatml("in", "out")
    msgs = conv["messages"]

    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "in"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "out"


def test_pair_to_chatml_with_system():
    conv = chatml.pair_to_chatml("in", "out", add_system="sys")
    msgs = conv["messages"]

    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_pairs_to_chatml_filters_empty_pairs():
    pairs = [
        {"input": "a", "output": "b"},
        {"input": " ", "output": "x"},
        {"input": "c", "output": ""},
    ]

    out = chatml.pairs_to_chatml(pairs)

    assert len(out) == 1
    msgs = out[0]["messages"]
    assert msgs[-2]["content"] == "a"
    assert msgs[-1]["content"] == "b"
