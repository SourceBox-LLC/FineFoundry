import helpers.datasets as ds


def test_guess_columns_prefers_standard_names():
    inn, out = ds.guess_input_output_columns(["input", "output"])
    assert inn == "input"
    assert out == "output"


def test_guess_columns_case_insensitive():
    names = ["Prompt", "Completion"]
    inn, out = ds.guess_input_output_columns(names)
    assert inn == "Prompt"
    assert out == "Completion"


def test_guess_columns_returns_none_for_unknown():
    inn, out = ds.guess_input_output_columns(["foo", "bar"])
    assert inn is None
    assert out is None
