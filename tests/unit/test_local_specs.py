import helpers.local_specs as ls


def test_bytes_to_gb_basic():
    one_gb = 1024 ** 3
    assert ls._bytes_to_gb(one_gb) == 1.0
    assert ls._bytes_to_gb(0) == 0.0


def test_gather_local_specs_shape():
    data = ls.gather_local_specs()

    assert isinstance(data, dict)
    for key in ["os", "python", "cpu_cores", "gpus", "capability", "red_flags"]:
        assert key in data

    assert isinstance(data["cpu_cores"], int)
    assert isinstance(data["gpus"], list)
    assert isinstance(data["red_flags"], list)
