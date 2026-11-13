import repvar


def test_public_names_are_importable():
    for name in repvar.__all__:
        assert hasattr(repvar, name), name
    assert repvar.__version__ == "0.1.0"


def test_top_level_api_round_trip():
    scores = repvar.simulate_binary(40, 3, mean=0.5, icc=0.4, seed=1)
    comp = repvar.variance_components(scores)
    design = repvar.mean_se(comp, 40, 3)
    assert design.se > 0
    assert 0 <= repvar.reliability(comp, 3) <= 1
