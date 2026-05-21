from starpower_core.vsa import VSAEngine


def test_vsa_encode_is_deterministic() -> None:
    engine = VSAEngine(dimensions=512)
    left = engine.encode("bando")
    right = engine.encode("bando")
    assert engine.similarity(left, right) == 1.0


def test_vsa_bind_and_unbind_cleanup() -> None:
    engine = VSAEngine(dimensions=512)
    body = engine.encode("body")
    voice = engine.encode("voice")
    bound = engine.bind(body, voice)
    recovered = engine.unbind(bound, voice)
    symbol, score = engine.cleanup(recovered)
    assert symbol == "body"
    assert score > 0.9
