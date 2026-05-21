from starpower_core.memory import RemMemoryGraph


def test_memory_remember_recall_and_audit() -> None:
    memory = RemMemoryGraph()
    record = memory.remember("Massive Magnetics is the body layer", tags=["body", "massive"])
    recalled = memory.recall("body layer")
    assert recalled[0].id == record.id
    assert memory.audit()["healthy"] is True


def test_memory_self_heals_corrupted_record() -> None:
    memory = RemMemoryGraph()
    record = memory.remember("B Heard is the voice layer", tags=["voice"])
    memory.records[record.id].text = "tampered"
    assert memory.audit()["healthy"] is False
    healed = memory.self_heal()
    assert healed["healthy"] is True
