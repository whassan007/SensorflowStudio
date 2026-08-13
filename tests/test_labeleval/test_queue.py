"""EventQueue abstraction tests."""

from sensorflow.evaluation.queue import InMemoryEventQueue, KafkaEventQueue, make_queue


def test_in_memory_publish_consume_ack():
    q = InMemoryEventQueue()
    for i in range(10):
        q.publish("evaluation", {"i": i})
    assert q.stats()["pending"] == 10

    batch = q.consume("evaluation", 4)
    assert [m["i"] for m in batch] == [0, 1, 2, 3]
    s = q.stats()
    assert s["pending"] == 6 and s["processing"] == 4

    q.ack("evaluation", 4)
    s = q.stats()
    assert s["processing"] == 0 and s["completed"] == 4 and s["failed"] == 0


def test_ack_failed_messages_counted():
    q = InMemoryEventQueue()
    q.publish("t", {"a": 1})
    q.consume("t", 1)
    q.ack("t", 0, failed=1)
    assert q.stats()["failed"] == 1


def test_topics_are_independent():
    q = InMemoryEventQueue()
    q.publish("a", {"x": 1})
    q.publish("b", {"x": 2})
    assert q.consume("a", 10)[0]["x"] == 1
    assert q.consume("b", 10)[0]["x"] == 2


def test_kafka_stub_raises_clear_error():
    try:
        KafkaEventQueue()
        assert False, "stub should not construct"
    except RuntimeError as e:
        assert "Kafka" in str(e)


def test_make_queue_always_returns_working_queue():
    q = make_queue("memory")
    q.publish("t", {"ok": True})
    assert q.consume("t", 1)[0]["ok"] is True
    # kafka preference degrades gracefully to a working queue
    q2 = make_queue("kafka")
    q2.publish("t", {"ok": 1})
    assert q2.consume("t", 1)
