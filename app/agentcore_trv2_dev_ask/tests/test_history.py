"""Tests for history pruning and the per-session store."""
from history import SessionLocks, SessionStore, prune_history


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"text": text}]}


TOOL_TURN = [
    user("Q1"),
    {"role": "assistant", "content": [{"text": "調べます。"}, {"toolUse": {"toolUseId": "t1", "name": "web", "input": {}}}]},
    {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "長い検索結果"}]}}]},
    assistant("A1"),
]


def test_tool_turn_keeps_question_and_final_answer_only():
    assert prune_history(TOOL_TURN, 10) == [user("Q1"), assistant("A1")]


def test_non_text_blocks_are_dropped():
    message = {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "x"}}}, {"text": "A"}]}
    assert prune_history([user("Q"), message], 10) == [user("Q"), assistant("A")]


def test_window_keeps_latest_and_starts_with_user():
    messages = []
    for i in range(6):
        messages += [user(f"Q{i}"), assistant(f"A{i}")]
    pruned = prune_history(messages, 5)
    assert pruned[0] == user("Q4")
    assert pruned == [user("Q4"), assistant("A4"), user("Q5"), assistant("A5")]


def test_failed_turn_leaves_no_dangling_question():
    messages = [user("Q1"), assistant("A1"), user("Q2 failed"), user("Q3"), assistant("A3")]
    assert prune_history(messages, 10) == [user("Q1"), assistant("A1"), user("Q3"), assistant("A3")]


def test_trailing_question_without_answer_is_dropped():
    assert prune_history([user("Q1"), assistant("A1"), user("Q2")], 10) == [user("Q1"), assistant("A1")]


def test_empty_text_is_dropped():
    assert prune_history([user("Q"), assistant("  ")], 10) == []


def test_store_returns_copies():
    store = SessionStore()
    store.put("s", [user("Q"), assistant("A")])
    got = store.get("s")
    got.append(user("mutated"))
    assert store.get("s") == [user("Q"), assistant("A")]
    assert store.get("unknown") == []


def test_store_evicts_least_recently_used():
    store = SessionStore(max_sessions=2)
    store.put("a", [user("a")])
    store.put("b", [user("b")])
    store.get("a")
    store.put("c", [user("c")])
    assert store.get("b") == []
    assert store.get("a") == [user("a")]


def test_a_session_gets_the_same_lock_each_time():
    locks = SessionLocks()
    assert locks.get("s1") is locks.get("s1")
    assert locks.get("s1") is not locks.get("s2")


def test_an_unused_old_lock_is_dropped_beyond_the_bound():
    locks = SessionLocks(max_locks=2)
    locks.get("s1")
    locks.get("s2")
    locks.get("s3")
    assert "s1" not in locks._locks and set(locks._locks) == {"s2", "s3"}


def test_a_lock_that_is_held_or_waited_for_is_not_dropped():
    import asyncio

    async def scenario():
        locks = SessionLocks(max_locks=1)
        lock = locks.get("s1")
        order = []

        async def use(name, hold):
            async with lock:
                order.append(name)
                await asyncio.sleep(hold)

        first = asyncio.create_task(use("first", 0.05))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(use("second", 0.0))  # waits for the first
        await asyncio.sleep(0.01)
        locks.get("s2")
        locks.get("s3")  # far over the bound; s1 is held and has a waiter
        assert locks.get("s1") is lock
        await asyncio.gather(first, second)
        assert order == ["first", "second"]

    asyncio.run(scenario())


def test_a_lock_between_the_release_and_the_waiter_waking_is_not_dropped():
    # locked() is False for a moment after the release although a waiter is about to take it (Agent #14).
    import asyncio

    async def scenario():
        locks = SessionLocks(max_locks=1)
        lock = locks.get("s1")
        await lock.__aenter__()
        waiter = asyncio.create_task(lock.__aenter__())
        await asyncio.sleep(0)  # the waiter is now waiting
        await lock.__aexit__(None, None, None)  # released; the waiter has not run yet
        locks.get("s2")
        assert locks.get("s1") is lock
        await waiter
        await lock.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_a_waiter_that_is_cancelled_stops_counting():
    import asyncio

    async def scenario():
        locks = SessionLocks(max_locks=1)
        lock = locks.get("s1")
        await lock.__aenter__()
        waiter = asyncio.create_task(lock.__aenter__())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await lock.__aexit__(None, None, None)
        assert lock.users == 0
        locks.get("s2")
        assert "s1" not in locks._locks

    asyncio.run(scenario())
