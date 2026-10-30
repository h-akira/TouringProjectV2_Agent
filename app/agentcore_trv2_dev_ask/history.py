"""Conversation history per session: only the questions and the final answers.

Tool calls and their (large) results are dropped after each turn so the next
question does not resend them to the model.
"""
from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from typing import Any

Message = dict[str, Any]


def _has_tool_block(message: Message) -> bool:
    return any("toolUse" in block or "toolResult" in block for block in message.get("content", []))


def _text_only(message: Message) -> Message | None:
    blocks = [
        {"text": block["text"]}
        for block in message.get("content", [])
        if isinstance(block.get("text"), str) and block["text"].strip()
    ]
    return {"role": message["role"], "content": blocks} if blocks else None


def prune_history(messages: list[Message], max_messages: int) -> list[Message]:
    """Keep alternating user/assistant text messages, latest first, ending with an answer."""
    kept: list[Message] = []
    for message in messages:
        if _has_tool_block(message):
            continue
        text = _text_only(message)
        if text is None:
            continue
        if kept and kept[-1]["role"] == text["role"]:
            kept[-1] = text  # keep the later one so roles alternate
        else:
            kept.append(text)
    if kept and kept[-1]["role"] == "user":
        kept.pop()  # a question without an answer (failed turn)
    kept = kept[-max_messages:] if max_messages > 0 else []
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


class SessionStore:
    """In-process history per runtime session, bounded by least-recently-used eviction."""

    def __init__(self, max_sessions: int = 128) -> None:
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, list[Message]] = OrderedDict()

    def get(self, session_id: str) -> list[Message]:
        if session_id not in self._sessions:
            return []
        self._sessions.move_to_end(session_id)
        return copy.deepcopy(self._sessions[session_id])

    def put(self, session_id: str, messages: list[Message]) -> None:
        self._sessions[session_id] = copy.deepcopy(messages)
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)


class SessionLock:
    """A lock that counts the calls holding it or waiting for it, so that it is not dropped while one is on its way
    (asyncio.Lock.locked() is False between a release and the waiter waking)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.users = 0

    async def __aenter__(self) -> "SessionLock":
        self.users += 1
        try:
            await self._lock.acquire()
        except BaseException:
            self.users -= 1
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._lock.release()
        self.users -= 1


class SessionLocks:
    """One lock per runtime session, so that calls on one session go one by one: each starts from the history the one
    before it left (Agent #6, adr/0014). Bounded like the histories; a lock with a call holding or waiting for it is
    never dropped (Agent #14)."""

    def __init__(self, max_locks: int = 256) -> None:
        self._max_locks = max_locks
        self._locks: OrderedDict[str, SessionLock] = OrderedDict()

    def get(self, session_id: str) -> SessionLock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = SessionLock()
        self._locks.move_to_end(session_id)
        for old_id in list(self._locks):
            if len(self._locks) <= self._max_locks:
                break
            if self._locks[old_id].users == 0 and old_id != session_id:
                del self._locks[old_id]
        return lock
