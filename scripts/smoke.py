"""Invoke the deployed runtime with fixed questions and check the event stream.

Usage: AWS_PROFILE=touring python3 scripts/smoke.py <runtime-arn>
Positions are public landmarks only.
"""
from __future__ import annotations

import json
import re
import sys
import uuid

import boto3

TOKYO_STATION = {"latitude": 35.6812, "longitude": 139.7671, "headingDegrees": 0.0}
FACTS = "現在日時: 2026-10-16 12:00（日本時間）\n現在地: 東京都千代田区丸の内（緯度35.6812 経度139.7671）\n進行方位: 北（右手は東、左手は西）\n"
MARKDOWN = re.compile(r"[*#`]|^\s*[-・]\s", re.MULTILINE)


def invoke(client, arn: str, session_id: str, question: str, location: dict | None) -> list[dict]:
    payload = {"prompt": FACTS + "質問: " + question, "location": location}
    response = client.invoke_agent_runtime(
        agentRuntimeArn=arn, runtimeSessionId=session_id, qualifier="DEFAULT", payload=json.dumps(payload).encode()
    )
    events = []
    for raw in response["response"].iter_lines():
        line = raw.decode("utf-8").strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def check(name: str, events: list[dict], expect_tool: str | None) -> bool:
    done = next((e for e in events if e.get("type") == "done"), None)
    tools = [e["name"] for e in events if e.get("type") == "tool"]
    errors = [e for e in events if e.get("type") == "error"]
    ok = done is not None and not errors and bool(done["answer"]) and not MARKDOWN.search(done["answer"])
    if expect_tool is not None:
        ok = ok and any(expect_tool in tool for tool in tools)
    print(f"[{'OK' if ok else 'NG'}] {name}: tools={tools} usage={done and done['usage']} model={done and done['modelId']}")
    print(f"      answer={done and done['answer']} errors={errors}")
    return ok


def main() -> int:
    arn = sys.argv[1]
    client = boto3.client("bedrock-agentcore", region_name="us-east-1")
    session = f"smoke-{uuid.uuid4()}"
    results = [
        check("知識だけで答える", invoke(client, arn, session, "富士山の高さは？", TOKYO_STATION), None),
        check("周辺検索", invoke(client, arn, session, "近くのコンビニはどこ？", TOKYO_STATION), "nearby_places"),
        check("Web 検索", invoke(client, arn, session, "今日の東京の天気は？", TOKYO_STATION), "earch"),
        check("会話の続き", invoke(client, arn, session, "さっきのコンビニまでの距離をもう一度教えて", TOKYO_STATION), None),
        check("現在地なし", invoke(client, arn, f"smoke-{uuid.uuid4()}", "右手に見える公園は？", None), None),
    ]
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
