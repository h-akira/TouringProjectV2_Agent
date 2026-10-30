"""Places around the rider, via Amazon Location Service (geo-places).

The geometry (distance, compass direction, left/right) is computed here so the
model never works it out itself, and the position comes from the payload via
invocation_state rather than from the model's arguments.
A failed lookup returns a sentence, not an exception, so the model can still answer.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Literal, Optional

import boto3
from strands import ToolContext, tool

PLACES_REGION = os.environ.get("PLACES_REGION", "us-east-1")
LOCATION_KEY = "location"

Kind = Literal[
    "公園", "山・自然", "観光地", "景色のよい場所", "神社・寺", "温泉", "ガソリンスタンド", "EV充電",
    "コンビニ", "道の駅・休憩所", "トイレ", "駐車場", "カフェ", "飲食店", "病院",
]
Side = Literal["右手", "左手", "前方", "後方"]

# Amazon Location category ids (case-sensitive ids, not display names).
KINDS: dict[str, list[str]] = {
    "公園": ["park-recreation_area", "garden"],
    "山・自然": ["natural_and_geographical", "mountain_or_hill"],
    "観光地": ["tourist_attraction"],
    "景色のよい場所": ["scenic_point"],
    "神社・寺": ["shrine", "temple"],
    "温泉": ["hot_spring"],
    "ガソリンスタンド": ["petrol-gasoline_station"],
    "EV充電": ["ev_charging_station"],
    "コンビニ": ["convenience_store"],
    "道の駅・休憩所": ["roadside_station", "rest_area", "parking_and_restroom_only_rest_area"],
    "トイレ": ["public_restroom-toilets"],
    "駐車場": ["parking"],
    "カフェ": ["coffee_shop"],
    "飲食店": ["restaurant", "casual_dining", "family_restaurant", "fast_food"],
    "病院": ["hospital"],
}

# Eight 45-degree sectors clockwise from straight ahead.
_SECTORS = ("前方", "右前方", "右手", "右後方", "後方", "左後方", "左手", "左前方")
# Diagonals belong to both neighbouring sides.
_SIDE_SECTORS: dict[str, set[str]] = {
    "右手": {"右前方", "右手", "右後方"},
    "左手": {"左前方", "左手", "左後方"},
    "前方": {"左前方", "前方", "右前方"},
    "後方": {"左後方", "後方", "右後方"},
}
_COMPASS_POINTS = (
    "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
    "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西",
)

MAX_RETURNED = 5
MAX_FETCHED = 20
MIN_RADIUS_KM = 0.2
MAX_RADIUS_KM = 50.0
DEFAULT_RADIUS_KM = 5.0

_NO_LOCATION = "現在地が分からないため、周辺を探せません。"
_FAILED = "周辺の検索に失敗しました。"

log = logging.getLogger("trv2.agent")

_client: Any = None
_client_lock = threading.Lock()


def _get_client() -> Any:
    """Build the client lazily so importing this module needs no credentials. Tools run in threads, and a boto3
    client is not safe to build from two threads at once, so the first build is taken one at a time."""
    global _client
    with _client_lock:
        if _client is None:
            _client = boto3.client("geo-places", region_name=PLACES_REGION)
        return _client


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def compass(bearing: float) -> str:
    """16-point compass name for a bearing."""
    return _COMPASS_POINTS[int((bearing % 360) / 22.5 + 0.5) % 16]


def side_of(bearing: float, heading: float) -> str:
    """Sector of a bearing for a rider travelling along heading."""
    relative = (bearing - heading) % 360
    return _SECTORS[int(relative / 45 + 0.5) % 8]


def format_distance(meters: float) -> str:
    """Distance as it should be read aloud."""
    if meters < 1000:
        return f"約{max(10, round(meters, -1)):.0f}m"
    return f"約{meters / 1000:.1f}km"


def _distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(a))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def read_location(invocation_state: Optional[dict]) -> Optional[dict]:
    """The rider's position and heading from the payload, or None if unusable."""
    location = (invocation_state or {}).get(LOCATION_KEY)
    if not isinstance(location, dict):
        return None
    lat, lon = _number(location.get("latitude")), _number(location.get("longitude"))
    if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return None
    return {"latitude": lat, "longitude": lon, "heading": _number(location.get("headingDegrees"))}


def describe_places(items: list[dict], location: dict, side: Optional[str] = None) -> list[str]:
    """Lines the model can read out as fact; distance and side come from each Position."""
    lat, lon, heading = location["latitude"], location["longitude"], location["heading"]
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        title, position = item.get("Title"), item.get("Position")
        if not title or not isinstance(position, list) or len(position) != 2 or title in seen:
            continue
        place_lon, place_lat = position
        bearing = bearing_degrees(lat, lon, place_lat, place_lon)
        place_side = side_of(bearing, heading) if heading is not None else None
        if side is not None and place_side not in _SIDE_SECTORS[side]:
            continue
        seen.add(title)
        where = f"{compass(bearing)}の方角 {format_distance(_distance_meters(lat, lon, place_lat, place_lon))}"
        if place_side is not None:
            where = f"{place_side}・{where}"
        lines.append(f"{title}（{where}）")
        if len(lines) >= MAX_RETURNED:
            break
    return lines


def _answer(lines: list[str], location: dict, side: Optional[str]) -> str:
    if not lines:
        if side is not None and location["heading"] is None:
            return "進行方向が分からないため、左右では絞り込めませんでした。"
        return "該当する場所は見つかりませんでした。"
    header = "現在地から近い順（距離と方角は計算済み。言い換えずにそのまま使うこと）:"
    if location["heading"] is None:
        header += "\n進行方向が分からないため、左右は付けていません。"
    return header + "\n" + "\n".join(f"- {line}" for line in lines)


@tool(context=True)
def nearby_places(kind: Kind, tool_context: ToolContext, side: Optional[Side] = None,
                  radius_km: float = DEFAULT_RADIUS_KM) -> str:
    """現在地の周辺にある場所を種類で探し、近い順に距離・方角・左右を返す。

    「右手に見える山は？」「近くのコンビニは？」のように種類で探すときに使う。
    店やチェーンの名前は find_place を使う。現在地はシステムが渡すので、座標は指定しない。

    Args:
        kind: 探す場所の種類。
        side: 「右手に見える」「左手の」のように向きを言われたときだけ指定する。
        radius_km: 探す半径（km）。目に見えるものなら3〜5、道の駅やガソリンスタンドなら20程度、山なら15程度。
    """
    location = read_location(tool_context.invocation_state)
    if location is None:
        return _NO_LOCATION
    categories = KINDS.get(kind)
    if not categories:
        return f"「{kind}」という種類では探せません。"
    # A radius that is not a finite number falls back to the default instead of raising (Agent #5).
    radius = _number(radius_km)
    radius_m = int(min(max(radius if radius is not None else DEFAULT_RADIUS_KM, MIN_RADIUS_KM), MAX_RADIUS_KM) * 1000)
    try:
        result = _get_client().search_nearby(
            QueryPosition=[location["longitude"], location["latitude"]],
            QueryRadius=radius_m,
            MaxResults=MAX_FETCHED if side else MAX_RETURNED * 2,
            Language="ja",
            Filter={"IncludeCategories": categories},
        )
    except Exception as error:  # noqa: BLE001 - the model answers without it
        log.warning("search_nearby failed: %s", type(error).__name__)
        return _FAILED
    return _answer(describe_places(result.get("ResultItems") or [], location, side), location, side)


@tool(context=True)
def find_place(name: str, tool_context: ToolContext, side: Optional[Side] = None) -> str:
    """店・チェーン・施設・地名を名前で探し、現在地から近い順に距離・方角・左右を返す。

    「一番近いマクドナルドは？」「芦ノ湖までどのくらい？」のように名前が分かっているときに使う。
    現在地はシステムが渡すので、座標は指定しない。

    Args:
        name: 探す名前（例: マクドナルド、ENEOS、芦ノ湖）。地名や「近くの」は付けない。
        side: 「右手の」「左手の」のように向きを言われたときだけ指定する。
    """
    location = read_location(tool_context.invocation_state)
    if location is None:
        return _NO_LOCATION
    if not name.strip():
        return "探す名前が空です。"
    try:
        result = _get_client().search_text(
            QueryText=name.strip(),
            BiasPosition=[location["longitude"], location["latitude"]],
            MaxResults=MAX_FETCHED if side else MAX_RETURNED * 2,
            Language="ja",
        )
    except Exception as error:  # noqa: BLE001 - the model answers without it
        log.warning("search_text failed: %s", type(error).__name__)
        return _FAILED
    items = sorted(result.get("ResultItems") or [], key=lambda item: item.get("Distance", float("inf")))
    return _answer(describe_places(items, location, side), location, side)
