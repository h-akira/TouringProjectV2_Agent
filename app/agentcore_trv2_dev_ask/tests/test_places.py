"""Tests for the place tools: geometry, payload handling and failure paths."""
import math
from types import SimpleNamespace

import pytest

from tools import places

TOKYO_STATION = {"latitude": 35.6812, "longitude": 139.7671}


def _context(location=None) -> SimpleNamespace:
    state = {} if location is None else {places.LOCATION_KEY: location}
    return SimpleNamespace(invocation_state=state)


def _place(title: str, lat: float, lon: float, distance: int = 0) -> dict:
    return {"Title": title, "Position": [lon, lat], "Distance": distance}


class _Client:
    def __init__(self, items=None, error=None):
        self.items = items or []
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def _reply(self, name: str, kwargs: dict) -> dict:
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return {"ResultItems": self.items}

    def search_nearby(self, **kwargs):
        return self._reply("search_nearby", kwargs)

    def search_text(self, **kwargs):
        return self._reply("search_text", kwargs)


@pytest.fixture
def client(monkeypatch):
    stub = _Client()
    monkeypatch.setattr(places, "_client", stub)
    return stub


@pytest.mark.parametrize(
    "bearing, heading, expected",
    [(0, 0, "前方"), (90, 0, "右手"), (180, 0, "後方"), (270, 0, "左手"), (45, 0, "右前方"), (10, 350, "前方"), (0, 90, "左手")],
)
def test_side_of(bearing, heading, expected):
    assert places.side_of(bearing, heading) == expected


def test_bearing_points_the_right_way():
    assert places.bearing_degrees(35.0, 139.0, 36.0, 139.0) == pytest.approx(0, abs=0.5)
    assert places.bearing_degrees(35.0, 139.0, 35.0, 140.0) == pytest.approx(90, abs=1)


@pytest.mark.parametrize("bearing, expected", [(0, "北"), (90, "東"), (225, "南西"), (359, "北")])
def test_compass(bearing, expected):
    assert places.compass(bearing) == expected


@pytest.mark.parametrize("meters, expected", [(3, "約10m"), (240, "約240m"), (1234, "約1.2km")])
def test_format_distance(meters, expected):
    assert places.format_distance(meters) == expected


@pytest.mark.parametrize(
    "state",
    [
        None,
        {},
        {places.LOCATION_KEY: None},
        {places.LOCATION_KEY: {"latitude": "35", "longitude": 139.0}},
        {places.LOCATION_KEY: {"latitude": True, "longitude": 139.0}},
        {places.LOCATION_KEY: {"latitude": math.nan, "longitude": 139.0}},
        {places.LOCATION_KEY: {"latitude": 95.0, "longitude": 139.0}},
    ],
)
def test_unusable_location_is_none(state):
    assert places.read_location(state) is None


def test_heading_is_optional_and_validated():
    loc = places.read_location({places.LOCATION_KEY: {**TOKYO_STATION, "headingDegrees": "north"}})
    assert loc == {"latitude": 35.6812, "longitude": 139.7671, "heading": None}
    loc = places.read_location({places.LOCATION_KEY: {**TOKYO_STATION, "headingDegrees": 90}})
    assert loc["heading"] == 90.0


def test_nearby_uses_payload_position_longitude_first(client):
    places.nearby_places(kind="公園", tool_context=_context(TOKYO_STATION))
    name, kwargs = client.calls[0]
    assert name == "search_nearby"
    assert kwargs["QueryPosition"] == [139.7671, 35.6812]
    assert kwargs["Filter"] == {"IncludeCategories": ["park-recreation_area", "garden"]}
    assert kwargs["Language"] == "ja"


def test_nearby_states_side_direction_and_distance(client):
    client.items = [_place("東の公園", 35.6812, 139.7771)]
    answer = places.nearby_places(kind="公園", tool_context=_context({**TOKYO_STATION, "headingDegrees": 0}))
    assert "東の公園（右手・東の方角 約900m）" in answer


def test_side_filter_keeps_only_that_side(client):
    client.items = [_place("東", 35.6812, 139.7771), _place("西", 35.6812, 139.7571)]
    answer = places.nearby_places(kind="公園", side="右手", tool_context=_context({**TOKYO_STATION, "headingDegrees": 0}))
    assert "東" in answer and "西（" not in answer


def test_side_filter_without_heading_says_why(client):
    client.items = [_place("東", 35.6812, 139.7771)]
    answer = places.nearby_places(kind="公園", side="右手", tool_context=_context(TOKYO_STATION))
    assert answer == "進行方向が分からないため、左右では絞り込めませんでした。"


def test_no_side_without_heading(client):
    client.items = [_place("東", 35.6812, 139.7771)]
    answer = places.nearby_places(kind="公園", tool_context=_context(TOKYO_STATION))
    assert "東（東の方角 約900m）" in answer
    assert "左右は付けていません" in answer


def test_duplicates_listed_once_and_capped(client):
    client.items = [_place("同じ", 35.6813, 139.7672)] * 2 + [_place(f"店{i}", 35.68 + i / 1000, 139.767) for i in range(10)]
    answer = places.nearby_places(kind="公園", tool_context=_context(TOKYO_STATION))
    assert answer.count("同じ（") == 1
    assert answer.count("\n- ") == places.MAX_RETURNED


def test_radius_is_clamped(client):
    places.nearby_places(kind="山・自然", radius_km=500, tool_context=_context(TOKYO_STATION))
    assert client.calls[0][1]["QueryRadius"] == int(places.MAX_RADIUS_KM * 1000)


def test_find_place_is_biased_to_rider_and_sorted_by_distance(client):
    client.items = [_place("遠い", 35.70, 139.77, distance=2000), _place("近い", 35.682, 139.767, distance=100)]
    answer = places.find_place(name="マクドナルド", tool_context=_context(TOKYO_STATION))
    assert client.calls[0][1]["BiasPosition"] == [139.7671, 35.6812]
    assert answer.index("近い") < answer.index("遠い")


def test_no_location_means_no_search(client):
    assert places.find_place(name="マクドナルド", tool_context=_context()) == "現在地が分からないため、周辺を探せません。"
    assert client.calls == []


def test_aws_error_is_a_sentence(client):
    client.error = RuntimeError("boom")
    assert places.nearby_places(kind="公園", tool_context=_context(TOKYO_STATION)) == "周辺の検索に失敗しました。"


def test_nothing_found(client):
    assert places.find_place(name="存在しない店", tool_context=_context(TOKYO_STATION)) == "該当する場所は見つかりませんでした。"


def test_every_kind_has_categories():
    assert set(places.KINDS) == set(places.Kind.__args__)
    assert all(places.KINDS[kind] for kind in places.KINDS)
