from types import SimpleNamespace
from typing import Any

import pytest

from program.apis.mdblist_api import (
    MdblistAPI,
    MdblistFetchResult,
    MdblistListItem,
)
from program.media.item import MediaItem
from program.services.content import mdblist as mdblist_module
from program.services.content.mdblist import Mdblist


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self.payload = payload
        self.headers = headers or {}
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_url_list_keeps_items_with_null_year_and_skips_only_invalid_entries() -> None:
    api = MdblistAPI("test-api-key")
    session = FakeSession(
        [
            FakeResponse(
                {
                    "movies": [
                        {
                            "id": 101,
                            "imdb_id": "tt0000101",
                            "mediatype": "movie",
                            "release_year": None,
                        },
                        {
                            "id": 102,
                            "mediatype": "episode",
                            "release_year": 2026,
                        },
                    ],
                    "shows": [
                        {
                            "id": 103,
                            "tvdb_id": 203,
                            "mediatype": "show",
                            "release_year": 2026,
                        }
                    ],
                    "pagination": {"has_more": False, "next_cursor": None},
                }
            )
        ]
    )
    api.session = session

    result = api.list_items_by_url("https://mdblist.com/lists/example/large")

    assert [item.mediatype for item in result.items] == ["movie", "show"]
    assert result.items[0].release_year is None
    assert result.items[1].tvdb_id == 203
    assert result.invalid_items == 1
    assert session.calls[0][0] == "/lists/example/large/items"


def test_id_list_follows_cursor_pagination_until_complete() -> None:
    api = MdblistAPI("test-api-key")
    session = FakeSession(
        [
            FakeResponse(
                {
                    "movies": [{"id": 101, "mediatype": "movie"}],
                    "shows": [],
                    "pagination": {"has_more": True, "next_cursor": "next-page"},
                }
            ),
            FakeResponse(
                {
                    "movies": [],
                    "shows": [{"id": 102, "tvdb_id": 202, "mediatype": "show"}],
                    "pagination": {"has_more": False, "next_cursor": None},
                }
            ),
        ]
    )
    api.session = session

    result = api.list_items_by_id(42)

    assert [(item.mediatype, item.id) for item in result.items] == [
        ("movie", 101),
        ("show", 102),
    ]
    assert result.pages == 2
    assert session.calls[0][1]["params"]["limit"] == 1000
    assert "cursor" not in session.calls[0][1]["params"]
    assert session.calls[1][1]["params"]["cursor"] == "next-page"


def test_import_continues_after_failed_list_and_deduplicates_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class APIStub:
        def list_items_by_url(self, url: str) -> MdblistFetchResult:
            if url.endswith("/failed"):
                raise RuntimeError("temporary response failure")

            if url.endswith("/first"):
                return MdblistFetchResult(
                    items=[
                        MdblistListItem(
                            id=101,
                            mediatype="movie",
                            release_year=None,
                        ),
                        MdblistListItem(
                            id=102,
                            tvdb_id=202,
                            mediatype="show",
                        ),
                        MdblistListItem(id=None, mediatype="movie"),
                    ]
                )

            return MdblistFetchResult(
                items=[
                    MdblistListItem(id=101, mediatype="movie"),
                    MdblistListItem(id=303, mediatype="movie"),
                ]
            )

    monkeypatch.setattr(
        mdblist_module,
        "item_exists_by_any_id",
        lambda **_kwargs: False,
    )

    service = Mdblist.__new__(Mdblist)
    service.key = "mdblist"
    service.api = APIStub()
    service.settings = SimpleNamespace(
        lists=[
            "https://mdblist.com/lists/example/failed",
            "https://mdblist.com/lists/example/first",
            "https://mdblist.com/lists/example/second",
        ]
    )

    result = next(service.run(MediaItem({})))

    assert [(item.tmdb_id, item.tvdb_id) for item in result.media_items] == [
        (101, None),
        (None, 202),
        (303, None),
    ]
