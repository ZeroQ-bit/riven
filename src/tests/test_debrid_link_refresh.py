from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import trio
from kink import di

from program.services.filesystem.vfs.db import (
    GetEntryByOriginalFilenameResult,
    VFSDatabase,
)
from program.services.streaming.exceptions import (
    DebridServiceException,
    DebridServiceUnableToConnectException,
)
from program.services.streaming.media_stream import MediaStream
from program.utils.debrid_link_status import (
    is_streamable_http_url,
    should_refresh_download_url,
)


def _make_stream(
    client: httpx.AsyncClient,
    *,
    provider: str,
    initial_url: str,
) -> MediaStream:
    stream = object.__new__(MediaStream)
    stream.async_client = client
    stream.target_url = SimpleNamespace(value=initial_url)
    stream.file_metadata = SimpleNamespace(
        file_size=1,
        path="/mount/movie.mkv",
        original_filename="movie-source.mkv",
    )
    stream.session_statistics = SimpleNamespace(total_session_connections=0)
    stream.fh = 1
    stream.provider = provider
    stream.enable_tracing = False
    return stream


def test_refreshable_statuses_are_provider_scoped():
    assert should_refresh_download_url(400, "torbox")
    assert not should_refresh_download_url(400, "realdebrid")
    assert should_refresh_download_url(404, "realdebrid")
    assert should_refresh_download_url(410, "alldebrid")
    assert should_refresh_download_url(503, "torbox")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://cdn.example/movie", True),
        ("http://cdn.example/movie", True),
        ("torbox:998877:1", False),
        ("998877:1", False),
        ("https:cdn.example/movie", False),
        (None, False),
    ],
)
def test_streamable_http_url_requires_http_scheme(
    url: str | None,
    expected: bool,
):
    assert is_streamable_http_url(url) is expected


def test_refresh_download_url_rejects_internal_torbox_locator(monkeypatch):
    old_url = "https://old.invalid/movie"
    fake_vfs = MagicMock()
    fake_vfs.get_entry_by_original_filename.return_value = (
        GetEntryByOriginalFilenameResult(
            original_filename="movie-source.mkv",
            download_url="torbox:998877:1",
            unrestricted_url=None,
            provider="torbox",
            provider_download_id="998877",
            size=1,
            created=None,
            modified=None,
            entry_type="media",
        )
    )
    monkeypatch.setitem(di._services, VFSDatabase, fake_vfs)

    async def run() -> None:
        stream = _make_stream(
            MagicMock(),
            provider="torbox",
            initial_url=old_url,
        )

        assert not await stream._refresh_download_url()
        assert stream.target_url.value == old_url

    trio.run(run)


def test_refresh_download_url_accepts_fresh_https_url(monkeypatch):
    old_url = "https://old.invalid/movie"
    fresh_url = "https://fresh.invalid/movie"
    fake_vfs = MagicMock()
    fake_vfs.get_entry_by_original_filename.return_value = (
        GetEntryByOriginalFilenameResult(
            original_filename="movie-source.mkv",
            download_url="torbox:998877:1",
            unrestricted_url=fresh_url,
            provider="torbox",
            provider_download_id="998877",
            size=1,
            created=None,
            modified=None,
            entry_type="media",
        )
    )
    monkeypatch.setitem(di._services, VFSDatabase, fake_vfs)

    async def run() -> None:
        stream = _make_stream(
            MagicMock(),
            provider="torbox",
            initial_url=old_url,
        )

        assert await stream._refresh_download_url()
        assert stream.target_url.value == fresh_url

    trio.run(run)


def test_torbox_refresh_returns_fresh_url_without_validation_get(monkeypatch):
    fresh_url = "https://fresh.invalid/movie"
    service = MagicMock()
    service.key = "torbox"
    service.unrestrict_link.return_value = SimpleNamespace(download=fresh_url)
    downloader = SimpleNamespace(services={"torbox": service})
    session = MagicMock()
    entry = SimpleNamespace(
        provider="torbox",
        download_url="torbox:998877:1",
        unrestricted_url=None,
        original_filename="movie-source.mkv",
    )
    validate = MagicMock(side_effect=AssertionError("validation GET was used"))
    monkeypatch.setattr(
        "program.services.filesystem.vfs.db.DebridCDNUrl.validate",
        validate,
    )

    result = VFSDatabase(downloader=downloader).refresh_unrestricted_url(
        entry,
        session,
    )

    assert result == fresh_url
    assert entry.unrestricted_url == fresh_url
    session.merge.assert_called_once_with(entry)
    session.commit.assert_called_once()
    validate.assert_not_called()


def test_torbox_http_400_refreshes_once_and_retries():
    old_url = "https://old.invalid/movie"
    fresh_url = "https://fresh.invalid/movie"
    requests: list[str] = []
    refresh_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if str(request.url) == old_url:
            return httpx.Response(400, request=request)
        return httpx.Response(
            206,
            request=request,
            content=b"x",
            headers={"Content-Length": "1"},
        )

    async def run() -> None:
        nonlocal refresh_calls
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stream = _make_stream(
                client,
                provider="torbox",
                initial_url=old_url,
            )

            async def refresh() -> bool:
                nonlocal refresh_calls
                refresh_calls += 1
                stream.target_url.value = fresh_url
                return True

            stream._refresh_download_url = refresh

            async with stream.establish_connection(start=0, end=0) as response:
                assert response.status_code == 206

    trio.run(run)

    assert refresh_calls == 1
    assert requests == [old_url, fresh_url]


def test_torbox_http_400_refresh_is_bounded():
    old_url = "https://old.invalid/movie"
    fresh_url = "https://fresh.invalid/movie"
    requests: list[str] = []
    refresh_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(400, request=request)

    async def run() -> None:
        nonlocal refresh_calls
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stream = _make_stream(
                client,
                provider="torbox",
                initial_url=old_url,
            )

            async def refresh() -> bool:
                nonlocal refresh_calls
                refresh_calls += 1
                stream.target_url.value = fresh_url
                return True

            stream._refresh_download_url = refresh

            with pytest.raises(DebridServiceUnableToConnectException):
                async with stream.establish_connection(start=0, end=0):
                    pass

    trio.run(run)

    assert refresh_calls == 1
    assert requests == [old_url, fresh_url]


def test_non_torbox_http_400_does_not_refresh():
    url = "https://invalid.example/movie"
    refresh_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, request=request)

    async def run() -> None:
        nonlocal refresh_calls
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stream = _make_stream(
                client,
                provider="realdebrid",
                initial_url=url,
            )

            async def refresh() -> bool:
                nonlocal refresh_calls
                refresh_calls += 1
                return True

            stream._refresh_download_url = refresh

            with pytest.raises(DebridServiceException):
                async with stream.establish_connection(start=0, end=0):
                    pass

    trio.run(run)

    assert refresh_calls == 0
