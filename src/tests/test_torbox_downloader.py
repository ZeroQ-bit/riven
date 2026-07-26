"""Tests for the TorBox downloader.

The downloader's networked methods are tested by mocking ``TorBoxAPI.session``
and bypassing ``TorBoxDownloader.__init__`` (which calls ``validate()`` and
needs the full settings stack). This keeps the tests hermetic and runnable
without the full Riven runtime environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from program.services.downloaders.torbox import (
    TORBOX_READY_STATES,
    TorBoxCreateTorrentResponse,
    TorBoxDownloader,
    TorBoxEnvelope,
    TorBoxError,
    TorBoxFile,
    TorBoxRequestDownload,
    TorBoxTorrent,
    TorBoxUser,
)
from program.utils.request import CircuitBreakerOpen
from routers.secure.default import DownloaderUserInfo

TEST_DATA = Path(__file__).parent / "test_data"
UBUNTU_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def load_fixture(name: str) -> Any:
    return json.loads((TEST_DATA / name).read_text())


def make_downloader(session: MagicMock) -> TorBoxDownloader:
    """Build a TorBoxDownloader with a mocked session, skipping __init__."""
    dl = TorBoxDownloader.__new__(TorBoxDownloader)
    dl.key = "torbox"
    dl.api = MagicMock()
    dl.api.api_key = "test-api-key"
    dl.api.session = session
    return dl


def make_response(json_body: Any, status_code: int = 200, reason: str = "OK"):
    """Build a fake response object compatible with how the downloader reads it."""
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.reason = reason
    resp.data = json_body
    return resp


# --- Pydantic response models --------------------------------------------------


class TestTorBoxModels:
    def test_envelope_parses_success(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_user_me.json"))
        assert env.success is True
        assert env.data["id"] == 12345

    def test_envelope_tolerates_null_data(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_controltorrent.json"))
        assert env.success is True
        assert env.data is None

    def test_user_parses_premium(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_user_me.json"))
        user = TorBoxUser.model_validate(env.data)
        assert user.plan == 2  # Pro plan
        assert user.premium_expires_at == "2030-12-31T23:59:59"
        assert user.email == "zeroq@example.com"

    def test_user_parses_free(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_user_me_free.json"))
        user = TorBoxUser.model_validate(env.data)
        assert user.plan == 0
        assert user.premium_expires_at is None

    def test_torrent_parses(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_mylist_ready.json"))
        torrent = TorBoxTorrent.model_validate(env.data)
        assert torrent.id == 998877
        assert torrent.download_state == "cached"
        assert torrent.download_finished is True
        assert torrent.cached is True
        assert torrent.files is not None and len(torrent.files) == 1

    def test_torrent_handles_both_size_keys(self):
        # TorBox sometimes returns "bytes" instead of "size"
        torrent = TorBoxTorrent.model_validate(
            {"id": 1, "name": "x", "bytes": 5000, "download_state": "queued"}
        )
        assert torrent.bytes == 5000

    def test_file_size_falls_back_to_bytes_alias(self):
        f = TorBoxFile.model_validate({"id": 1, "name": "v.mkv", "bytes": 1234})
        assert f.get_size() == 1234

    def test_file_prefers_size_field(self):
        f = TorBoxFile.model_validate(
            {"id": 1, "name": "v.mkv", "size": 999, "bytes": 1234}
        )
        assert f.get_size() == 999

    def test_request_download_parses(self):
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_requestdl.json"))
        dl = TorBoxRequestDownload.model_validate(env.data)
        assert dl.url.startswith("https://")
        assert dl.filesize == 4000000000

    def test_create_torrent_receipt_parses(self):
        """Regression: createtorrent returns a receipt with torrent_id (not id),
        no name field. Parsing must not raise.
        """
        env = TorBoxEnvelope.model_validate(load_fixture("torbox_createtorrent.json"))
        created = TorBoxCreateTorrentResponse.model_validate(env.data)
        assert created.torrent_id == 998877
        assert created.hash == UBUNTU_HASH

    def test_create_torrent_real_shape_parses(self):
        """The real createtorrent response uses snake_case keys and an auth_id
        UUID field; ensure none of that breaks parsing.
        """
        env = TorBoxEnvelope.model_validate(
            {
                "success": True,
                "detail": "ok",
                "data": {
                    "torrent_id": 555,
                    "hash": "abc123",
                    "queued_id": None,
                    "auth_id": "d3b046a1-8fe2-6c8eca7abefb",
                    "active_limit": 10,
                    "current_active_downloads": 0,
                },
            }
        )
        created = TorBoxCreateTorrentResponse.model_validate(env.data)
        assert created.torrent_id == 555


# --- Static helpers ------------------------------------------------------------


class TestReadyStates:
    def test_cached_is_ready(self):
        info = MagicMock(status="cached")
        assert TorBoxDownloader._is_ready(info) is True

    def test_downloading_not_ready(self):
        info = MagicMock(status="downloading")
        assert TorBoxDownloader._is_ready(info) is False

    def test_case_insensitive(self):
        info = MagicMock(status="COMPLETED")
        assert TorBoxDownloader._is_ready(info) is True

    def test_unknown_state_not_ready(self):
        info = MagicMock(status="metaDL")
        assert TorBoxDownloader._is_ready(info) is False

    def test_ready_states_is_frozenset(self):
        assert isinstance(TORBOX_READY_STATES, frozenset)
        assert "cached" in TORBOX_READY_STATES


class TestParseLink:
    def test_colon_pair(self):
        assert TorBoxDownloader._parse_link("998877:1") == (998877, 1)

    def test_full_url(self):
        url = "https://api.torbox.app/v1/api/torrents/requestdl?token=k&torrent_id=998877&file_id=1"
        assert TorBoxDownloader._parse_link(url) == (998877, 1)

    def test_url_missing_file_id_defaults_zero(self):
        url = "https://api.torbox.app/v1/api/torrents/requestdl?token=k&torrent_id=5"
        assert TorBoxDownloader._parse_link(url) == (5, 0)


# --- Networked methods (session mocked) ---------------------------------------


class TestGetUserInfo:
    def test_premium_user(self):
        session = MagicMock()
        session.get.return_value = make_response(load_fixture("torbox_user_me.json"))
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.service == "torbox"
        assert info.premium_status == "premium"
        assert info.user_id == 12345
        assert info.premium_days_left is not None
        # TorBox exposes no username; email is reused for the username field.
        assert info.username == "zeroq@example.com"
        assert info.email == "zeroq@example.com"

    def test_free_user(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("torbox_user_me_free.json")
        )
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.premium_status == "free"

    def test_returns_none_on_http_error(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"success": False, "detail": "BAD_TOKEN"}, status_code=401
        )
        dl = make_downloader(session)

        assert dl.get_user_info() is None

    def test_real_api_shape_parses_without_error(self):
        """Regression: the real TorBox /user/me response uses an ISO date string
        for premium_expires_at (not a unix timestamp) and has no username field.
        This must not raise — previously it caused get_user_info() to return
        None and the dashboard to 500.
        """
        session = MagicMock()
        session.get.return_value = make_response(
            {
                "success": True,
                "detail": "",
                "data": {
                    "id": 42,
                    "email": "real@example.com",
                    "plan": 1,
                    "premium_expires_at": "2027-06-01T00:00:00",
                    "user_referral": "abc",
                },
            }
        )
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.user_id == 42
        assert info.premium_status == "premium"
        assert info.premium_expires_at is not None
        assert info.premium_days_left is not None


class TestIsCached:
    def test_cache_hit(self):
        session = MagicMock()
        session.post.return_value = make_response(
            load_fixture("torbox_checkcached_hit.json")
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is True
        # Verify the hash was lower-cased in the request body
        _, kwargs = session.post.call_args
        assert kwargs["json"]["hashes"] == [UBUNTU_HASH]

    def test_cache_miss(self):
        session = MagicMock()
        session.post.return_value = make_response(
            load_fixture("torbox_checkcached_miss.json")
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is False

    def test_http_error_raises_circuit_breaker(self):
        # 5xx responses trip the per-domain circuit breaker (mirrors RealDebrid).
        session = MagicMock()
        session.post.return_value = make_response(
            {"success": False, "detail": "error"}, status_code=500
        )
        dl = make_downloader(session)

        with pytest.raises(CircuitBreakerOpen):
            dl._is_cached(UBUNTU_HASH)

    def test_client_error_returns_false(self):
        # 4xx (non-429) failures are not circuit-breaker events; _is_cached
        # treats them as "not cached" rather than raising.
        session = MagicMock()
        session.post.return_value = make_response(
            {"success": False, "detail": "bad request"}, status_code=400
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is False


class TestAddTorrent:
    def test_success_returns_id(self):
        session = MagicMock()
        session.post.return_value = make_response(
            load_fixture("torbox_createtorrent.json")
        )
        dl = make_downloader(session)

        torrent_id = dl.add_torrent(UBUNTU_HASH)

        assert torrent_id == 998877
        # Verify magnet was built from the lower-cased hash
        _, kwargs = session.post.call_args
        assert kwargs["data"]["magnet"] == f"magnet:?xt=urn:btih:{UBUNTU_HASH}"

    def test_duplicate_resolves_existing_via_mylist(self):
        session = MagicMock()
        # createtorrent returns 400 duplicate, mylist lookup returns the id
        session.post.side_effect = [
            make_response(
                {"success": False, "detail": "Torrent is a duplicate."},
                status_code=400,
            ),
            # controltorrent would be next if we deleted; but duplicate path shouldn't delete
        ]
        session.get.return_value = make_response(
            {
                "success": True,
                "detail": "",
                "data": [
                    {
                        "id": 555555,
                        "hash": UBUNTU_HASH,
                        "name": "Ubuntu 24.04 LTS",
                    }
                ],
            }
        )
        dl = make_downloader(session)

        torrent_id = dl.add_torrent(UBUNTU_HASH)

        assert torrent_id == 555555

    def test_api_error_raises(self):
        # 4xx (non-429, non-duplicate) surfaces as TorBoxError.
        session = MagicMock()
        session.post.return_value = make_response(
            {"success": False, "detail": "bad magnet"}, status_code=400
        )
        dl = make_downloader(session)

        with pytest.raises(TorBoxError):
            dl.add_torrent(UBUNTU_HASH)

    def test_server_error_raises_circuit_breaker(self):
        # 5xx trips the circuit breaker before the TorBoxError path.
        session = MagicMock()
        session.post.return_value = make_response(
            {"success": False, "detail": "server error"}, status_code=503
        )
        dl = make_downloader(session)

        with pytest.raises(CircuitBreakerOpen):
            dl.add_torrent(UBUNTU_HASH)


class TestGetTorrentInfo:
    def test_ready_torrent_normalizes(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("torbox_mylist_ready.json")
        )
        dl = make_downloader(session)

        info = dl.get_torrent_info(998877)

        assert info.id == 998877
        assert info.name == "Ubuntu 24.04 LTS"
        assert info.status == "cached"
        assert info.infohash == UBUNTU_HASH
        assert info.bytes == 4000000000
        assert 1 in info.files
        assert info.files[1].filename == "ubuntu-24.04.iso"
        assert info.files[1].bytes == 4000000000

    def test_list_payload_unwrapped(self):
        session = MagicMock()
        # Some mylist responses wrap the single item in a list
        body = load_fixture("torbox_mylist_ready.json")
        body["data"] = [body["data"]]
        session.get.return_value = make_response(body)
        dl = make_downloader(session)

        info = dl.get_torrent_info(998877)
        assert info.id == 998877

    def test_api_error_raises(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"success": False, "detail": "not found"}, status_code=404
        )
        dl = make_downloader(session)

        with pytest.raises(TorBoxError):
            dl.get_torrent_info(999)

    def test_real_mylist_shape_parses(self):
        """Regression: the real /torrents/mylist response returns created_at as
        an ISO 8601 string (e.g. '2026-07-26T02:33:36Z'), not a unix timestamp.
        Previously this raised an int_parsing ValidationError, every get_torrent_info
        call failed, and all streams got blacklisted.
        """
        session = MagicMock()
        session.get.return_value = make_response(
            {
                "success": True,
                "detail": "",
                "data": {
                    "id": 1234,
                    "name": "Some.Movie.2026.1080p",
                    "hash": "abc123",
                    "size": 2000000000,
                    "download_state": "cached",
                    "download_finished": True,
                    "created_at": "2026-07-26T02:33:36Z",
                    "expires_at": "2026-08-26T02:33:36Z",
                    "progress": 100,
                    "auth_id": "d3b046a1-8fe2-6c8eca7abefb",
                    "server": 1,
                    "peers": 0,
                    "seeds": 0,
                    "active": True,
                    "files": [
                        {"id": 1, "name": "Some.Movie.2026.1080p.mkv", "size": 2000000000}
                    ],
                },
            }
        )
        dl = make_downloader(session)

        info = dl.get_torrent_info(1234)

        assert info.id == 1234
        assert info.status == "cached"
        assert info.created_at is not None
        assert info.progress == 100


class TestDeleteTorrent:
    def test_success(self):
        session = MagicMock()
        session.post.return_value = make_response(
            load_fixture("torbox_controltorrent.json")
        )
        dl = make_downloader(session)

        # Should not raise
        dl.delete_torrent(998877)

        _, kwargs = session.post.call_args
        assert kwargs["json"]["operation"] == "Delete"
        assert kwargs["json"]["torrent_id"] == 998877

    def test_api_error_raises(self):
        session = MagicMock()
        session.post.return_value = make_response(
            {"success": False, "detail": "nope"}, status_code=400
        )
        dl = make_downloader(session)

        with pytest.raises(TorBoxError):
            dl.delete_torrent(998877)


class TestSelectFiles:
    def test_is_noop(self):
        # TorBox auto-selects; select_files must not touch the session.
        dl = make_downloader(MagicMock())
        dl.select_files(998877, [1, 2])  # should not raise
        dl.api.session.post.assert_not_called()


class TestUnrestrictLink:
    def test_returns_unrestricted_link(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("torbox_requestdl.json")
        )
        dl = make_downloader(session)

        link = dl.unrestrict_link("998877:1")

        assert link is not None
        assert link.download.startswith("https://")
        assert link.filename == "ubuntu-24.04.iso"
        assert link.filesize == 4000000000

        # token query param must be present (spec requirement)
        _, kwargs = session.get.call_args
        assert kwargs["params"]["token"] == "test-api-key"
        assert kwargs["params"]["torrent_id"] == 998877
        assert kwargs["params"]["file_id"] == 1

    def test_returns_none_on_http_error(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"success": False, "detail": "unavailable"}, status_code=502
        )
        dl = make_downloader(session)

        assert dl.unrestrict_link("998877:1") is None


class TestRouterSerialization:
    """Regression: /downloader_user_info builds DownloaderUserInfo(service=...).

    The router model has its own service Literal that must include "torbox" or
    the dashboard returns 500 the moment TorBox is initialized. See the
    companion fix in src/routers/secure/default.py.
    """

    def test_downloader_user_info_accepts_torbox(self):
        # Must not raise ValidationError.
        info = DownloaderUserInfo(
            service="torbox",
            username="zeroq",
            user_id=12345,
            premium_status="premium",
        )
        assert info.service == "torbox"

    def test_downloader_user_info_accepts_other_services(self):
        # Ensure the fix didn't drop the existing services.
        for svc in ("realdebrid", "alldebrid", "debridlink"):
            info = DownloaderUserInfo(
                service=svc, user_id=1, premium_status="premium"
            )
            assert info.service == svc
