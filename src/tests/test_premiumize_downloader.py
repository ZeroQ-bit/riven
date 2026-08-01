"""Tests for the Premiumize downloader.

The downloader's networked methods are tested by mocking ``PremiumizeAPI.session``
and bypassing ``PremiumizeDownloader.__init__`` (which calls ``validate()`` and
needs the full settings stack). This keeps the tests hermetic and runnable
without the full Riven runtime environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from program.services.downloaders.premiumize import (
    PREMIUMIZE_READY_STATES,
    ROOT_FOLDER_MARKER,
    PremiumizeDownloader,
    PremiumizeError,
    PremiumizeTransfer,
)
from program.utils.request import CircuitBreakerOpen
from routers.secure.default import DownloaderUserInfo

TEST_DATA = Path(__file__).parent / "test_data"
UBUNTU_HASH = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


def load_fixture(name: str) -> Any:
    return json.loads((TEST_DATA / name).read_text())


def make_downloader(session: MagicMock) -> PremiumizeDownloader:
    """Build a PremiumizeDownloader with a mocked session, skipping __init__."""
    dl = PremiumizeDownloader.__new__(PremiumizeDownloader)
    dl.key = "premiumize"
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


class TestPremiumizeModels:
    def test_transfer_parses_multi_file(self):
        env = load_fixture("premiumize_transfer_list_ready.json")
        transfer = PremiumizeTransfer.model_validate(env["transfers"][0])
        assert transfer.id == "tr_abcdef123456"
        assert transfer.status == "finished"
        assert transfer.folder_id == "fld_show_001"
        assert transfer.file_id is None
        assert transfer.progress == 1.0

    def test_transfer_parses_single_file(self):
        transfer = PremiumizeTransfer.model_validate(
            {
                "id": "tr_x",
                "name": "Movie.mkv",
                "status": "finished",
                "file_id": "file_9",
                "folder_id": None,
            }
        )
        assert transfer.file_id == "file_9"
        assert transfer.folder_id is None


# --- Static helpers ------------------------------------------------------------


class TestReadyStates:
    def test_finished_is_ready(self):
        info = MagicMock(status="finished")
        assert PremiumizeDownloader._is_ready(info) is True

    def test_seeding_is_ready(self):
        info = MagicMock(status="seeding")
        assert PremiumizeDownloader._is_ready(info) is True

    def test_running_not_ready(self):
        info = MagicMock(status="running")
        assert PremiumizeDownloader._is_ready(info) is False

    def test_case_insensitive(self):
        info = MagicMock(status="FINISHED")
        assert PremiumizeDownloader._is_ready(info) is True

    def test_ready_states_is_frozenset(self):
        assert isinstance(PREMIUMIZE_READY_STATES, frozenset)
        assert "finished" in PREMIUMIZE_READY_STATES
        assert "seeding" in PREMIUMIZE_READY_STATES


class TestParseLink:
    def test_synthetic_format_with_folder(self):
        assert PremiumizeDownloader._parse_link(
            "premiumize:fld_show_001:4f2a01"
        ) == ("fld_show_001", "4f2a01")

    def test_synthetic_format_root(self):
        assert PremiumizeDownloader._parse_link(
            f"premiumize:{ROOT_FOLDER_MARKER}:file_9"
        ) == (ROOT_FOLDER_MARKER, "file_9")

    def test_compact_format(self):
        assert PremiumizeDownloader._parse_link("fld_show_001:4f2a01") == (
            "fld_show_001",
            "4f2a01",
        )

    def test_bare_file_id_assumes_root(self):
        assert PremiumizeDownloader._parse_link("4f2a01") == (
            ROOT_FOLDER_MARKER,
            "4f2a01",
        )


class TestFileIdToInt:
    def test_hex_id_is_stable_and_positive(self):
        a = PremiumizeDownloader._file_id_to_int("4f2a01")
        b = PremiumizeDownloader._file_id_to_int("4f2a01")
        assert a == b
        assert a > 0

    def test_distinct_hex_ids_differ(self):
        assert PremiumizeDownloader._file_id_to_int("4f2a01") != (
            PremiumizeDownloader._file_id_to_int("4f2a02")
        )


# --- Networked methods (session mocked) ---------------------------------------


class TestGetUserInfo:
    def test_premium_user(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_account_info.json")
        )
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.service == "premiumize"
        assert info.premium_status == "premium"
        assert info.user_id == 7654321
        assert info.premium_expires_at is not None
        assert info.premium_days_left is not None
        # No username exposed; customer_id is reused for the username field.
        assert info.username == "7654321"
        assert info.email is None

    def test_free_user(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_account_info_free.json")
        )
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.premium_status == "free"
        assert info.premium_expires_at is None

    def test_returns_none_on_http_error(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error", "message": "BAD_TOKEN"}, status_code=401
        )
        dl = make_downloader(session)

        assert dl.get_user_info() is None

    def test_real_api_shape_parses_without_error(self):
        """Regression: the real account/info response returns premium_until as a
        unix timestamp (not an ISO string) and surfaces limit_used / booster_points.
        This must not raise.
        """
        session = MagicMock()
        session.get.return_value = make_response(
            {
                "status": "success",
                "customer_id": "42",
                "premium_until": 1893456000,
                "limit_used": 0.5,
                "booster_points": 100,
            }
        )
        dl = make_downloader(session)

        info = dl.get_user_info()

        assert info is not None
        assert info.user_id == "42"
        assert info.premium_status == "premium"
        assert info.premium_expires_at is not None
        assert info.premium_days_left is not None


class TestIsCached:
    def test_cache_hit(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_cache_check_hit.json")
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is True
        # Verify the magnet was built from the lower-cased hash and passed as items[]
        _, kwargs = session.get.call_args
        assert kwargs["params"]["items[]"] == f"magnet:?xt=urn:btih:{UBUNTU_HASH}"
        assert kwargs["params"]["apikey"] == "test-api-key"

    def test_cache_miss(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_cache_check_miss.json")
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is False

    def test_http_error_5xx_raises_circuit_breaker(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error", "message": "error"}, status_code=500
        )
        dl = make_downloader(session)

        with pytest.raises(CircuitBreakerOpen):
            dl._is_cached(UBUNTU_HASH)

    def test_client_error_returns_false(self):
        # 4xx (non-429) failures are not circuit-breaker events; _is_cached
        # treats them as "not cached" rather than raising.
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error", "message": "bad request"}, status_code=400
        )
        dl = make_downloader(session)

        assert dl._is_cached(UBUNTU_HASH) is False


class TestAddTorrent:
    def test_success_returns_id(self):
        session = MagicMock()
        session.post.return_value = make_response(
            load_fixture("premiumize_transfer_create.json")
        )
        dl = make_downloader(session)

        transfer_id = dl.add_torrent(UBUNTU_HASH)

        assert transfer_id == "tr_abcdef123456"
        # Verify magnet was built from the lower-cased hash
        _, kwargs = session.post.call_args
        assert kwargs["data"]["src"] == f"magnet:?xt=urn:btih:{UBUNTU_HASH}"
        assert kwargs["params"]["apikey"] == "test-api-key"

    def test_api_error_raises(self):
        session = MagicMock()
        session.post.return_value = make_response(
            {"status": "error", "message": "bad magnet"}, status_code=400
        )
        dl = make_downloader(session)

        with pytest.raises(PremiumizeError):
            dl.add_torrent(UBUNTU_HASH)

    def test_server_error_raises_circuit_breaker(self):
        session = MagicMock()
        session.post.return_value = make_response(
            {"status": "error", "message": "server error"}, status_code=503
        )
        dl = make_downloader(session)

        with pytest.raises(CircuitBreakerOpen):
            dl.add_torrent(UBUNTU_HASH)


class TestGetTorrentInfo:
    def test_ready_multi_file_torrent_normalizes(self):
        """Multi-file transfer: folder_id set -> /folder/list enumerates files."""
        session = MagicMock()
        # transfer/list then folder/list
        session.get.side_effect = [
            make_response(load_fixture("premiumize_transfer_list_ready.json")),
            make_response(load_fixture("premiumize_folder_list_show.json")),
        ]
        dl = make_downloader(session)

        info = dl.get_torrent_info("tr_abcdef123456")

        assert info.id == "tr_abcdef123456"
        assert info.name == "Some.Show.2026.S01.1080p"
        assert info.status == "finished"
        assert info.bytes == 1500000000 + 1450000000  # two files summed
        assert len(info.files) == 2  # the 'folder' entry is skipped
        # second call asked folder/list with id=fld_show_001
        second_call_args = session.get.call_args_list[1]
        assert second_call_args.kwargs["params"]["id"] == "fld_show_001"

    def test_download_url_populated_as_synthetic_link(self):
        """Regression: get_torrent_info must populate a non-empty download_url
        for each file so _update_attributes creates a MediaEntry and items
        advance past Unknown. Uses a synthetic 'premiumize:<folder>:<file>' form
        (no API call) to avoid eagerly tripping the rate limit.
        """
        session = MagicMock()
        session.get.side_effect = [
            make_response(load_fixture("premiumize_transfer_list_ready.json")),
            make_response(load_fixture("premiumize_folder_list_show.json")),
        ]
        dl = make_downloader(session)

        info = dl.get_torrent_info("tr_abcdef123456")

        for f in info.files.values():
            assert f.download_url, "download_url must be populated"
            assert f.download_url.startswith("premiumize:fld_show_001:")
        assert len(info.links) == 2

    def test_single_file_transfer_uses_root(self):
        """Single-file transfer: no folder_id -> /folder/list called without id,
        resolving to the cloud root."""
        session = MagicMock()
        session.get.side_effect = [
            make_response(
                {
                    "status": "success",
                    "transfers": [
                        {
                            "id": "tr_single_1",
                            "name": "Movie.mkv",
                            "status": "finished",
                            "folder_id": None,
                            "file_id": "4f2a99",
                        }
                    ],
                }
            ),
            make_response(
                {
                    "status": "success",
                    "content": [
                        {
                            "id": "4f2a99",
                            "name": "Movie.2026.1080p.mkv",
                            "type": "file",
                            "size": 2000000000,
                            "link": "https://www.premiumize.me/files?file_id=4f2a99",
                        }
                    ],
                }
            ),
        ]
        dl = make_downloader(session)

        info = dl.get_torrent_info("tr_single_1")

        assert info.bytes == 2000000000
        assert len(info.files) == 1
        # root folder => no 'id' param passed to folder/list
        second_call_args = session.get.call_args_list[1]
        assert "id" not in second_call_args.kwargs["params"]
        # synthetic url uses the root marker
        f = next(iter(info.files.values()))
        assert f.download_url == f"premiumize:{ROOT_FOLDER_MARKER}:4f2a99"

    def test_transfer_not_found_raises(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "success", "transfers": []}
        )
        dl = make_downloader(session)

        with pytest.raises(PremiumizeError):
            dl.get_torrent_info("tr_missing")

    def test_api_error_raises(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error", "message": "not found"}, status_code=404
        )
        dl = make_downloader(session)

        with pytest.raises(PremiumizeError):
            dl.get_torrent_info("tr_999")


class TestDeleteTorrent:
    def test_success(self):
        session = MagicMock()
        session.post.return_value = make_response(
            {"status": "success", "message": ""}
        )
        dl = make_downloader(session)

        # Should not raise
        dl.delete_torrent("tr_abcdef123456")

        _, kwargs = session.post.call_args
        assert kwargs["data"]["id"] == "tr_abcdef123456"
        assert kwargs["params"]["apikey"] == "test-api-key"

    def test_api_error_raises(self):
        session = MagicMock()
        session.post.return_value = make_response(
            {"status": "error", "message": "nope"}, status_code=400
        )
        dl = make_downloader(session)

        with pytest.raises(PremiumizeError):
            dl.delete_torrent("tr_abcdef123456")


class TestSelectFiles:
    def test_is_noop(self):
        # Premiumize auto-selects; select_files must not touch the session.
        dl = make_downloader(MagicMock())
        dl.select_files("tr_1", [1, 2])  # should not raise
        dl.api.session.post.assert_not_called()


class TestUnrestrictLink:
    def test_returns_unrestricted_link(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_folder_list_show.json")
        )
        dl = make_downloader(session)

        link = dl.unrestrict_link("premiumize:fld_show_001:4f2a01")

        assert link is not None
        assert link.download == "https://www.premiumize.me/files?file_id=4f2a01&token=abc"
        assert link.filename == "Some.Show.2026.S01E01.1080p.mkv"
        assert link.filesize == 1500000000

        # folder id passed through for lookup
        _, kwargs = session.get.call_args
        assert kwargs["params"]["id"] == "fld_show_001"
        assert kwargs["params"]["apikey"] == "test-api-key"

    def test_root_folder_omits_id_param(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {
                "status": "success",
                "content": [
                    {
                        "id": "4f2a99",
                        "name": "Movie.mkv",
                        "type": "file",
                        "size": 100,
                        "link": "https://cdn.premiumize.me/x",
                    }
                ],
            }
        )
        dl = make_downloader(session)

        link = dl.unrestrict_link(f"premiumize:{ROOT_FOLDER_MARKER}:4f2a99")

        assert link is not None
        _, kwargs = session.get.call_args
        assert "id" not in kwargs["params"]

    def test_file_not_in_folder_returns_none(self):
        session = MagicMock()
        session.get.return_value = make_response(
            load_fixture("premiumize_folder_list_show.json")
        )
        dl = make_downloader(session)

        assert dl.unrestrict_link("premiumize:fld_show_001:does_not_exist") is None

    def test_non_streamable_link_returns_none(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {
                "status": "success",
                "content": [
                    {
                        "id": "4f2a01",
                        "name": "v.mkv",
                        "type": "file",
                        "size": 1,
                        "link": "not-a-url",
                    }
                ],
            }
        )
        dl = make_downloader(session)

        assert dl.unrestrict_link("premiumize:fld_show_001:4f2a01") is None

    def test_returns_none_on_http_error(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error", "message": "unavailable"}, status_code=502
        )
        dl = make_downloader(session)

        assert dl.unrestrict_link("premiumize:fld_show_001:4f2a01") is None


class TestCleanupTransfers:
    """Premiumize cloud-storage cleanup: old finished transfers are deleted so
    the cloud storage quota doesn't fill up and block new downloads.
    """

    def _make_transfers(self, n, status="finished"):
        return [{"id": f"t{i}", "status": status, "name": f"item {i}"} for i in range(n)]

    def test_deletes_old_finished_beyond_keep_recent(self):
        session = MagicMock()
        # 60 finished transfers, keep_recent=20 -> 40 oldest deleted.
        # transfer/list is newest-first, so deletions are the LAST 40.
        session.get.return_value = make_response(
            {"status": "success", "transfers": self._make_transfers(60)}
        )
        session.post.return_value = make_response({"status": "success"})
        dl = make_downloader(session)

        deleted = dl.cleanup_transfers(keep_recent=20)

        assert deleted == 40
        # delete_torrent posts to transfer/delete 40 times
        assert session.post.call_count == 40

    def test_keeps_all_when_under_threshold(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "success", "transfers": self._make_transfers(15)}
        )
        dl = make_downloader(session)

        assert dl.cleanup_transfers(keep_recent=20) == 0
        session.post.assert_not_called()

    def test_never_deletes_active_transfers(self):
        session = MagicMock()
        transfers = self._make_transfers(60) + [
            {"id": "active1", "status": "running", "name": "running"},
            {"id": "active2", "status": "queued", "name": "queued"},
        ]
        session.get.return_value = make_response(
            {"status": "success", "transfers": transfers}
        )
        session.post.return_value = make_response({"status": "success"})
        dl = make_downloader(session)

        deleted = dl.cleanup_transfers(keep_recent=20)
        assert deleted == 40  # only finished beyond threshold
        # none of the delete calls target the active transfers
        for call in session.post.call_args_list:
            assert call.kwargs["data"]["id"] not in ("active1", "active2")

    def test_returns_zero_on_api_error(self):
        session = MagicMock()
        session.get.return_value = make_response(
            {"status": "error"}, status_code=500
        )
        dl = make_downloader(session)
        # _maybe_backoff raises CircuitBreakerOpen on 500; cleanup catches it
        assert dl.cleanup_transfers() == 0


class TestRouterSerialization:
    """Regression: /downloader_user_info builds DownloaderUserInfo(service=...).

    The router model's service Literal must include "premiumize" or the
    dashboard returns 500 the moment Premiumize is initialized.
    """

    def test_downloader_user_info_accepts_premiumize(self):
        # Must not raise ValidationError.
        info = DownloaderUserInfo(
            service="premiumize",
            username="7654321",
            user_id=7654321,
            premium_status="premium",
        )
        assert info.service == "premiumize"

    def test_downloader_user_info_accepts_other_services(self):
        # Ensure the change didn't drop the existing services.
        for svc in ("realdebrid", "alldebrid", "debridlink", "torbox"):
            info = DownloaderUserInfo(service=svc, user_id=1, premium_status="premium")
            assert info.service == svc
