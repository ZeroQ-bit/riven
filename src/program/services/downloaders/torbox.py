from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from program.media.item import ProcessedItemType
from program.services.downloaders.models import (
    DebridFile,
    InvalidDebridFileException,
    TorrentContainer,
    TorrentFile,
    TorrentInfo,
    UnrestrictedLink,
    UserInfo,
)
from program.settings import settings_manager
from program.utils.request import CircuitBreakerOpen, SmartResponse, SmartSession

from .shared import DownloaderBase, premium_days_left

# --- TorBox API response models -------------------------------------------------
# TorBox wraps every response in {"success": bool, "detail": str, "data": <T>}.
# The OpenAPI spec leaves the 200 body schema empty; the shapes below come from
# the TorBox JS SDK and observed responses.


class TorBoxEnvelope(BaseModel):
    """Generic TorBox response envelope."""

    success: bool
    detail: str | None = None
    data: Any = None


class TorBoxUser(BaseModel):
    id: int
    # TorBox has no username; email is the account identifier.
    email: str | None = None
    plan: int = 0  # 0 = Free, 1 = Essential, 2 = Pro, 3 = Standard
    premium_expires_at: str | None = None  # ISO 8601 date string


class TorBoxFile(BaseModel):
    id: int
    name: str
    size: int = 0
    # TorBox uses "bytes" for the file size in some responses; alias it because
    # `bytes` shadows the Python builtin.
    bytes_size: int = Field(default=0, alias="bytes")

    model_config = {"populate_by_name": True}

    def get_size(self) -> int:
        return self.size or self.bytes_size


class TorBoxCreateTorrentResponse(BaseModel):
    """Creation receipt from /torrents/createtorrent.

    This is NOT a full torrent object - it only carries the new torrent's id
    (as ``torrent_id``) and the infohash. ``name`` and other fields are absent
    and must be fetched via /torrents/mylist if needed.
    """

    torrent_id: int | None = None
    queued_id: int | None = None
    hash: str | None = None


class TorBoxTorrent(BaseModel):
    """A full torrent object from /torrents/mylist.

    Field types per the TorBox JS SDK (get-torrent-list-ok-response-data).
    Note: created_at and expires_at are ISO 8601 strings, NOT unix timestamps.
    """

    id: int
    name: str
    hash: str | None = None
    info_hash: str | None = None
    size: int = 0
    bytes: int = 0
    download_state: str = "queued"
    download_finished: bool = False
    cached: bool = False
    created_at: str | None = None  # ISO 8601 string, e.g. "2026-07-26T02:33:36Z"
    expires_at: str | None = None  # ISO 8601 string
    progress: float | None = None
    files: list[TorBoxFile] | None = None


class TorBoxRequestDownload(BaseModel):
    """Response from /torrents/requestdl."""

    url: str
    filename: str | None = None
    filesize: int | None = None


class TorBoxError(Exception):
    """Base exception for TorBox related errors."""


class TorBoxAPI:
    """
    Minimal TorBox API client using SmartSession for retries, rate limits, and circuit breaker.
    """

    BASE_URL = "https://api.torbox.app/v1/api"

    def __init__(self, api_key: str, proxy_url: str | None = None) -> None:
        """
        Args:
            api_key: TorBox API key.
            proxy_url: Optional proxy URL used for both HTTP and HTTPS.
        """
        self.api_key = api_key
        self.proxy_url = proxy_url

        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        self.session = SmartSession(
            base_url=self.BASE_URL,
            rate_limits={
                # TorBox default: 300 req/min ~= 5 rps, capacity 300.
                # NOTE: createtorrent is separately capped at 60/hour; SmartSession
                # does not support per-path limits, so this is a domain-wide bucket.
                "api.torbox.app": {
                    "rate": 300 / 60,
                    "capacity": 300,
                },
            },
            proxies=proxies,
            retries=2,
            backoff_factor=0.5,
        )
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})


# States TorBox reports that indicate the torrent is fully available.
TORBOX_READY_STATES = frozenset(
    {"cached", "completed", "downloaded", "finished", "uploading"}
)


class TorBoxDownloader(DownloaderBase):
    """
    TorBox downloader with lean exception handling.

    Notes on failure & breaker behavior:
    - Network/transport failures are retried by SmartSession, then counted against the
      per-domain CircuitBreaker; once OPEN, SmartSession raises CircuitBreakerOpen.
    - HTTP status codes are not exceptions; we check response.ok and map to messages
      via _handle_error(...).
    - TorBox auto-selects files when a torrent is added, so select_files() is a no-op
      (mirrors the DebridLink pattern).
    """

    def __init__(self) -> None:
        self.key = "torbox"
        self.settings = settings_manager.settings.downloaders.torbox
        self.api: TorBoxAPI | None = None
        self.initialized = self.validate()

    def validate(self) -> bool:
        """
        Validate settings and current premium status.

        Returns:
            True if ready, else False.
        """
        if not self._validate_settings():
            return False

        proxy_url = self.PROXY_URL or None
        self.api = TorBoxAPI(api_key=self.settings.api_key, proxy_url=proxy_url)

        return self._validate_premium()

    def _validate_settings(self) -> bool:
        """
        Returns:
            True when enabled and API key present; otherwise False.
        """
        if not self.settings.enabled:
            return False

        if not self.settings.api_key:
            logger.warning("TorBox API key is not set")
            return False

        return True

    def _validate_premium(self) -> bool:
        """
        Returns:
            True if premium membership is active; otherwise False.
        """
        try:
            user_info = self.get_user_info()

            if not user_info:
                logger.error("Failed to get TorBox user info")
                return False

            if user_info.premium_status != "premium":
                logger.error("TorBox premium membership required")
                return False

            if user_info.premium_expires_at:
                logger.info(premium_days_left(user_info.premium_expires_at))

            return True
        except Exception as e:
            logger.error(f"Failed to validate TorBox premium status: {e}")
            return False

    def _handle_error(self, response: SmartResponse) -> str:
        """
        Map HTTP status codes / TorBox error envelopes to error messages.
        """
        status = response.status_code

        if status == 401:
            return "Unauthorized - check API key"
        if status == 403:
            return "Forbidden"
        if status == 404:
            return "Not found"
        if status == 429:
            return "Rate limit exceeded"
        if status >= 500:
            return "TorBox server error"

        # Try to extract TorBox's own detail message
        try:
            envelope = TorBoxEnvelope.model_validate(response.json())
            if envelope.detail:
                return envelope.detail
        except Exception:
            pass

        return response.reason or f"HTTP {status}"

    def _maybe_backoff(self, response: SmartResponse) -> None:
        """
        Promote TorBox 429/5xx responses to a service-level backoff signal.
        """
        code = response.status_code

        if code == 429 or (500 <= code < 600):
            # Name must match the breaker key configured in SmartSession rate_limits.
            raise CircuitBreakerOpen("api.torbox.app")

    def get_instant_availability(
        self,
        infohash: str,
        item_type: ProcessedItemType,
        **kwargs: Any,
    ) -> TorrentContainer | None:
        """
        Check whether a torrent is cached on TorBox and return its files if so.

        Uses /torrents/checkcached first (cheap, 1h cache). Only when cached do we
        add the torrent and fetch its files - this avoids burning the 60/hour
        createtorrent budget on non-cached hashes.
        """
        torrent_id: int | None = None

        try:
            if not self._is_cached(infohash):
                logger.debug(f"TorBox: {infohash} not cached")
                return None

            torrent_id = self.add_torrent(infohash)
            container, reason, info = self._process_torrent(
                torrent_id, infohash, item_type
            )

            if container is None and reason:
                logger.debug(f"Availability check failed [{infohash}]: {reason}")

                if torrent_id:
                    try:
                        self.delete_torrent(torrent_id)
                    except Exception as e:
                        logger.debug(
                            f"Failed to delete failed torrent {torrent_id}: {e}"
                        )

                return None

            # Cache torrent_id + info on the container so the download phase can
            # skip add_torrent + get_torrent_info (saves 2 API calls per stream).
            if container:
                container.torrent_id = torrent_id
                container.torrent_info = info

            return container

        except CircuitBreakerOpen:
            logger.debug(f"Circuit breaker OPEN for TorBox; skipping {infohash}")

            if torrent_id:
                try:
                    self.delete_torrent(torrent_id)
                except Exception:
                    pass

            raise
        except TorBoxError as e:
            logger.warning(f"Availability check failed [{infohash}]: {e}")

            if torrent_id:
                try:
                    self.delete_torrent(torrent_id)
                except Exception:
                    pass

            return None
        except InvalidDebridFileException as e:
            logger.debug(
                f"Availability check failed [{infohash}]: Invalid debrid file(s) - {e}"
            )

            if torrent_id:
                try:
                    self.delete_torrent(torrent_id)
                except Exception:
                    pass

            return None
        except Exception as e:
            logger.debug(f"Availability check failed [{infohash}]: {e}")

            if torrent_id:
                try:
                    self.delete_torrent(torrent_id)
                except Exception:
                    pass

            return None

    def _is_cached(self, infohash: str) -> bool:
        """
        Query /torrents/checkcached for a single infohash.
        """
        assert self.api

        response = self.api.session.post(
            "torrents/checkcached",
            json={"hashes": [infohash.lower()]},
            params={"format": "object"},
        )
        self._maybe_backoff(response)

        if not response.ok:
            logger.debug(
                f"TorBox checkcached failed for {infohash}: {self._handle_error(response)}"
            )
            return False

        try:
            envelope = TorBoxEnvelope.model_validate(response.json())
        except Exception as e:
            logger.debug(f"TorBox checkcached unparseable for {infohash}: {e}")
            return False

        if not envelope.success or envelope.data is None:
            return False

        # object format returns { "<hash>": bool }
        data = envelope.data
        if isinstance(data, dict):
            return bool(data.get(infohash.lower(), False))

        # list format returns [bool, ...] in hash order
        if isinstance(data, list) and data:
            return bool(data[0])

        return False

    def _process_torrent(
        self,
        torrent_id: int,
        infohash: str,
        item_type: ProcessedItemType,
    ) -> tuple[TorrentContainer | None, str | None, TorrentInfo | None]:
        """
        Process a single torrent and return (container, reason, info).
        """
        info = self.get_torrent_info(torrent_id)

        if not info:
            return None, "no torrent info returned by TorBox", None

        if not info.files:
            return None, "no files present in the torrent", None

        if not self._is_ready(info):
            return None, f"Not instantly available (status={info.status})", None

        files = list[DebridFile]()

        for file_id, file in info.files.items():
            try:
                df = DebridFile.create(
                    path=file.path,
                    filename=file.filename,
                    filesize_bytes=file.bytes,
                    filetype=item_type,
                    file_id=file_id,
                )

                if download_url := file.download_url:
                    df.download_url = download_url
                    logger.debug(f"Using correlated download URL for {file.filename}")

                files.append(df)
            except InvalidDebridFileException as e:
                logger.debug(f"{infohash}: {e}")

        if not files:
            return None, "no valid files after validation", None

        return TorrentContainer(infohash=infohash, files=files), None, info

    @staticmethod
    def _is_ready(info: TorrentInfo) -> bool:
        """
        Determine whether a TorBox torrent is fully available for download.
        TorBox's authoritative ready signal is the `cached`/`download_finished`
        flags on the raw object; TorrentInfo normalizes status to the
        download_state string.
        """
        status = (info.status or "").lower()
        return status in TORBOX_READY_STATES

    def add_torrent(self, infohash: str) -> int:
        """
        Add a torrent by infohash via magnet link.

        Returns:
            TorBox torrent id.

        Raises:
            CircuitBreakerOpen: If the per-domain breaker is OPEN.
            TorBoxError: If the API returns a failing status.
        """
        assert self.api

        magnet = f"magnet:?xt=urn:btih:{infohash.lower()}"

        response = self.api.session.post(
            "torrents/createtorrent",
            data={
                "magnet": magnet,
                "seed": "3",
                "allow_zip": "false",
                "as_queued": "false",
                "add_only_if_cached": "false",
            },
        )
        self._maybe_backoff(response)

        if not response.ok:
            # TorBox returns 400 with a "duplicate" detail when the magnet is
            # already in the user's list. Resolve the existing id via mylist.
            detail = self._handle_error(response).lower()
            if response.status_code == 400 and "duplicate" in detail:
                existing = self._find_torrent_by_hash(infohash)
                if existing is not None:
                    return existing
            raise TorBoxError(self._handle_error(response))

        try:
            envelope = TorBoxEnvelope.model_validate(response.json())
        except Exception as e:
            raise TorBoxError(f"Unparseable createtorrent response: {e}")

        if not envelope.success or envelope.data is None:
            raise TorBoxError(envelope.detail or "createtorrent failed")

        # createtorrent returns a creation receipt, not a full torrent object.
        try:
            created = TorBoxCreateTorrentResponse.model_validate(envelope.data)
        except Exception as e:
            raise TorBoxError(f"createtorrent returned no usable receipt: {e}")

        torrent_id = created.torrent_id or created.queued_id
        if not torrent_id:
            raise TorBoxError(
                f"No torrent ID in createtorrent response: {envelope.detail}"
            )

        return torrent_id

    def _find_torrent_by_hash(self, infohash: str) -> int | None:
        """
        Look up an existing torrent id by infohash via /torrents/mylist.
        """
        assert self.api

        response = self.api.session.get(
            "torrents/mylist", params={"bypass_cache": "true", "limit": 1000}
        )
        self._maybe_backoff(response)

        if not response.ok:
            return None

        try:
            envelope = TorBoxEnvelope.model_validate(response.json())
        except Exception:
            return None

        if not envelope.success or not isinstance(envelope.data, list):
            return None

        target = infohash.lower()
        for item in envelope.data:
            if not isinstance(item, dict):
                continue
            h = str(item.get("hash") or item.get("info_hash") or "").lower()
            if h == target:
                tid = item.get("id")
                if isinstance(tid, int):
                    return tid
        return None

    def select_files(self, torrent_id: int | str, file_ids: list[int]) -> None:
        """
        Select which files to download from the torrent.

        TorBox auto-selects all files when a torrent is added; explicit selection
        is not required (mirrors the DebridLink behavior).
        """

    def get_torrent_info(self, torrent_id: int | str) -> TorrentInfo:
        """
        Retrieve torrent information and normalize into TorrentInfo.

        Uses /torrents/mylist?id=<id> to fetch a single torrent.
        """
        assert self.api

        response = self.api.session.get(
            "torrents/mylist", params={"id": torrent_id, "bypass_cache": "true"}
        )
        self._maybe_backoff(response)

        if not response.ok:
            logger.debug(
                f"Failed to get torrent info for {torrent_id}: {self._handle_error(response)}"
            )
            raise TorBoxError(self._handle_error(response))

        try:
            envelope = TorBoxEnvelope.model_validate(response.json())
        except Exception as e:
            raise TorBoxError(f"Unparseable mylist response: {e}")

        if not envelope.success or envelope.data is None:
            raise TorBoxError(envelope.detail or "mylist returned no data")

        data = envelope.data
        # mylist?id= may return a single object or a one-element list
        if isinstance(data, list):
            if not data:
                raise TorBoxError(f"Torrent {torrent_id} not found")
            data = data[0]

        try:
            torrent = TorBoxTorrent.model_validate(data)
        except Exception as e:
            raise TorBoxError(f"mylist returned no usable torrent: {e}")

        # Build files dict. TorBox file entries use {id, name, size/bytes}.
        files = dict[int, TorrentFile]()
        links = list[str]()

        for file in torrent.files or []:
            files[file.id] = TorrentFile(
                id=file.id,
                path=file.name,
                bytes=file.get_size(),
                selected=1,  # all files available on TorBox
                download_url="",  # populated on demand via unrestrict_link
            )

        created_at = None
        if torrent.created_at:
            try:
                created_at = datetime.fromisoformat(
                    torrent.created_at.replace("Z", "+00:00")
                )
            except ValueError as e:
                logger.debug(f"Failed to parse TorBox created_at: {e}")

        return TorrentInfo(
            id=torrent.id,
            name=torrent.name,
            status=torrent.download_state,
            infohash=torrent.hash or torrent.info_hash,
            bytes=torrent.size or torrent.bytes,
            created_at=created_at,
            progress=torrent.progress,
            files=files,
            links=links,
        )

    def delete_torrent(self, torrent_id: int | str) -> None:
        """
        Delete a torrent on TorBox via /torrents/controltorrent.

        Raises:
            CircuitBreakerOpen: If the per-domain breaker is OPEN.
            TorBoxError: If the API returns a failing status.
        """
        assert self.api

        response = self.api.session.post(
            "torrents/controltorrent",
            json={
                "operation": "Delete",
                "torrent_id": int(torrent_id),
                "all": False,
            },
        )
        self._maybe_backoff(response)

        if not response.ok:
            raise TorBoxError(self._handle_error(response))

    def unrestrict_link(self, link: str) -> UnrestrictedLink | None:
        """
        Resolve a TorBox CDN download URL for a torrent file.

        TorBox's /torrents/requestdl requires the API key both as the Bearer
        header (set on the session) and as the `token` query param. The `link`
        passed in here is expected to be a torrent_id:file_id pair, OR a full
        requestdl URL we can re-fetch.
        """
        try:
            assert self.api

            torrent_id, file_id = self._parse_link(link)

            response = self.api.session.get(
                "torrents/requestdl",
                params={
                    "token": self.api.api_key,
                    "torrent_id": torrent_id,
                    "file_id": file_id,
                    "redirect": "false",
                },
                timeout=10,
            )
            self._maybe_backoff(response)

            if not response.ok:
                logger.warning(
                    f"TorBox requestdl failed for {link}: {self._handle_error(response)}"
                )
                return None

            try:
                envelope = TorBoxEnvelope.model_validate(response.json())
            except Exception as e:
                logger.debug(f"TorBox requestdl unparseable for {link}: {e}")
                return None

            if not envelope.success or not envelope.data:
                return None

            dl = TorBoxRequestDownload.model_validate(envelope.data)

            # Fetch filename/size from torrent info if the CDN response omitted them.
            filename = dl.filename or f"torbox-{torrent_id}-{file_id}"
            filesize = dl.filesize or 0

            return UnrestrictedLink(
                download=dl.url,
                filename=filename,
                filesize=filesize,
            )
        except CircuitBreakerOpen as e:
            logger.warning(f"Circuit breaker OPEN while unrestricting TorBox link: {e}")
            return None
        except Exception as e:
            logger.debug(f"TorBox unrestrict_link failed for {link}: {e}")
            return None

    @staticmethod
    def _parse_link(link: str) -> tuple[int, int]:
        """
        Parse a TorBox link into (torrent_id, file_id).

        Accepts either "torrent_id:file_id" or a full requestdl URL with those
        query params.
        """
        if ":" in link and "/" not in link:
            tid_str, fid_str = link.split(":", 1)
            return int(tid_str), int(fid_str)

        # Best-effort parse of query params from a URL.
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(link)
        qs = parse_qs(parsed.query)
        torrent_id = int(qs.get("torrent_id", ["0"])[0])
        file_id = int(qs.get("file_id", ["0"])[0])
        return torrent_id, file_id

    def get_user_info(self) -> UserInfo | None:
        """
        Get normalized user information from TorBox.

        Returns:
            UserInfo with normalized fields, or None on error.
        """
        try:
            assert self.api

            response = self.api.session.get("user/me")
            self._maybe_backoff(response)

            if not response.ok:
                logger.error(f"Failed to get TorBox user info: {self._handle_error(response)}")
                return None

            try:
                envelope = TorBoxEnvelope.model_validate(response.json())
            except Exception as e:
                logger.error(f"TorBox user/me unparseable: {e}")
                return None

            if not envelope.success or envelope.data is None:
                logger.error(f"TorBox user/me failed: {envelope.detail}")
                return None

            user = TorBoxUser.model_validate(envelope.data)

            expiration = None
            premium_days = None

            if user.premium_expires_at:
                try:
                    # TorBox returns an ISO 8601 string, e.g. "2025-12-01T00:00:00".
                    expiration = datetime.fromisoformat(
                        user.premium_expires_at.replace("Z", "+00:00")
                    )
                    time_left = expiration - datetime.now(expiration.tzinfo)
                    premium_days = time_left.days
                except ValueError as e:
                    logger.debug(f"Failed to parse TorBox expiration: {e}")

            return UserInfo(
                service="torbox",
                # TorBox exposes no username; reuse email for the username field
                # so the dashboard shows a meaningful identifier.
                username=user.email,
                email=user.email,
                user_id=user.id,
                premium_status="premium" if user.plan > 0 else "free",
                premium_expires_at=(
                    expiration.replace(tzinfo=None) if expiration else None
                ),
                premium_days_left=premium_days,
            )
        except CircuitBreakerOpen as e:
            logger.warning(f"Circuit breaker OPEN while getting TorBox user info: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get TorBox user info: {e}")
            return None
