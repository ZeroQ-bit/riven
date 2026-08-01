from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel

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
from program.utils.debrid_link_status import is_streamable_http_url
from program.utils.request import CircuitBreakerOpen, SmartResponse, SmartSession

from .shared import DownloaderBase, premium_days_left

# --- Premiumize API response models --------------------------------------------
# Premiumize wraps every response in {"status": "success"|"error", "message": str}.
# The payload fields sit at the top level alongside status (e.g. "transfers",
# "content", "response", "customer_id").

BASE_URL = "https://www.premiumize.me/api"


class PremiumizeResponse(BaseModel):
    """Generic Premiumize response envelope."""

    status: str  # "success" | "error"
    message: str | None = None


class PremiumizeUser(BaseModel):
    customer_id: int | str | None = None
    # Unix timestamp (seconds) of premium expiry; null/0 for free accounts.
    premium_until: int | None = None
    limit_used: float | None = None
    booster_points: int | None = None


class PremiumizeTransfer(BaseModel):
    """A transfer object from /transfer/list."""

    id: str
    name: str | None = None
    # One of: queued, running, finished, seeding, error
    status: str | None = None
    progress: float | None = None
    message: str | None = None
    folder_id: str | None = None  # set when the transfer produced a folder
    file_id: str | None = None  # set for single-file transfers


class PremiumizeContent(BaseModel):
    """A file or folder entry from /folder/list."""

    id: str
    name: str | None = None
    type: str = "file"  # "file" | "folder"
    size: int = 0  # bytes (files only)
    link: str | None = None  # CDN download URL (files only)
    created_at: int | None = None  # unix timestamp


class PremiumizeError(Exception):
    """Base exception for Premiumize related errors."""


# Premiumize statuses that indicate the transfer's files are available.
PREMIUMIZE_READY_STATES = frozenset({"finished", "seeding"})

# Marker used in synthetic download URLs to mean "the file lives in the cloud
# root folder" (single-file transfers where transfer/list returns file_id but
# no folder_id).
ROOT_FOLDER_MARKER = "root"


class PremiumizeAPI:
    """
    Minimal Premiumize API client using SmartSession for retries, rate limits,
    and circuit breaker.

    Auth: the API key is passed as the ``apikey`` query parameter on every
    request. Premiumize also accepts a Bearer header, but the ``apikey`` param
    path is the one exercised by the long-standing plex_debrid integration and
    is accepted uniformly across every endpoint (including folder listings).
    """

    BASE_URL = BASE_URL

    def __init__(self, api_key: str, proxy_url: str | None = None) -> None:
        self.api_key = api_key
        self.proxy_url = proxy_url

        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        self.session = SmartSession(
            base_url=self.BASE_URL,
            rate_limits={
                # Premiumize does not publish hard limits; the service is
                # generous. 5 rps with a 30-request burst mirrors the conservative
                # TorBox bucket and keeps the circuit breaker from starving
                # large library rescrapes.
                "www.premiumize.me": {
                    "rate": 5,
                    "capacity": 30,
                },
            },
            proxies=proxies,
            retries=2,
            backoff_factor=0.5,
        )


class PremiumizeDownloader(DownloaderBase):
    """
    Premiumize downloader.

    Failure & breaker behavior mirrors TorBox:
    - SmartSession retries transport failures, then counts them against the
      per-domain CircuitBreaker; once OPEN it raises CircuitBreakerOpen.
    - HTTP status codes are surfaced via _handle_error(...); 429/5xx promote
      to a CircuitBreakerOpen so the manager can back off this service.
    - Premiumize auto-selects all files for a transfer, so select_files() is
      a no-op.

    CDN URLs are resolved lazily (see TorBox): get_torrent_info stores a
    synthetic ``premiumize:<folder>:<file>`` link per file, and unrestrict_link
    resolves it to a real Premiumize CDN URL at VFS read time. This avoids
    eagerly burning /folder/list for every file of every scanned torrent and
    sidesteps CDN-link expiry.
    """

    def __init__(self) -> None:
        self.key = "premiumize"
        self.settings = settings_manager.settings.downloaders.premiumize
        self.api: PremiumizeAPI | None = None
        self.initialized = self.validate()

    def validate(self) -> bool:
        """Validate settings and current premium status."""
        if not self._validate_settings():
            return False

        proxy_url = self.PROXY_URL or None
        self.api = PremiumizeAPI(api_key=self.settings.api_key, proxy_url=proxy_url)

        return self._validate_premium()

    def _validate_settings(self) -> bool:
        if not self.settings.enabled:
            return False

        if not self.settings.api_key:
            logger.warning("Premiumize API key is not set")
            return False

        return True

    def _validate_premium(self) -> bool:
        try:
            user_info = self.get_user_info()

            if not user_info:
                logger.error("Failed to get Premiumize user info")
                return False

            if user_info.premium_status != "premium":
                logger.error("Premiumize premium membership required")
                return False

            if user_info.premium_expires_at:
                logger.info(premium_days_left(user_info.premium_expires_at))

            return True
        except Exception as e:
            logger.error(f"Failed to validate Premiumize premium status: {e}")
            return False

    # --- shared helpers --------------------------------------------------------

    def _handle_error(self, response: SmartResponse) -> str:
        """Map HTTP status / Premiumize envelope to an error message."""
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
            return "Premiumize server error"

        try:
            envelope = PremiumizeResponse.model_validate(response.json())
            if envelope.message:
                return envelope.message
        except Exception:
            pass

        return response.reason or f"HTTP {status}"

    def _maybe_backoff(self, response: SmartResponse) -> None:
        """Promote Premiumize 429/5xx to a service-level backoff signal."""
        code = response.status_code

        if code == 429 or (500 <= code < 600):
            raise CircuitBreakerOpen("www.premiumize.me")

    def _authed_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build query params carrying the apikey plus any extras."""
        params: dict[str, Any] = {"apikey": self.api.api_key if self.api else ""}
        if extra:
            params.update(extra)
        return params

    # --- abstract method implementations --------------------------------------

    def get_instant_availability(
        self,
        infohash: str,
        item_type: ProcessedItemType,
        **kwargs: Any,
    ) -> TorrentContainer | None:
        """
        Check whether a torrent is cached on Premiumize and return its files.

        Uses /cache/check first (cheap). Only when cached do we add the
        transfer and fetch its files.
        """
        transfer_id: str | None = None

        try:
            if not self._is_cached(infohash):
                logger.debug(f"Premiumize: {infohash} not cached")
                return None

            transfer_id = self.add_torrent(infohash)
            container, reason, info = self._process_torrent(
                transfer_id, infohash, item_type
            )

            if container is None and reason:
                logger.debug(f"Availability check failed [{infohash}]: {reason}")

                if transfer_id:
                    try:
                        self.delete_torrent(transfer_id)
                    except Exception as e:
                        logger.debug(
                            f"Failed to delete failed transfer {transfer_id}: {e}"
                        )

                return None

            # Cache transfer_id + info so the download phase can skip
            # add_torrent + get_torrent_info (saves API calls per stream).
            if container:
                container.torrent_id = transfer_id
                container.torrent_info = info

            return container

        except CircuitBreakerOpen:
            logger.debug(f"Circuit breaker OPEN for Premiumize; skipping {infohash}")

            if transfer_id:
                try:
                    self.delete_torrent(transfer_id)
                except Exception:
                    pass

            raise
        except PremiumizeError as e:
            # Premiumize returns "Your space is full!" when the cloud storage
            # quota is exhausted. Clean up old transfers immediately so the
            # download can proceed on the next attempt.
            if "space is full" in str(e).lower():
                logger.info(
                    "Premiumize cloud storage full during availability check; "
                    "triggering immediate cleanup"
                )
                self.cleanup_transfers(keep_recent=10)
            else:
                logger.warning(f"Availability check failed [{infohash}]: {e}")

            if transfer_id:
                try:
                    self.delete_torrent(transfer_id)
                except Exception:
                    pass

            return None
        except InvalidDebridFileException as e:
            logger.debug(
                f"Availability check failed [{infohash}]: Invalid debrid file(s) - {e}"
            )

            if transfer_id:
                try:
                    self.delete_torrent(transfer_id)
                except Exception:
                    pass

            return None
        except Exception as e:
            logger.debug(f"Availability check failed [{infohash}]: {e}")

            if transfer_id:
                try:
                    self.delete_torrent(transfer_id)
                except Exception:
                    pass

            return None

    def _is_cached(self, infohash: str) -> bool:
        """Query /cache/check for a single infohash (via its magnet link)."""
        assert self.api

        magnet = f"magnet:?xt=urn:btih:{infohash.lower()}"

        response = self.api.session.get(
            "cache/check",
            params=self._authed_params({"items[]": magnet}),
        )
        self._maybe_backoff(response)

        if not response.ok:
            logger.debug(
                f"Premiumize cache/check failed for {infohash}: {self._handle_error(response)}"
            )
            return False

        try:
            body = response.json()
        except Exception as e:
            logger.debug(f"Premiumize cache/check unparseable for {infohash}: {e}")
            return False

        if body.get("status") != "success":
            return False

        cache_flags = body.get("response")
        # response is a list of bools parallel to items[]
        if isinstance(cache_flags, list) and cache_flags:
            return bool(cache_flags[0])

        return False

    def _process_torrent(
        self,
        transfer_id: str,
        infohash: str,
        item_type: ProcessedItemType,
    ) -> tuple[TorrentContainer | None, str | None, TorrentInfo | None]:
        """Process a single transfer and return (container, reason, info)."""
        info = self.get_torrent_info(transfer_id)

        if not info:
            return None, "no transfer info returned by Premiumize", None

        if not info.files:
            return None, "no files present in the transfer", None

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
        """A Premiumize transfer is fully available when finished or seeding."""
        status = (info.status or "").lower()
        return status in PREMIUMIZE_READY_STATES

    def add_torrent(self, infohash: str) -> str:
        """
        Add a torrent by infohash via magnet link.

        Returns:
            The Premiumize transfer id.

        Raises:
            CircuitBreakerOpen: If the per-domain breaker is OPEN.
            PremiumizeError: If the API returns a failing status.
        """
        assert self.api

        magnet = f"magnet:?xt=urn:btih:{infohash.lower()}"

        response = self.api.session.post(
            "transfer/create",
            params=self._authed_params(),
            data={"src": magnet},
        )
        self._maybe_backoff(response)

        if not response.ok:
            raise PremiumizeError(self._handle_error(response))

        try:
            body = response.json()
        except Exception as e:
            raise PremiumizeError(f"Unparseable transfer/create response: {e}")

        if body.get("status") != "success":
            raise PremiumizeError(body.get("message") or "transfer/create failed")

        transfer_id = body.get("id")
        if not transfer_id:
            raise PremiumizeError(
                f"No transfer id in transfer/create response: {body.get('message')}"
            )

        return str(transfer_id)

    def select_files(self, torrent_id: int | str, file_ids: list[int]) -> None:
        """
        Select which files to download from the transfer.

        Premiumize auto-selects all files when a transfer is added; explicit
        selection is not required (mirrors the TorBox / DebridLink behavior).
        """

    def get_torrent_info(self, torrent_id: int | str) -> TorrentInfo:
        """
        Retrieve transfer information and normalize into TorrentInfo.

        Uses /transfer/list to fetch the transfer, then /folder/list to
        enumerate its files (transfer.folder_id, or the cloud root for
        single-file transfers).
        """
        assert self.api

        transfer = self._find_transfer_by_id(str(torrent_id))

        if transfer is None:
            raise PremiumizeError(f"Transfer {torrent_id} not found")

        files, total_bytes, folder_token = self._collect_files(transfer)

        # Premiumize's /transfer/list does not reliably return created_at;
        # leave it unset rather than guess.
        created_at = None

        return TorrentInfo(
            id=transfer.id,
            name=transfer.name or "",
            status=transfer.status,
            infohash=None,
            bytes=total_bytes,
            created_at=created_at,
            progress=transfer.progress,
            files=files,
            links=list(files[f].download_url for f in files),
            alternative_filename=folder_token,
        )

    def _find_transfer_by_id(self, transfer_id: str) -> PremiumizeTransfer | None:
        """Look up a single transfer via /transfer/list."""
        assert self.api

        response = self.api.session.get(
            "transfer/list", params=self._authed_params()
        )
        self._maybe_backoff(response)

        if not response.ok:
            raise PremiumizeError(self._handle_error(response))

        try:
            body = response.json()
        except Exception as e:
            raise PremiumizeError(f"Unparseable transfer/list response: {e}")

        if body.get("status") != "success":
            raise PremiumizeError(body.get("message") or "transfer/list failed")

        transfers = body.get("transfers") or []
        for item in transfers:
            if not isinstance(item, dict):
                continue
            if str(item.get("id")) == transfer_id:
                try:
                    return PremiumizeTransfer.model_validate(item)
                except Exception as e:
                    logger.debug(f"transfer/list item unparseable: {e}")
                    return None

        return None

    def _collect_files(
        self, transfer: PremiumizeTransfer
    ) -> tuple[dict[int, TorrentFile], int, str]:
        """
        Enumerate the files for a finished transfer.

        Returns:
            (files dict keyed by int file id, total bytes, folder_token) where
            folder_token is either the folder id or ROOT_FOLDER_MARKER.

        Each file gets a synthetic ``premiumize:<folder_token>:<file_id>``
        download_url resolved lazily by unrestrict_link().
        """
        assert self.api

        # Determine which folder holds the files. Multi-file torrents expose
        # folder_id; single-file torrents expose file_id and live in the root.
        folder_id = transfer.folder_id
        is_root = not folder_id

        response = self.api.session.get(
            "folder/list",
            params=self._authed_params({"id": folder_id} if folder_id else None),
        )
        self._maybe_backoff(response)

        if not response.ok:
            raise PremiumizeError(self._handle_error(response))

        try:
            body = response.json()
        except Exception as e:
            raise PremiumizeError(f"Unparseable folder/list response: {e}")

        if body.get("status") != "success":
            raise PremiumizeError(body.get("message") or "folder/list failed")

        folder_token = folder_id or ROOT_FOLDER_MARKER
        files = dict[int, TorrentFile]()
        total_bytes = 0

        for item in body.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "file":
                continue

            try:
                content = PremiumizeContent.model_validate(item)
            except Exception as e:
                logger.debug(f"folder/list file unparseable: {e}")
                continue

            try:
                # Premiumize file ids are hex strings; map to an int key so the
                # TorrentFile.id / DebridFile.file_id contract (int) holds. The
                # original string id is preserved inside the synthetic URL.
                file_key = self._file_id_to_int(content.id)
            except ValueError:
                logger.debug(f"Premiumize file id not convertible: {content.id}")
                continue

            synthetic_url = f"premiumize:{folder_token}:{content.id}"

            files[file_key] = TorrentFile(
                id=file_key,
                path=content.name or "",
                bytes=content.size,
                selected=1,  # all files available on Premiumize
                download_url=synthetic_url,
            )
            total_bytes += content.size

        return files, total_bytes, folder_token

    def delete_torrent(self, torrent_id: int | str) -> None:
        """
        Delete a transfer on Premiumize via /transfer/delete.

        Raises:
            CircuitBreakerOpen: If the per-domain breaker is OPEN.
            PremiumizeError: If the API returns a failing status.
        """
        assert self.api

        response = self.api.session.post(
            "transfer/delete",
            params=self._authed_params(),
            data={"id": str(torrent_id)},
        )
        self._maybe_backoff(response)

        if not response.ok:
            raise PremiumizeError(self._handle_error(response))

    def cleanup_transfers(self, keep_recent: int = 20) -> int:
        """
        Periodically delete old finished transfers to free Premiumize cloud
        storage. Premiumize (unlike Real-Debrid/TorBox) enforces a cloud
        storage quota; without cleanup, finished transfers accumulate and
        eventually trigger 'Your space is full' errors that block all new
        downloads.

        Keeps the ``keep_recent`` most-recent finished transfers (so files
        currently being streamed via the VFS stay available) and deletes the
        rest, oldest first, until cloud usage drops below the safe threshold.
        Active (running/queued/seeding) transfers are never touched.

        Returns:
            The number of transfers deleted.
        """
        try:
            assert self.api

            response = self.api.session.get(
                "transfer/list", params=self._authed_params()
            )
            self._maybe_backoff(response)

            if not response.ok:
                logger.debug(
                    f"Premiumize cleanup: transfer/list failed: {self._handle_error(response)}"
                )
                return 0

            try:
                body = response.json()
            except Exception:
                return 0

            if body.get("status") != "success":
                return 0

            transfers = body.get("transfers") or []
            # Only finished transfers are candidates for cleanup; never delete
            # active ones (running/queued/seeding).
            finished = [
                t for t in transfers
                if isinstance(t, dict) and t.get("status") == "finished"
            ]

            if len(finished) <= keep_recent:
                return 0

            # transfer/list is newest-first per the Premiumize docs, so the
            # oldest transfers to delete are at the end of the list.
            to_delete = finished[keep_recent:]

            deleted = 0
            for t in to_delete:
                tid = t.get("id")
                if not tid:
                    continue
                try:
                    self.delete_torrent(tid)
                    deleted += 1
                except Exception as e:
                    logger.debug(f"Premiumize cleanup: failed to delete transfer {tid}: {e}")

            if deleted:
                logger.info(
                    f"Premiumize cleanup: freed {deleted} old transfer(s) from cloud storage"
                )

            return deleted
        except CircuitBreakerOpen:
            logger.debug("Premiumize cleanup: circuit breaker OPEN, skipping")
            return 0
        except Exception as e:
            logger.debug(f"Premiumize cleanup: failed: {e}")
            return 0

    def unrestrict_link(self, link: str) -> UnrestrictedLink | None:
        """
        Resolve a Premiumize CDN download URL for a file.

        The ``link`` is expected to be the synthetic
        ``premiumize:<folder_token>:<file_id>`` form produced by
        get_torrent_info. The file's live ``link`` is re-fetched via
        /folder/list so expired CDN URLs are refreshed transparently.
        """
        try:
            assert self.api

            folder_token, file_id = self._parse_link(link)

            params = self._authed_params()
            if folder_token and folder_token != ROOT_FOLDER_MARKER:
                params["id"] = folder_token

            response = self.api.session.get("folder/list", params=params, timeout=10)
            self._maybe_backoff(response)

            if not response.ok:
                logger.warning(
                    f"Premiumize folder/list failed while unrestricting {link}: "
                    f"{self._handle_error(response)}"
                )
                return None

            try:
                body = response.json()
            except Exception as e:
                logger.debug(f"Premiumize folder/list unparseable for {link}: {e}")
                return None

            if body.get("status") != "success":
                return None

            for item in body.get("content") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id")) == file_id:
                    download_url = item.get("link")
                    if not download_url or not is_streamable_http_url(download_url):
                        logger.debug(
                            f"Premiumize file {file_id} has no streamable link"
                        )
                        return None

                    return UnrestrictedLink(
                        download=download_url,
                        filename=item.get("name") or f"premiumize-{file_id}",
                        filesize=int(item.get("size") or 0),
                    )

            logger.debug(f"Premiumize file {file_id} not found in folder {folder_token}")
            return None
        except CircuitBreakerOpen as e:
            logger.warning(
                f"Circuit breaker OPEN while unrestricting Premiumize link: {e}"
            )
            return None
        except Exception as e:
            logger.debug(f"Premiumize unrestrict_link failed for {link}: {e}")
            return None

    @staticmethod
    def _parse_link(link: str) -> tuple[str, str]:
        """
        Parse a Premiumize synthetic link into (folder_token, file_id).

        Accepts:
        - Synthetic format: "premiumize:<folder_token>:<file_id>" (used
          internally to defer CDN URL resolution until the VFS reads the file).
        - Compact format: "<folder_token>:<file_id>".
        """
        if link.startswith("premiumize:"):
            link = link[len("premiumize") :]
            if link.startswith(":"):
                link = link[1:]

        if ":" in link:
            folder_token, file_id = link.split(":", 1)
            return folder_token, file_id

        # Bare file id: assume it lives in the cloud root.
        return ROOT_FOLDER_MARKER, link

    @staticmethod
    def _file_id_to_int(file_id: str) -> int:
        """
        Convert a Premiumize file id (hex string) into a stable int key.

        Premiumize ids are variable-length hex strings (e.g. "4f2a..."). We
        cannot rely on them being short, so fold them via a 31-bit hash so the
        value fits TorrentFile.id (int) without collisions in practice.
        """
        # Interpret as a base-16 integer, then mask to a positive 31-bit range
        # to keep it a small, stable key. Collisions across a single torrent's
        # files are extremely unlikely and only affect the in-memory dict key.
        value = int(file_id, 16) if all(c in "0123456789abcdefABCDEF" for c in file_id) else abs(hash(file_id))
        return value & 0x7FFFFFFF

    def get_user_info(self) -> UserInfo | None:
        """
        Get normalized user information from Premiumize.

        Returns:
            UserInfo with normalized fields, or None on error.
        """
        try:
            assert self.api

            response = self.api.session.get(
                "account/info", params=self._authed_params()
            )
            self._maybe_backoff(response)

            if not response.ok:
                logger.error(
                    f"Failed to get Premiumize user info: {self._handle_error(response)}"
                )
                return None

            try:
                body = response.json()
            except Exception as e:
                logger.error(f"Premiumize account/info unparseable: {e}")
                return None

            if body.get("status") != "success":
                logger.error(
                    f"Premiumize account/info failed: {body.get('message')}"
                )
                return None

            try:
                user = PremiumizeUser.model_validate(body)
            except Exception as e:
                logger.error(f"Premiumize account/info returned no usable data: {e}")
                return None

            expiration = None
            premium_days = None

            if user.premium_until:
                try:
                    expiration = datetime.fromtimestamp(
                        int(user.premium_until), tz=timezone.utc
                    )
                    time_left = expiration - datetime.now(tz=timezone.utc)
                    premium_days = time_left.days
                except (ValueError, TypeError, OSError) as e:
                    logger.debug(f"Failed to parse Premiumize expiration: {e}")

            is_premium = bool(
                user.premium_until and int(user.premium_until) > 0
            )

            return UserInfo(
                service="premiumize",
                # Premiumize exposes no username; reuse customer_id string so the
                # dashboard shows a meaningful identifier.
                username=str(user.customer_id) if user.customer_id is not None else None,
                email=None,
                user_id=user.customer_id if user.customer_id is not None else 0,
                premium_status="premium" if is_premium else "free",
                premium_expires_at=(
                    expiration.replace(tzinfo=None) if expiration else None
                ),
                premium_days_left=premium_days,
            )
        except CircuitBreakerOpen as e:
            logger.warning(
                f"Circuit breaker OPEN while getting Premiumize user info: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to get Premiumize user info: {e}")
            return None
