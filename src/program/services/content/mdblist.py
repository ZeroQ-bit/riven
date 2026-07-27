"""Mdblist content module"""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from kink import di
from loguru import logger

from program.apis.mdblist_api import (
    MdblistAPI,
    MdblistFetchResult,
    MdblistListItem,
)
from program.core.runner import MediaItemGenerator, Runner, RunnerResult
from program.db.db_functions import item_exists_by_any_id
from program.media.item import MediaItem
from program.settings import settings_manager
from program.settings.models import MdblistModel


@dataclass
class ImportCounts:
    """Counters used to make MDBList import decisions visible in logs."""

    fetched: int = 0
    new: int = 0
    existing: int = 0
    duplicates: int = 0
    missing_ids: int = 0
    invalid: int = 0

    def include(self, other: "ImportCounts") -> None:
        self.fetched += other.fetched
        self.new += other.new
        self.existing += other.existing
        self.duplicates += other.duplicates
        self.missing_ids += other.missing_ids
        self.invalid += other.invalid


class Mdblist(Runner[MdblistModel]):
    """Content class for mdblist"""

    is_content_service = True

    def __init__(self):
        super().__init__()

        self.settings = settings_manager.settings.content.mdblist

        if not self.enabled:
            return

        self.api = di[MdblistAPI]
        self.initialized = self.validate()

        if not self.initialized:
            return

        self.requests_per_2_minutes = self._calculate_request_time()

        logger.success("mdblist initialized")

    def validate(self):
        if not self.settings.enabled:
            return False

        if self.settings.api_key == "" or len(self.settings.api_key) != 25:
            logger.error("Mdblist api key is not set.")
            return False

        if not self.settings.lists:
            logger.error("Mdblist is enabled, but you haven't added any lists.")
            return False

        return self.api.validate()

    def run(self, item: MediaItem) -> MediaItemGenerator:  # noqa: ARG002
        """Fetch media from every configured MDBList."""

        items_to_yield: list[MediaItem] = []
        seen_items: set[tuple[str, int]] = set()
        totals = ImportCounts()
        successful_lists = 0
        failed_lists = 0

        for list_id in self.settings.lists:
            if not list_id:
                continue

            list_label = self._list_label(list_id)
            try:
                result = self._fetch_list(list_id)
            except Exception as error:
                failed_lists += 1
                if self._is_rate_limit_error(error):
                    logger.warning(
                        "MDBList rate limit reached while importing {}; "
                        "remaining lists will retry on the next sync",
                        list_label,
                    )
                    break

                logger.error(
                    "MDBList failed to import {}; continuing with the next list: {}",
                    list_label,
                    error,
                )
                continue

            successful_lists += 1
            new_items, counts = self._collect_new_items(result, seen_items)
            items_to_yield.extend(new_items)
            totals.include(counts)
            self._log_list_result(list_label, result.pages, counts)

        logger.info(
            "MDBList import complete: lists={}, failed_lists={}, fetched={}, new={}, "
            "existing={}, duplicates={}, missing_ids={}, invalid={}",
            successful_lists,
            failed_lists,
            totals.fetched,
            totals.new,
            totals.existing,
            totals.duplicates,
            totals.missing_ids,
            totals.invalid,
        )

        yield RunnerResult(media_items=items_to_yield)

    def _fetch_list(self, list_id: int | str) -> MdblistFetchResult:
        if isinstance(list_id, int):
            return self.api.list_items_by_id(list_id)

        return self.api.list_items_by_url(list_id)

    def _collect_new_items(
        self,
        result: MdblistFetchResult,
        seen_items: set[tuple[str, int]],
    ) -> tuple[list[MediaItem], ImportCounts]:
        counts = ImportCounts(
            fetched=len(result.items),
            invalid=result.invalid_items,
        )
        media_items: list[MediaItem] = []

        for list_item in result.items:
            media_item = self._to_media_item(list_item, seen_items, counts)
            if media_item is not None:
                media_items.append(media_item)

        counts.new = len(media_items)
        return media_items, counts

    def _to_media_item(
        self,
        list_item: MdblistListItem,
        seen_items: set[tuple[str, int]],
        counts: ImportCounts,
    ) -> MediaItem | None:
        provider = "movie" if list_item.mediatype == "movie" else "show"
        provider_id = list_item.id if provider == "movie" else list_item.tvdb_id

        if provider_id is None:
            counts.missing_ids += 1
            return None

        identity = (provider, provider_id)
        if identity in seen_items:
            counts.duplicates += 1
            return None
        seen_items.add(identity)

        if provider == "movie":
            exists = item_exists_by_any_id(
                imdb_id=list_item.imdb_id,
                tmdb_id=str(provider_id),
            )
            item_data = {"tmdb_id": provider_id, "requested_by": self.key}
        else:
            exists = item_exists_by_any_id(
                imdb_id=list_item.imdb_id,
                tvdb_id=str(provider_id),
            )
            item_data = {"tvdb_id": provider_id, "requested_by": self.key}

        if exists:
            counts.existing += 1
            return None

        return MediaItem(item_data)

    @staticmethod
    def _log_list_result(
        list_label: str,
        pages: int,
        counts: ImportCounts,
    ) -> None:
        logger.info(
            "MDBList {}: fetched={}, new={}, existing={}, duplicates={}, "
            "missing_ids={}, invalid={}, pages={}",
            list_label,
            counts.fetched,
            counts.new,
            counts.existing,
            counts.duplicates,
            counts.missing_ids,
            counts.invalid,
            pages,
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        message = str(error)
        return "rate limit" in message.lower() or "429" in message

    @staticmethod
    def _list_label(list_id: int | str) -> str:
        """Return a useful list identifier without leaking URL query parameters."""

        if isinstance(list_id, int):
            return f"list ID {list_id}"

        parts = urlsplit(list_id)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    def _calculate_request_time(self):
        """Calculate requests per 2 minutes based on mdblist limits"""

        limits = self.api.my_limits()

        assert limits and limits.api_requests

        daily_requests = limits.api_requests

        return daily_requests / 24 / 60 * 2
