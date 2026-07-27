from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import quote, urlsplit, urlunsplit

from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
)

from program.utils.request import SmartResponse, SmartSession


class MdblistAPIError(Exception):
    """Base exception for MdblistAPI related errors"""

    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__(error)


class MdblistAPIErrorResponse(BaseModel):
    error: StrictStr


class MdblistListItem(BaseModel):
    """The identity fields Riven needs from an MDBList item."""

    id: StrictInt | None = None
    imdb_id: StrictStr | None = None
    tvdb_id: StrictInt | None = Field(
        default=None,
        validation_alias=AliasChoices("tvdb_id", "tvdbid"),
    )
    mediatype: Literal["movie", "show"]
    release_year: StrictInt | None = None


@dataclass(frozen=True)
class MdblistFetchResult:
    """Validated items and diagnostics from one configured MDBList."""

    items: list[MdblistListItem]
    invalid_items: int = 0
    pages: int = 1


class MdblistAPI:
    """Handles Mdblist API communication"""

    BASE_URL = "https://api.mdblist.com"

    def __init__(self, api_key: str):
        self.session = SmartSession(
            base_url=self.BASE_URL,
            rate_limits={
                "api.mdblist.com": {
                    # 60 calls per minute
                    "rate": 1,
                    "capacity": 60,
                }
            },
            retries=3,
            backoff_factor=0.3,
        )

        self.common_query_params = {"apikey": api_key}

    def validate(self):
        try:
            response = self.session.get(
                "/user",
                params=self.common_query_params,
            )

            if not response.ok:
                error_response = MdblistAPIErrorResponse.model_validate(response.json())

                raise MdblistAPIError(error_response.error)

            return True
        except MdblistAPIError as e:
            logger.error(f"Mdblist error: {e.error}")

            return False

    def my_limits(self):
        """Wrapper for mdblist api method 'My limits'"""

        from schemas.mdblist import GetMyLimits200Response  # noqa: PLC0415

        response = self.session.get(
            "/user",
            params=self.common_query_params,
        )

        return GetMyLimits200Response.from_dict(response.json())

    def list_items_by_id(self, list_id: int) -> MdblistFetchResult:
        """Wrapper for mdblist api method 'List items'"""

        return self._list_items_paginated(
            f"/lists/{list_id}/items",
            f"list ID {list_id}",
        )

    def list_items_by_url(self, url: str) -> MdblistFetchResult:
        """Get all items from an MDBList web URL."""

        safe_url = self._safe_url(url)
        endpoint = self._api_items_endpoint(url)
        if endpoint is not None:
            return self._list_items_paginated(endpoint, safe_url)

        url = url if url.endswith("/") else f"{url}/"
        url = url if url.endswith("json/") else f"{url}json/"
        response = self.session.get(
            url,
            params=self.common_query_params,
        )
        self._raise_for_status(response, safe_url)

        raw_payload = response.json()
        if not isinstance(raw_payload, list):
            raise MdblistAPIError(f"MDBList {safe_url} returned an invalid response")
        payload = cast(list[object], raw_payload)

        items, invalid_items = self._validate_items(payload, safe_url)

        return MdblistFetchResult(items=items, invalid_items=invalid_items)

    def _list_items_paginated(
        self,
        endpoint: str,
        source: str,
    ) -> MdblistFetchResult:
        """Fetch every cursor page from an MDBList API list endpoint."""

        items: list[MdblistListItem] = []
        invalid_items = 0
        pages = 0
        cursor: str | None = None
        seen_cursors: set[str] = set()

        while True:
            params: dict[str, str | int] = {
                **self.common_query_params,
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            response = self.session.get(
                endpoint,
                params=params,
            )
            self._raise_for_status(response, source)

            raw_payload = response.json()
            if not isinstance(raw_payload, dict):
                raise MdblistAPIError(f"MDBList {source} returned an invalid response")
            payload = cast(dict[str, object], raw_payload)

            raw_movies = payload.get("movies")
            raw_shows = payload.get("shows")
            if raw_movies is not None and not isinstance(raw_movies, list):
                raise MdblistAPIError(f"MDBList {source} returned invalid item buckets")
            if raw_shows is not None and not isinstance(raw_shows, list):
                raise MdblistAPIError(f"MDBList {source} returned invalid item buckets")

            movies = cast(list[object], raw_movies) if raw_movies is not None else []
            shows = cast(list[object], raw_shows) if raw_shows is not None else []

            page_items, page_invalid = self._validate_items(
                [*movies, *shows],
                source,
            )
            items.extend(page_items)
            invalid_items += page_invalid
            pages += 1

            raw_pagination = payload.get("pagination")
            pagination = (
                cast(dict[str, object], raw_pagination)
                if isinstance(raw_pagination, dict)
                else {}
            )
            raw_next_cursor = pagination.get("next_cursor")
            next_cursor = (
                str(raw_next_cursor)
                if raw_next_cursor is not None
                else response.headers.get("X-Next-Cursor")
            )
            raw_has_more = pagination.get("has_more")
            has_more = raw_has_more if isinstance(raw_has_more, bool) else None
            if has_more is None:
                has_more_header = response.headers.get("X-Has-More")
                if has_more_header is not None:
                    has_more = has_more_header.lower() == "true"

            if has_more is False or not next_cursor:
                break

            if next_cursor in seen_cursors:
                raise MdblistAPIError(f"MDBList {source} repeated a pagination cursor")

            seen_cursors.add(next_cursor)
            cursor = next_cursor

        return MdblistFetchResult(
            items=items,
            invalid_items=invalid_items,
            pages=pages,
        )

    @staticmethod
    def _api_items_endpoint(url: str) -> str | None:
        """Translate supported MDBList web URLs to cursor-paginated API routes."""

        parts = urlsplit(url)
        if parts.netloc.lower() not in {"mdblist.com", "www.mdblist.com"}:
            return None

        segments = [segment for segment in parts.path.split("/") if segment]
        if len(segments) < 3 or segments[0] != "lists":
            return None

        first = quote(segments[1], safe="")
        second = quote(segments[2], safe="")
        if segments[1] == "share":
            return f"/lists/share/{second}/items"
        if segments[1] == "official":
            return f"/lists/official/{second}/items"

        return f"/lists/{first}/{second}/items"

    @staticmethod
    def _safe_url(url: str) -> str:
        """Remove query parameters and fragments before including a URL in logs."""

        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def _raise_for_status(response: SmartResponse, source: str) -> None:
        if not response.ok:
            raise MdblistAPIError(
                f"MDBList {source} request failed with HTTP {response.status_code}"
            )

    @staticmethod
    def _validate_items(
        raw_items: list[object],
        source: str,
    ) -> tuple[list[MdblistListItem], int]:
        """Validate entries independently so one malformed record cannot drop a list."""

        items: list[MdblistListItem] = []
        invalid_items = 0

        for index, raw_item in enumerate(raw_items):
            try:
                items.append(MdblistListItem.model_validate(raw_item))
            except ValidationError as error:
                invalid_items += 1
                fields = sorted(
                    {
                        str(detail["loc"][0])
                        for detail in error.errors(include_url=False)
                        if detail["loc"]
                    }
                )
                logger.warning(
                    "Skipping invalid MDBList item {} from {} (fields: {})",
                    index,
                    source,
                    ", ".join(fields) or "unknown",
                )

        return items, invalid_items
