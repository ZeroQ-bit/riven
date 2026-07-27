from http import HTTPStatus
from typing import TypeGuard
from urllib.parse import urlparse


def is_streamable_http_url(url: str | None) -> TypeGuard[str]:
    """Return whether a URL can be handed to the HTTP streaming client."""

    if not url:
        return False

    parsed = urlparse(url)

    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def should_refresh_download_url(status_code: int, provider: str) -> bool:
    """Return whether a failed debrid URL should be regenerated once."""

    return status_code in (
        HTTPStatus.NOT_FOUND,
        HTTPStatus.GONE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ) or (provider.lower() == "torbox" and status_code == HTTPStatus.BAD_REQUEST)
