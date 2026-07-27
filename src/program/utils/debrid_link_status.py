from http import HTTPStatus


def should_refresh_download_url(status_code: int, provider: str) -> bool:
    """Return whether a failed debrid URL should be regenerated once."""

    return status_code in (
        HTTPStatus.NOT_FOUND,
        HTTPStatus.GONE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ) or (provider.lower() == "torbox" and status_code == HTTPStatus.BAD_REQUEST)
