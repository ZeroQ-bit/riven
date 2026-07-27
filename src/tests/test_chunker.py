import pytest
import trio_util
from kink import di

from program.services.streaming.chunker import Chunk, ChunkCacheNotifier, Chunker


class ChunkCacheNotifierStub:
    def get_emitter(self, *, chunk: Chunk) -> trio_util.AsyncBool:  # noqa: ARG002
        return trio_util.AsyncBool(False)


@pytest.fixture(autouse=True)
def configure_chunk_cache_notifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        di._services,
        ChunkCacheNotifier,
        ChunkCacheNotifierStub(),
    )


@pytest.mark.parametrize(
    ("file_size", "position"),
    [
        (2816, 2204),  # Body length is exactly two chunks.
        (3716, 3154),  # Final body chunk is only partially filled.
    ],
)
def test_range_crossing_footer_never_builds_chunk_inside_footer(
    file_size: int,
    position: int,
) -> None:
    chunker = Chunker(
        cache_key="media",
        chunk_size=1024,
        header_size=256,
        footer_size=512,
        file_size=file_size,
    )

    chunks = chunker.get_chunk_range(position=position, size=200).chunks

    assert chunks[-1] == chunker.footer_chunk
    assert chunks[-2].end == chunker.footer_chunk.start - 1
    assert all(chunk.start <= chunk.end for chunk in chunks)
