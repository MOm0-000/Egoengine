from pathlib import Path

from video_to_spider.manifest import stage_cache_key


def test_stage_cache_key_changes_with_config_and_content(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"one")
    first = stage_cache_key("ingest", {"start": 0}, [source])
    assert first == stage_cache_key("ingest", {"start": 0}, [source])
    assert first != stage_cache_key("ingest", {"start": 1}, [source])
    source.write_bytes(b"two")
    assert first != stage_cache_key("ingest", {"start": 0}, [source])

