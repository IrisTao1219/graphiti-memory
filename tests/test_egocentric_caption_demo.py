import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import AsyncMock

DEMO_PATH = (
    Path(__file__).parents[1] / 'examples' / 'egocentric_caption' / 'egocentric_caption_demo.py'
)
SPEC = spec_from_file_location('egocentric_caption_demo', DEMO_PATH)
assert SPEC is not None and SPEC.loader is not None
DEMO = module_from_spec(SPEC)
sys.modules[SPEC.name] = DEMO
SPEC.loader.exec_module(DEMO)


def test_parse_srt_handles_chinese_and_multiline_text(tmp_path: Path) -> None:
    path = tmp_path / 'sample.srt'
    path.write_text(
        '1\n00:09:43,500 --> 00:09:44,800\n我拿着手机\n\n'
        '2\n00:09:45.000 --> 00:09:47.366\n我面向餐桌\n拿着手机\n',
        encoding='utf-8',
    )

    captions = DEMO.parse_srt(path)

    assert captions == [
        DEMO.Caption(index=1, start_ms=583_500, end_ms=584_800, text='我拿着手机'),
        DEMO.Caption(index=2, start_ms=585_000, end_ms=587_366, text='我面向餐桌 拿着手机'),
    ]


async def test_ingest_lets_graphiti_create_episode_uuid(tmp_path: Path) -> None:
    path = tmp_path / 'sample.srt'
    path.write_text(
        '1\n00:00:01,000 --> 00:00:02,000\n我拿着手机\n',
        encoding='utf-8',
    )
    graphiti = AsyncMock()
    graphiti.add_episode.return_value.episode.uuid = 'created-by-graphiti'

    await DEMO.ingest(
        graphiti,
        path,
        'neo4j',
        DEMO.datetime(2026, 1, 1, tzinfo=DEMO.timezone.utc),
    )

    call = graphiti.add_episode.await_args
    assert 'uuid' not in call.kwargs
