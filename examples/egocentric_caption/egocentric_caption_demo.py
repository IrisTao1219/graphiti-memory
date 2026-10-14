"""Minimal SRT-to-Graphiti demo using Ollama and Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType

DEFAULT_SRT = Path('data/dense_captions/A1_JAKE/DAY1/A1_JAKE_DAY1_11000000.srt')
TIME_PATTERN = re.compile(
    r'(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*'
    r'(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})'
)


@dataclass(frozen=True)
class Caption:
    index: int
    start_ms: int
    end_ms: int
    text: str


def timestamp_to_ms(value: str) -> int:
    hours, minutes, rest = value.replace('.', ',').split(':')
    seconds, milliseconds = rest.split(',')
    return int(hours) * 3_600_000 + int(minutes) * 60_000 + int(seconds) * 1_000 + int(milliseconds)


def parse_srt(path: Path) -> list[Caption]:
    """Parse ordinary UTF-8 SRT without adding a subtitle dependency."""
    raw = path.read_text(encoding='utf-8-sig').strip()
    captions: list[Caption] = []
    for block_number, block in enumerate(re.split(r'\r?\n\s*\r?\n', raw), start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f'Invalid SRT block {block_number}: expected index, time and text')

        try:
            index = int(lines[0])
        except ValueError as exc:
            raise ValueError(f'Invalid SRT index in block {block_number}: {lines[0]!r}') from exc

        match = TIME_PATTERN.fullmatch(lines[1])
        if match is None:
            raise ValueError(f'Invalid SRT time range in block {block_number}: {lines[1]!r}')

        start_ms = timestamp_to_ms(match.group('start'))
        end_ms = timestamp_to_ms(match.group('end'))
        if end_ms < start_ms:
            raise ValueError(f'SRT block {block_number} ends before it starts')

        captions.append(
            Caption(index=index, start_ms=start_ms, end_ms=end_ms, text=' '.join(lines[2:]))
        )

    return captions


def iso_duration(milliseconds: int) -> str:
    total_seconds, millis = divmod(milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'


def build_graphiti() -> Graphiti:
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    llm_model = os.getenv('OLLAMA_LLM_MODEL', 'qwen3:1.7b')
    embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'embeddinggemma:300m-qat-q4_0')
    embedding_dim = int(os.getenv('EMBEDDING_DIM', '768'))

    llm_config = LLMConfig(
        api_key='ollama',
        model=llm_model,
        small_model=llm_model,
        base_url=base_url,
        temperature=0,
    )
    llm_client = OpenAIGenericClient(
        config=llm_config,
        structured_output_mode=os.getenv('OLLAMA_STRUCTURED_OUTPUT', 'json_schema'),
    )

    return Graphiti(
        uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        user=os.getenv('NEO4J_USER', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD', 'testpass'),
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key='ollama',
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                base_url=base_url,
            )
        ),
        cross_encoder=OpenAIRerankerClient(config=llm_config, client=llm_client.client),
        max_coroutines=int(os.getenv('SEMAPHORE_LIMIT', '1')),
    )


async def ingest(graphiti: Graphiti, path: Path, group_id: str, recorded_at: datetime) -> None:
    captions = parse_srt(path)
    previous_episode_uuid: str | None = None
    saga_name = path.stem

    await graphiti.build_indices_and_constraints()
    print(f'Ingesting {len(captions)} captions into group {group_id!r}...')

    for position, caption in enumerate(captions, start=1):
        episode = {
            'observer': 'CameraWearer',
            'caption': caption.text,
            'video_time_start': iso_duration(caption.start_ms),
            'video_time_end': iso_duration(caption.end_ms),
            'source_file': path.name,
        }
        result = await graphiti.add_episode(
            group_id=group_id,
            name=f'{saga_name} caption {caption.index}',
            episode_body=json.dumps(episode, ensure_ascii=False),
            source=EpisodeType.json,
            source_description='第一视角视频的详细字幕；CameraWearer 表示佩戴相机的人',
            reference_time=recorded_at + timedelta(milliseconds=caption.start_ms),
            previous_episode_uuids=[previous_episode_uuid] if previous_episode_uuid else [],
            saga=saga_name,
            saga_previous_episode_uuid=previous_episode_uuid,
            custom_extraction_instructions=(
                '这是第一视角视频观察。CameraWearer 是同一个相机佩戴者。'
                '提取人物、物体、地点、动作和空间关系。把“我”规范为 CameraWearer。'
                '只有字幕明确描述状态结束或出现矛盾时，才使已有关系失效；'
                '后续字幕没有重复某状态，不代表该状态已经结束。'
            ),
        )
        previous_episode_uuid = result.episode.uuid
        print(f'[{position}/{len(captions)}] {iso_duration(caption.start_ms)} {caption.text}')


async def query(graphiti: Graphiti, group_id: str, question: str, limit: int) -> None:
    results = await graphiti.search(query=question, group_ids=[group_id], num_results=limit)
    print(f'\nQuestion: {question}')
    if not results:
        print('No facts found.')
        return

    for index, edge in enumerate(results, start=1):
        interval = f'{edge.valid_at or "?"} -> {edge.invalid_at or "now"}'
        print(f'{index}. {edge.fact}  [{interval}]')


def parse_recorded_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry-run', 'ingest', 'query', 'both'))
    parser.add_argument('--srt', type=Path, default=DEFAULT_SRT)
    parser.add_argument('--group-id', default=None)
    parser.add_argument('--question', default='我刚才拿着什么？')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument(
        '--recorded-at',
        default='1970-01-01T00:00:00+00:00',
        help='Recording start time in ISO-8601; SRT offsets are added to it.',
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    path = args.srt.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'SRT file not found: {path}')

    captions = parse_srt(path)
    if args.mode == 'dry-run':
        for caption in captions:
            print(json.dumps(asdict(caption), ensure_ascii=False))
        return

    # In current Graphiti versions, a Neo4j group_id also selects the database.
    # Neo4j Community exposes the default `neo4j` database only; the Saga keeps
    # individual recording sessions separate inside it.
    group_id = args.group_id or os.getenv('NEO4J_DATABASE', 'neo4j')
    graphiti = build_graphiti()
    try:
        if args.mode in ('ingest', 'both'):
            await ingest(graphiti, path, group_id, parse_recorded_at(args.recorded_at))
        if args.mode in ('query', 'both'):
            await query(graphiti, group_id, args.question, args.limit)
    finally:
        await graphiti.close()


def main() -> None:
    load_dotenv()
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    asyncio.run(async_main(build_parser().parse_args()))


if __name__ == '__main__':
    main()
