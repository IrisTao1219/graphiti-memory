# Egocentric caption demo

This demo imports timestamped SRT captions into Graphiti with a local Ollama LLM and
embedding model, then searches the resulting temporal facts in Neo4j.

## Prerequisites

Start Ollama and ensure the two models exist:

```bash
ollama serve
ollama pull qwen3:1.7b
ollama pull embeddinggemma:300m-qat-q4_0
```

Start Neo4j. The repository's test credentials are used by default:

```bash
docker run --name graphiti-neo4j --rm -d \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -e 'NEO4J_PLUGINS=["apoc"]' \
  neo4j:5.26-community
```

If Neo4j Desktop is already in use, set its credentials in a repository-root `.env`
instead. All supported settings are shown in `.env.example`.

Keep `NEO4J_DATABASE=neo4j` when using Neo4j Community. In this Graphiti version,
`group_id` also routes to a Neo4j database; recording sessions are separated by Saga.

## Run

From the repository root, verify SRT parsing without Ollama or Neo4j:

```bash
uv run python examples/egocentric_caption/egocentric_caption_demo.py dry-run
```

Ingest the default local SRT and immediately query it:

```bash
uv run python examples/egocentric_caption/egocentric_caption_demo.py both \
  --question '我刚才拿着什么？'
```

Run `ingest`/`both` once per fresh database for this minimal demo. Re-running ingestion
creates another set of Episode provenance records; use `query` for subsequent questions.

Query an already-ingested session without ingesting it again:

```bash
uv run python examples/egocentric_caption/egocentric_caption_demo.py query \
  --question '我面向哪里？'
```

For a different file, pass `--srt path/to/file.srt`. Use `--recorded-at` when the
recording's real start time is known, for example
`--recorded-at 2026-10-14T11:00:00+08:00`.

The first ingestion is intentionally sequential (`SEMAPHORE_LIMIT=1`) because small local
models are more reliable with one structured-output request at a time.
