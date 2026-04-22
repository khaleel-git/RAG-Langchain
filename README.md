# RAG Pipeline with Gemini and LangChain

This project builds a Retrieval-Augmented Generation (RAG) pipeline over PDF files or a job description.

- Embeddings: HuggingFace (`all-MiniLM-L6-v2`)
- Vector database: Chroma (persistent local index)
- LLM: Gemini (Google Generative AI)
- Inputs: PDF files from a folder (recursive) or a job description file

## Project Files

- `main.py`: RAG pipeline script
- `.env`: local secrets (Google API key)
- `.env.example`: template for environment variables
- `Documents/`: default PDF source folder
- `chroma_db/`: persisted vector index

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install langchain langchain-core langchain-chroma langchain-huggingface langchain-text-splitters sentence-transformers pypdf langchain-google-genai python-dotenv
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

If you want to ask repeated questions about a job description, create a text file such as `job_description.txt` and point the app to it.

## How It Works

1. Loads PDFs recursively from `Documents/` (or a custom folder).
2. Extracts text page-by-page using `pypdf`.
3. Splits text into chunks (`chunk_size=150`, `chunk_overlap=30`).
4. Creates embeddings and stores them in Chroma.
5. Retrieves top-k relevant chunks for each query.
6. Sends grounded prompt + context to Gemini.
7. If Gemini call fails, prints retrieved chunks as fallback.

Note: In this project, "training" means indexing documents for retrieval, not fine-tuning a model.

## Run Commands

Ask a question and then continue in interactive mode:

```bash
python main.py --pdf-dir Documents --interactive
```

If you want a one-off prompt only, pass the question directly:

```bash
python main.py --pdf-dir Documents "Summarize my profile"
```

Force reindex when files change:

```bash
python main.py --pdf-dir Documents --reindex "What skills are listed in my documents?"
```

Force reindex with custom chunking settings (great for classroom comparison demos):

```bash
python main.py --pdf-dir Documents --reindex --chunk-size 300 --chunk-overlap 60 "What skills are listed in my documents?"
```

Use a different PDF folder:

```bash
python main.py --pdf-dir /Users/khaleel/Documents --reindex "List all certifications"
```

Use text-only mode (skip PDF ingestion):

```bash
python main.py --text-only --source-file notes.txt "Summarize the notes"
```

Use a job description file and keep asking questions:

```bash
python main.py --job-description job_description.txt --interactive
```

Prompt for job description path or paste text interactively:

```bash
python main.py --job-description --interactive
```

You can also use a PDF job description:

```bash
python main.py --job-description job_description.pdf --interactive
```

## CLI Options

- `--pdf-dir`: PDF source directory (default: `Documents`)
- `--persist-dir`: Chroma index directory (default: `chroma_db`)
- `--reindex`: rebuild index from source documents
- `--text-only`: skip PDF ingestion and use text input
- `--source-file`: text file for `--text-only` mode
- `--job-description`: text or PDF file used as the active job-description context
- `--k`: number of retrieved chunks per query (default: `3`)
- `--chunk-size`: chunk size used during indexing (default: `150`)
- `--chunk-overlap`: chunk overlap used during indexing (default: `30`)
- `--env-file`: environment file path (default: `.env`)
- `--model`: override Gemini model (for example `gemini-2.5-flash`)
- `--interactive`: keep asking questions after the initial run

## Interactive Use

If you run:

```bash
python main.py --pdf-dir Documents --interactive
```

the script will answer the default starter questions and then keep prompting with `Q>` so you can ask more questions in the same session.

Type `exit`, `quit`, or press Enter on an empty line to stop.

## Chat Mode

Run the chat app from the project folder:

```bash
python main.py
```

It will load your documents and then keep prompting you for questions.

If you want to start directly in interactive mode without the starter questions:

```bash
python main.py --interactive
```

## Job Description Workflow

Use this mode when you want the assistant to help tailor your answers to a specific role.

1. Save the job description as `job_description.txt` or `job_description.pdf`.
2. Run the app with `--job-description`.
3. Ask follow-up questions such as:
	1. What skills are required?
	2. Which parts of my profile should I highlight?
	3. How should I tailor my resume for this role?

## Troubleshooting

If you see `GOOGLE_API_KEY is not set`:

1. Confirm `.env` exists in project root.
2. Confirm `GOOGLE_API_KEY` is present and valid.
3. Re-run from project root so `.env` is discovered.

If you see `429 RESOURCE_EXHAUSTED`:

1. Check Gemini API quota and billing for your Google project.
2. Wait for quota reset window if rate-limited.
3. Retry with smaller query volume.

## Next Improvements

1. Add metadata filtering (source filename/page constraints).
2. Add citation-style answers with source + page references.
3. Add a lightweight API or UI on top of `main.py`.

## Teaching Guide: Full Pipeline Internals

This section explains exactly how your code implements a production-style RAG flow.

### 1) Data ingestion stage

The pipeline starts by loading source content into LangChain `Document` objects:

- PDF folder mode: scans recursively, extracts page text with `pypdf`, and stores metadata (`source`, `page`).
- Job description mode: can read `.txt`, `.pdf`, or interactive pasted text.

Why this matters:

- `Document` is the canonical unit passed to splitters, embedders, and vector stores.
- Metadata is crucial for traceability and later source citations.

### 2) Chunking stage

The code uses `RecursiveCharacterTextSplitter` with:

- `chunk_size = 150`
- `chunk_overlap = 30`

Why chunking exists:

- LLM context windows are finite.
- Retrieval quality is better with semantically focused chunks.

Why overlap exists:

- Important sentence boundaries often fall between chunks.
- Overlap preserves nearby context so retrieval does not lose meaning at boundaries.

Approximate chunk count for a document of length `N` with size `s` and overlap `o`:

$$
	ext{chunks} \approx \left\lceil\frac{N - s}{s - o}\right\rceil + 1
$$

### 3) Embedding stage

Each chunk is converted into a dense vector using:

- model: `sentence-transformers/all-MiniLM-L6-v2`

Conceptually:

- Chunk text -> embedding model -> vector in high-dimensional space.
- Similar meaning => vectors are close by distance/similarity metrics.

Why this model is used here:

- Fast local inference.
- Good quality/latency balance for a teaching and prototyping setup.

### 4) Vector database stage (Chroma)

The vector DB stores:

- chunk text
- chunk embedding
- metadata

At query time it does:

1. Embed query text with the same embedding model.
2. Compute similarity to stored vectors.
3. Return top-k nearest chunks.

That is your retrieval step before generation.

### 5) Why SQLite appears

In local persistent mode (`persist_directory` set), Chroma keeps local storage artifacts in that folder.
Depending on Chroma version/configuration, this can include SQLite-based metadata/state files and additional index files.

Important teaching point:

- Chroma is the vector DB interface.
- SQLite is often part of local persistence internals, not your retrieval algorithm itself.

### 6) Generation stage (Gemini)

After retrieval, the code constructs a grounded prompt:

- system instruction enforces context-only answering.
- human message includes user question + concatenated retrieved context.

Then Gemini generates the answer.

If Gemini call fails (quota/network/model issue), the app falls back to printing retrieved chunks.

### 7) End-to-end query flow

1. User question arrives.
2. Retriever gets top-k chunks from Chroma.
3. Prompt is assembled with question + retrieved context.
4. Gemini answers from that context.
5. Response is shown in terminal chat loop.

### 8) Key configuration knobs and trade-offs

- `chunk_size`:
	- smaller: more precise retrieval, more chunks, more storage and indexing time.
	- larger: broader context per chunk, but less precise retrieval.
- `chunk_overlap`:
	- higher: better boundary continuity, but more duplicated content.
- `k` (top-k retrieval):
	- higher: more context coverage, more prompt tokens/cost.
	- lower: faster and cheaper, risk of missing evidence.

In this codebase, these knobs are exposed directly as CLI flags:

- `--chunk-size`
- `--chunk-overlap`
- `--k`

### 9) Gemini limits you should teach

Limits vary by model and account tier and can change over time.
In practice you should consider four categories:

1. Context/window token limits per request.
2. Requests per minute/day quotas.
3. Input/output token quotas per minute/day.
4. Model availability by API version and account.

Operational recommendation:

- Always design for graceful failure (already implemented with retrieval fallback).
- Keep prompts compact and retrieval targeted.
- Monitor quota errors (`429 RESOURCE_EXHAUSTED`) and model errors (`404 NOT_FOUND`).

### 10) Embedding and chunking practical limits

- Very small chunks lose semantics.
- Very large chunks reduce retrieval precision and increase prompt cost.
- High overlap inflates index size.
- Poor source text extraction (OCR/noisy PDFs) degrades retrieval quality no matter which LLM is used.

### 11) Classroom demo sequence

1. Run with `--reindex` once.
2. Ask factual questions that are explicitly in documents.
3. Ask out-of-context questions and observe grounded behavior.
4. Lower and raise `k` to show answer quality/cost trade-offs.
5. Change chunk settings and compare retrieval outputs.