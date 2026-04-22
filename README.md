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

It will load your documents, ask a few starter questions, and then keep prompting you for more questions.

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