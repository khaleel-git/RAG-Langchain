# NVIDIA ATS + Cover Letter Pipeline

This project builds a local RAG pipeline over your resume PDFs and then:

1. Scores your profile against `job_description.txt`
2. Saves a formatted report to `match.md`
3. Generates a cover letter PDF if score and language rules pass
4. Copies the generated PDF to your target Downloads folder

The LLM provider is NVIDIA only.

## Project Files

- `main.py`: end-to-end pipeline
- `.env`: secrets and model settings
- `.env.example`: environment template
- `Documents/`: source PDF folder (recursive scan)
- `job_description.txt`: job description input
- `chroma_db/`: rebuilt vector index
- `match.md`: ATS result output
- `Khaleel_CoverLetter.pdf`: generated cover letter

## Setup

1. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
```

Then edit `.env` with your NVIDIA keys/models.

## Run

Default run:

```bash
python main.py
```

Force provider explicitly:

```bash
python main.py --provider nvidia
```

Use specific models at runtime:

```bash
python main.py \
  --ats-model nvidia/nemotron-3-super-120b-a12b \
  --cover-model nvidia/nvidia-nemotron-nano-9b-v2
```

Use a different job description file:

```bash
python main.py --job-description role_description.txt
```

## CLI Options

- `--job-description`: job description file path (`.txt` or `.pdf`)
- `--pdf-dir`: folder containing profile PDFs
- `--persist-dir`: Chroma directory (rebuilt on each run)
- `--match-file`: markdown output path
- `--cover-letter-output`: PDF output path
- `--k`: retrieved chunk count
- `--chunk-size`: indexing chunk size
- `--chunk-overlap`: indexing chunk overlap
- `--env-file`: env file path
- `--model`: global model override fallback
- `--ats-model`: ATS model override
- `--cover-model`: cover-letter model override
- `--provider`: `nvidia` or `auto` (auto resolves to nvidia)

## Notes

- Cover letter is generated only when score is above the configured threshold and language checks pass.
- ATS and cover calls can use different NVIDIA API keys:
  - `NVIDIA_ATS_API_KEY`
  - `NVIDIA_COVER_API_KEY`
  - If missing, both fall back to `NVIDIA_API_KEY`.
