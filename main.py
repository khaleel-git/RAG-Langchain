from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install with:\n"
        "pip install langchain langchain-core "
        "langchain-chroma langchain-huggingface "
        "langchain-text-splitters sentence-transformers pypdf "
        "langchain-google-genai python-dotenv\n"
        f"Import error: {exc}"
    )

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_CHAIN_AVAILABLE = True
except ImportError:
    GEMINI_CHAIN_AVAILABLE = False
    ChatGoogleGenerativeAI: Any = None

DEFAULT_TEXT = """
Khaleel Ahmad is an experienced Data Engineer and cloud-native practitioner.
He has hands-on expertise with Python, SQL, Apache Airflow, Kubernetes, and data pipelines.
He has worked on production-scale ingestion systems and automation workflows.
He focuses on reliable ETL/ELT architecture, observability, and performance optimization.
He is interested in Retrieval-Augmented Generation (RAG), LLM applications, and MLOps.
""".strip()

DEFAULT_QUERIES = [
    "Who is Khaleel Ahmad?",
    "What technologies does Khaleel work with?",
    "What are Khaleel's key interests?",
]

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def load_environment(env_file: str) -> None:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)


def find_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
    )


def load_pdf_documents(pdf_dir: Path) -> List[Document]:
    pdf_files = find_pdfs(pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files found under: {pdf_dir}")

    docs: List[Document] = []
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_idx,
                    },
                )
            )

    if not docs:
        raise ValueError(f"PDF files found, but no extractable text under: {pdf_dir}")

    return docs


def count_pdf_files(pdf_dir: Path) -> int:
    return len(find_pdfs(pdf_dir))


def load_text(source_file: Optional[str]) -> str:
    if not source_file:
        return DEFAULT_TEXT

    with open(source_file, "r", encoding="utf-8") as f:
        data = f.read().strip()

    if not data:
        raise ValueError(f"Source file is empty: {source_file}")

    return data


def load_job_description(job_description_path: Optional[str]) -> Optional[List[Document]]:
    if not job_description_path:
        return None

    path = Path(job_description_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Job description file not found: {path}")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        docs: List[Document] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_idx, "type": "job_description"},
                )
            )
        if not docs:
            raise ValueError(f"Job description PDF has no extractable text: {path}")
        return docs

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Job description file is empty: {path}")

    return [Document(page_content=text, metadata={"source": str(path), "type": "job_description"})]


def build_vectorstore(docs: List[Document], persist_directory: Optional[str] = None) -> Chroma:
    if not docs:
        raise ValueError("No documents provided for vectorstore creation.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_or_build_vectorstore(
    docs: List[Document],
    persist_directory: str,
    force_reindex: bool,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not force_reindex and Path(persist_directory).exists():
        return Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    return build_vectorstore(docs=docs, persist_directory=persist_directory)


def build_rag_components(model: Optional[str]):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounded assistant. Use only the provided context. "
                "If the answer is not in context, say you do not know.",
            ),
            (
                "human",
                "Question: {input}\n\nContext:\n{context}",
            ),
        ]
    )

    if not GEMINI_CHAIN_AVAILABLE:
        return None, "Gemini dependencies are missing. Install langchain-google-genai."
    if not os.getenv("GOOGLE_API_KEY"):
        return None, "GOOGLE_API_KEY is not set. Add it in .env."

    selected_model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0)
    return (llm, prompt), None


def answer_with_context(query: str, retriever, rag_components) -> str:
    llm, prompt = rag_components
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.format_messages(input=query, context=context)
    response = llm.invoke(messages)
    return str(response.content)


def print_retrieved_chunks(query: str, retriever, reason: Optional[str] = None) -> None:
    docs = retriever.invoke(query)
    print(f"\nQ: {query}")
    reason_text = reason or "LLM is unavailable"
    print(f"A: {reason_text}; showing retrieved context instead.")
    for idx, doc in enumerate(docs, start=1):
        content = " ".join(doc.page_content.split())
        wrapped = textwrap.fill(content, width=100)
        print(f"\n[Chunk {idx}] source={doc.metadata.get('source', 'unknown')}")
        print(wrapped)


def run_queries(queries: List[str], retriever, rag_components, fallback_reason: Optional[str]) -> None:
    for q in queries:
        if rag_components is None:
            print_retrieved_chunks(q, retriever, fallback_reason)
            continue

        try:
            answer = answer_with_context(q, retriever, rag_components)
            print(f"\nQ: {q}")
            print(f"A: {answer}")
        except Exception as exc:
            print(f"\nQ: {q}")
            print(f"A: Failed to run Gemini generation ({exc}). Falling back to retrieved chunks.")
            print_retrieved_chunks(q, retriever, "LLM request failed")


def interactive_loop(retriever, rag_components, fallback_reason: Optional[str]) -> None:
    print("\nInteractive mode. Type your question and press Enter.")
    print("Type 'exit', 'quit', or press Enter on an empty line to stop.\n")

    while True:
        try:
            query = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in {"exit", "quit"}:
            break

        run_queries([query], retriever, rag_components, fallback_reason)


def build_documents_for_session(args: argparse.Namespace) -> List[Document]:
    if args.job_description:
        docs = load_job_description(args.job_description)
        assert docs is not None
        print(f"Loaded job description from: {args.job_description}")
        return docs

    if args.text_only:
        raw_text = load_text(args.source_file)
        return [Document(page_content=raw_text, metadata={"source": "profile_text"})]

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = Path.cwd() / pdf_dir

    docs = load_pdf_documents(pdf_dir)
    pdf_count = count_pdf_files(pdf_dir)
    print(f"Loaded {len(docs)} PDF pages from {pdf_count} PDF files at: {pdf_dir}")
    return docs


def start_chat_session(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = argparse.Namespace(
            queries=[],
            source_file=None,
            pdf_dir="Documents",
            persist_dir="chroma_db",
            reindex=False,
            text_only=False,
            job_description=None,
            k=3,
            env_file=".env",
            model=None,
            interactive=True,
        )

    load_environment(args.env_file)
    docs = build_documents_for_session(args)

    persist_dir = args.persist_dir
    if not Path(persist_dir).is_absolute():
        persist_dir = str(Path.cwd() / persist_dir)

    vectorstore = load_or_build_vectorstore(
        docs=docs,
        persist_directory=persist_dir,
        force_reindex=args.reindex,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    rag_components, fallback_reason = build_rag_components(args.model)

    if getattr(args, "queries", None):
        run_queries(args.queries, retriever, rag_components, fallback_reason)

    interactive_loop(retriever, rag_components, fallback_reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG pipeline using LangChain + Chroma")
    parser.add_argument(
        "queries",
        nargs="*",
        help="Questions to ask the RAG pipeline. If omitted, built-in sample questions are used.",
    )
    parser.add_argument(
        "--source-file",
        dest="source_file",
        help="Optional path to a text file to index instead of the built-in sample text.",
    )
    parser.add_argument(
        "--pdf-dir",
        dest="pdf_dir",
        default="Documents",
        help="Project-relative folder containing PDFs to index recursively (default: Documents).",
    )
    parser.add_argument(
        "--persist-dir",
        dest="persist_dir",
        default="chroma_db",
        help="Directory used to persist Chroma index (default: chroma_db).",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the vector index from source documents.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use built-in text/source-file only, skipping PDF ingestion.",
    )
    parser.add_argument(
        "--job-description",
        dest="job_description",
        help="Path to a job description file (.txt or .pdf) to use as the active context.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve for each query (default: 3).",
    )
    parser.add_argument(
        "--env-file",
        dest="env_file",
        default=".env",
        help="Path to .env file that contains GOOGLE_API_KEY (default: .env).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Gemini model name (for example: gemini-1.5-flash).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep accepting questions after the initial run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive or (not args.queries and sys.stdin.isatty()):
        start_chat_session(args)
        return

    load_environment(args.env_file)

    if args.job_description:
        docs = load_job_description(args.job_description)
        assert docs is not None
        print(f"Loaded job description from: {args.job_description}")
    elif args.text_only:
        raw_text = load_text(args.source_file)
        docs = [Document(page_content=raw_text, metadata={"source": "profile_text"})]
    else:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_absolute():
            pdf_dir = Path.cwd() / pdf_dir

        docs = load_pdf_documents(pdf_dir)
        pdf_count = count_pdf_files(pdf_dir)
        print(f"Loaded {len(docs)} PDF pages from {pdf_count} PDF files at: {pdf_dir}")

    persist_dir = args.persist_dir
    if not Path(persist_dir).is_absolute():
        persist_dir = str(Path.cwd() / persist_dir)

    vectorstore = load_or_build_vectorstore(
        docs=docs,
        persist_directory=persist_dir,
        force_reindex=args.reindex,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    rag_components, fallback_reason = build_rag_components(args.model)

    if args.job_description:
        queries = args.queries if args.queries else [
            "What is this job description asking for?",
            "What are the top required skills?",
            "What should I emphasize in my application?",
        ]
    else:
        queries = args.queries if args.queries else DEFAULT_QUERIES

    run_queries(queries, retriever, rag_components, fallback_reason)

    if args.interactive or (not args.queries and sys.stdin.isatty()):
        interactive_loop(retriever, rag_components, fallback_reason)


if __name__ == "__main__":
    main()
