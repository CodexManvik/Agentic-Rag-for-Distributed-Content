import argparse
from pathlib import Path

from app.services.ingestion import ingest_pdf, ingest_web_page, reset_index


DEFAULT_URLS: list[str] = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://docs.langchain.com/oss/python/langchain/overview",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest public web pages and PDFs into Chroma")
    parser.add_argument("--urls", nargs="*", default=DEFAULT_URLS, help="List of public URLs")
    parser.add_argument("--pdf-dir", default="", help="Directory containing PDFs")
    parser.add_argument("--reset", action="store_true", help="Reset index before ingestion")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("Starting ingestion run")

    if args.reset:
        print("Resetting index")
        reset_index()

    totals = {
        "documents_processed": 0,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": 0,
    }

    for url in args.urls:
        print(f"Ingesting: {url}")
        stats = ingest_web_page(url)
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["skipped_duplicates"] += stats["skipped_duplicates"]
        totals["errors"] += len(stats["errors"])
        print(f"Completed: {url} -> {stats['chunks_added']} chunks")
        for err in stats["errors"]:
            print(f"Error: {err}")

    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        for pdf in pdf_files:
            print(f"Ingesting PDF: {pdf}")
            stats = ingest_pdf(str(pdf))
            totals["documents_processed"] += stats["documents_processed"]
            totals["chunks_added"] += stats["chunks_added"]
            totals["skipped_duplicates"] += stats["skipped_duplicates"]
            totals["errors"] += len(stats["errors"])
            print(f"Completed PDF: {pdf.name} -> {stats['chunks_added']} chunks")
            for err in stats["errors"]:
                print(f"Error: {err}")

    print(
        "Ingestion complete. "
        f"Documents: {totals['documents_processed']}, "
        f"Chunks added: {totals['chunks_added']}, "
        f"Duplicates skipped: {totals['skipped_duplicates']}, "
        f"Errors: {totals['errors']}"
    )


if __name__ == "__main__":
    main()
