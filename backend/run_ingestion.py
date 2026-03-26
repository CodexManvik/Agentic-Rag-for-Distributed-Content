from app.services.ingestion import ingest_web_page


PUBLIC_URLS: list[str] = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://docs.langchain.com/oss/python/langchain/overview",
]


def main() -> None:
    print("Starting ingestion run")
    total_chunks = 0

    for url in PUBLIC_URLS:
        print(f"Ingesting: {url}")
        try:
            chunk_count = ingest_web_page(url)
            total_chunks += chunk_count
            print(f"Completed: {url} -> {chunk_count} chunks")
        except Exception as exc:
            print(f"Failed: {url} -> {exc}")

    print(f"Ingestion complete. Total chunks ingested: {total_chunks}")


if __name__ == "__main__":
    main()
