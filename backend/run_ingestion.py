import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import requests
import yaml

from app.services.compliance import is_url_allowlisted


DEFAULT_URLS: list[str] = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://docs.langchain.com/oss/python/langchain/overview",
]


class ReportError(TypedDict):
    source: str
    error: str


class IngestionReport(TypedDict):
    run_timestamp: str
    resource_pack_name: str
    resource_pack_path: str
    processed_urls: list[str]
    processed_pdfs: list[str]
    processed_docs: list[str]
    documents_processed: int
    chunks_added: int
    skipped_duplicates: int
    errors: list[ReportError]
    total_duration_seconds: float
    success_count: int
    failed_count: int
    source_results: list[dict[str, Any]]


def _resolve_report_paths(json_path: str) -> tuple[Path, Path]:
    json_file = Path(json_path)
    if not json_file.is_absolute():
        json_file = Path.cwd() / json_file
    md_file = json_file.with_suffix(".md")
    return json_file, md_file


def _load_resource_pack(path: str) -> dict[str, Any]:
    pack_path = Path(path)
    if not pack_path.exists():
        raise FileNotFoundError(f"Resource pack not found: {pack_path}")
    with pack_path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    if not isinstance(parsed, dict):
        raise ValueError("Resource pack must be a YAML object")
    return parsed


def _pick_sources(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str], str, str]:
    cli_urls = args.urls or []
    cli_pdfs: list[str] = []
    cli_pdf_urls = args.pdf_urls or []
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        cli_pdfs = [str(p) for p in sorted(pdf_dir.glob("*.pdf"))] if pdf_dir.exists() else []

    cli_docs = args.doc_paths or []

    if cli_urls or cli_pdfs or cli_pdf_urls or cli_docs:
        return cli_urls, cli_pdfs, cli_pdf_urls, cli_docs, "cli_overrides", ""

    if args.use_pack:
        pack = _load_resource_pack(args.resource_pack)
        pack_urls = pack.get("web_urls", [])
        pack_pdfs = pack.get("pdf_paths", [])
        pack_pdf_urls = pack.get("pdf_urls", [])
        pack_docs = pack.get("doc_paths", [])
        pack_base = Path(args.resource_pack).resolve().parent
        urls = [str(u) for u in pack_urls if isinstance(u, str)]
        pdfs = [str((pack_base / p).resolve()) for p in pack_pdfs if isinstance(p, str)]
        pdf_urls = [str(u) for u in pack_pdf_urls if isinstance(u, str)]
        docs = [str((pack_base / p).resolve()) for p in pack_docs if isinstance(p, str)]
        return urls, pdfs, pdf_urls, docs, str(pack.get("name", "unknown_pack")), str(Path(args.resource_pack))

    return list(DEFAULT_URLS), [], [], [], "built_in_defaults", ""


def _validate_sources(urls: list[str], pdfs: list[str], pdf_urls: list[str], docs: list[str]) -> tuple[list[ReportError], int]:
    errors: list[ReportError] = []
    valid_count = 0

    for url in urls:
        allowed, host = is_url_allowlisted(url)
        if not allowed:
            errors.append({"source": url, "error": f"Domain '{host}' is not allowlisted"})
            continue
        try:
            response = requests.get(url, timeout=15)
            if response.status_code >= 400:
                errors.append({"source": url, "error": f"HTTP {response.status_code}"})
            else:
                valid_count += 1
        except Exception as exc:
            errors.append({"source": url, "error": str(exc)})

    for pdf in pdfs:
        path = Path(pdf)
        if not path.exists():
            errors.append({"source": pdf, "error": "PDF file not found"})
        else:
            valid_count += 1

    for pdf_url in pdf_urls:
        allowed, host = is_url_allowlisted(pdf_url)
        if not allowed:
            errors.append({"source": pdf_url, "error": f"Domain '{host}' is not allowlisted"})
            continue
        try:
            response = requests.get(pdf_url, timeout=20)
            if response.status_code >= 400:
                errors.append({"source": pdf_url, "error": f"HTTP {response.status_code}"})
            elif "pdf" not in response.headers.get("content-type", "").lower() and not pdf_url.lower().endswith(".pdf"):
                errors.append({"source": pdf_url, "error": "URL does not appear to be a PDF"})
            else:
                valid_count += 1
        except Exception as exc:
            errors.append({"source": pdf_url, "error": str(exc)})

    for doc in docs:
        path = Path(doc)
        if not path.exists():
            errors.append({"source": doc, "error": "Document file not found"})
            continue
        if path.suffix.lower() not in {".md", ".txt", ".rst"}:
            errors.append({"source": doc, "error": "Unsupported document extension"})
            continue
        valid_count += 1

    return errors, valid_count


def _write_report(report: IngestionReport, report_json_path: str) -> None:
    json_path, md_path = _resolve_report_paths(report_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Ingestion Report",
        "",
        f"Run timestamp: {report['run_timestamp']}",
        f"Resource pack: {report['resource_pack_name']}",
        f"Resource pack path: {report['resource_pack_path']}",
        f"Total duration seconds: {report['total_duration_seconds']:.2f}",
        "",
        "## Summary",
        "",
        f"- documents_processed: {report['documents_processed']}",
        f"- chunks_added: {report['chunks_added']}",
        f"- skipped_duplicates: {report['skipped_duplicates']}",
        f"- success_count: {report['success_count']}",
        f"- failed_count: {report['failed_count']}",
        "",
        "## Processed URLs",
    ]
    lines.extend([f"- {u}" for u in report["processed_urls"]] or ["- none"])
    lines.append("")
    lines.append("## Processed PDFs")
    lines.extend([f"- {p}" for p in report["processed_pdfs"]] or ["- none"])
    lines.append("")
    lines.append("## Processed Docs")
    lines.extend([f"- {d}" for d in report["processed_docs"]] or ["- none"])
    lines.append("")
    lines.append("## Errors")
    if report["errors"]:
        for err in report["errors"]:
            lines.append(f"- {err['source']}: {err['error']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Source Results")
    if report.get("source_results"):
        for item in report["source_results"]:
            lines.append(
                "- "
                f"{item.get('source_type')} | {item.get('status')} | {item.get('domain')} | "
                f"chunks={item.get('chunks_added', 0)} | source={item.get('source')}"
            )
            if item.get("error"):
                lines.append(f"  error: {item['error']}")
    else:
        lines.append("- none")
    lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest public web pages and PDFs into Chroma")
    parser.add_argument("--urls", nargs="*", default=None, help="List of public URLs")
    parser.add_argument("--pdf-urls", nargs="*", default=None, help="List of public PDF URLs")
    parser.add_argument("--pdf-dir", default="", help="Directory containing PDFs")
    parser.add_argument("--doc-paths", nargs="*", default=None, help="List of local markdown/text documents")
    parser.add_argument("--reset", action="store_true", help="Reset index before ingestion")
    parser.add_argument(
        "--resource-pack",
        default="backend/resources/resource_pack.yaml",
        help="Path to resource pack YAML",
    )
    parser.add_argument(
        "--use-pack",
        action="store_true",
        help="Use resource pack URLs and PDF paths when CLI sources are not provided",
    )
    parser.add_argument(
        "--save-report",
        default="backend/resources/ingestion_report.json",
        help="Path to write ingestion report JSON (MD written alongside)",
    )
    parser.add_argument(
        "--validate-resources",
        action="store_true",
        help="Validate selected resource URLs/PDFs for allowlist and reachability",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    print("Starting ingestion run")

    try:
        urls, pdfs, pdf_urls, docs, pack_name, pack_path = _pick_sources(args)
    except Exception as exc:
        print(f"Failed to resolve sources: {exc}")
        sys.exit(1)

    if args.validate_resources:
        errors, valid_count = _validate_sources(urls, pdfs, pdf_urls, docs)
        report: IngestionReport = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resource_pack_name": pack_name,
            "resource_pack_path": pack_path,
            "processed_urls": urls,
            "processed_pdfs": [*pdfs, *pdf_urls],
            "processed_docs": docs,
            "documents_processed": valid_count,
            "chunks_added": 0,
            "skipped_duplicates": 0,
            "errors": errors,
            "total_duration_seconds": round(time.perf_counter() - start, 3),
            "success_count": valid_count,
            "failed_count": len(errors),
            "source_results": [],
        }
        _write_report(report, args.save_report)
        print(f"Validation finished. valid={valid_count}, errors={len(errors)}")
        sys.exit(0 if len(errors) == 0 else 1)

    from app.services.ingestion import ingest_pdf, ingest_text_file, ingest_web_page, reset_index

    if args.reset:
        print("Resetting index")
        reset_index()

    totals = {
        "documents_processed": 0,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": 0,
    }
    report_errors: list[ReportError] = []
    success_count = 0
    source_results: list[dict[str, Any]] = []

    for url in urls:
        print(f"Ingesting: {url}")
        stats = ingest_web_page(url)
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["skipped_duplicates"] += stats["skipped_duplicates"]
        totals["errors"] += len(stats["errors"])
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": url, "error": err})
            source_results.append(
                {
                    "source": url,
                    "source_type": "web",
                    "domain": urlparse(url).netloc.lower(),
                    "status": "failed",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": "; ".join(stats["errors"]),
                }
            )
        else:
            success_count += 1
            source_results.append(
                {
                    "source": url,
                    "source_type": "web",
                    "domain": urlparse(url).netloc.lower(),
                    "status": "ok",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": None,
                }
            )
        print(f"Completed: {url} -> {stats['chunks_added']} chunks")
        for err in stats["errors"]:
            print(f"Error: {err}")

    for pdf in pdfs:
        print(f"Ingesting PDF: {pdf}")
        stats = ingest_pdf(str(pdf))
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["skipped_duplicates"] += stats["skipped_duplicates"]
        totals["errors"] += len(stats["errors"])
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": pdf, "error": err})
            source_results.append(
                {
                    "source": pdf,
                    "source_type": "pdf_local",
                    "domain": "local",
                    "status": "failed",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": "; ".join(stats["errors"]),
                }
            )
        else:
            success_count += 1
            source_results.append(
                {
                    "source": pdf,
                    "source_type": "pdf_local",
                    "domain": "local",
                    "status": "ok",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": None,
                }
            )
        print(f"Completed PDF: {Path(pdf).name} -> {stats['chunks_added']} chunks")
        for err in stats["errors"]:
            print(f"Error: {err}")

    for pdf_url in pdf_urls:
        print(f"Ingesting PDF URL: {pdf_url}")
        try:
            allowed, host = is_url_allowlisted(pdf_url)
            if not allowed:
                raise ValueError(f"Domain '{host}' is not allowlisted")
            response = requests.get(pdf_url, timeout=25)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            stats = ingest_pdf(tmp_path)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as exc:
            stats = {
                "documents_processed": 1,
                "chunks_added": 0,
                "skipped_duplicates": 0,
                "errors": [str(exc)],
            }

        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["skipped_duplicates"] += stats["skipped_duplicates"]
        totals["errors"] += len(stats["errors"])
        domain = urlparse(pdf_url).netloc.lower()
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": pdf_url, "error": err})
            source_results.append(
                {
                    "source": pdf_url,
                    "source_type": "pdf_url",
                    "domain": domain,
                    "status": "failed",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": "; ".join(stats["errors"]),
                }
            )
        else:
            success_count += 1
            source_results.append(
                {
                    "source": pdf_url,
                    "source_type": "pdf_url",
                    "domain": domain,
                    "status": "ok",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": None,
                }
            )
        print(f"Completed PDF URL: {pdf_url} -> {stats['chunks_added']} chunks")
        for err in stats["errors"]:
            print(f"Error: {err}")

    for doc in docs:
        print(f"Ingesting document: {doc}")
        stats = ingest_text_file(str(doc))
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["skipped_duplicates"] += stats["skipped_duplicates"]
        totals["errors"] += len(stats["errors"])
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": doc, "error": err})
            source_results.append(
                {
                    "source": doc,
                    "source_type": "project_doc",
                    "domain": "local",
                    "status": "failed",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": "; ".join(stats["errors"]),
                }
            )
        else:
            success_count += 1
            source_results.append(
                {
                    "source": doc,
                    "source_type": "project_doc",
                    "domain": "local",
                    "status": "ok",
                    "documents_processed": stats["documents_processed"],
                    "chunks_added": stats["chunks_added"],
                    "skipped_duplicates": stats["skipped_duplicates"],
                    "error": None,
                }
            )
        print(f"Completed document: {Path(doc).name} -> {stats['chunks_added']} chunks")
        for err in stats["errors"]:
            print(f"Error: {err}")

    duration = round(time.perf_counter() - start, 3)
    report: IngestionReport = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resource_pack_name": pack_name,
        "resource_pack_path": pack_path,
        "processed_urls": urls,
        "processed_pdfs": [*pdfs, *pdf_urls],
        "processed_docs": docs,
        "documents_processed": totals["documents_processed"],
        "chunks_added": totals["chunks_added"],
        "skipped_duplicates": totals["skipped_duplicates"],
        "errors": report_errors,
        "total_duration_seconds": duration,
        "success_count": success_count,
        "failed_count": len(report_errors),
        "source_results": source_results,
    }
    _write_report(report, args.save_report)

    print(
        "Ingestion complete. "
        f"Documents: {totals['documents_processed']}, "
        f"Chunks added: {totals['chunks_added']}, "
        f"Duplicates skipped: {totals['skipped_duplicates']}, "
        f"Errors: {totals['errors']}, "
        f"Duration: {duration:.2f}s"
    )

    total_sources = len(urls) + len(pdfs) + len(pdf_urls) + len(docs)
    if total_sources > 0 and success_count == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
