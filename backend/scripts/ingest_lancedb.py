"""
LanceDB data ingestion script.

Ingests documents from various sources (web pages, PDFs, text files) into LanceDB
vector store for RAG applications. Supports multi-knowledge-base organization
and provides detailed ingestion reporting.

Usage:
    python backend/scripts/ingest_lancedb.py --reset --use-pack
    python backend/scripts/ingest_lancedb.py --urls https://example.com --knowledge-base public
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger

# Add backend directory to Python path for app imports
_script_path = Path(__file__).resolve()
_backend_dir = _script_path.parent.parent  # backend/scripts -> backend
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# LangChain imports for document processing
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install langchain langchain-community sentence-transformers")
    sys.exit(1)

# Import project modules
try:
    import yaml
    from app.services.compliance import is_url_allowlisted
    from app.vectorstore.lancedb_store import LanceDBVectorStore
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Backend dir: {_backend_dir}")
    sys.exit(1)


# --- Type definitions ---

class IngestionStats(TypedDict):
    """Statistics for a single document ingestion."""
    documents_processed: int
    chunks_added: int
    errors: list[str]


class ReportError(TypedDict):
    """Error entry in ingestion report."""
    source: str
    error: str


class IngestionReport(TypedDict):
    """Complete ingestion run report."""
    run_timestamp: str
    resource_pack_name: str
    resource_pack_path: str
    knowledge_base: str
    processed_urls: list[str]
    processed_pdfs: list[str]
    processed_docs: list[str]
    documents_processed: int
    chunks_added: int
    errors: list[ReportError]
    total_duration_seconds: float
    success_count: int
    failed_count: int
    source_results: list[dict[str, Any]]


# --- Global configuration ---

DEFAULT_URLS: list[str] = [
    "https://python.langchain.com/docs/introduction/",
    "https://docs.langchain.com/oss/python/langchain/overview",
]

# Default LanceDB path
DEFAULT_LANCEDB_PATH = "backend/lancedb_data"

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitter configuration (from resource pack defaults)
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 200


# --- Helper functions ---

def _timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _ensure_allowed_domain(url: str, public_sources_only: bool = True) -> None:
    """
    Check if URL domain is allowlisted.
    
    Args:
        url: URL to check
        public_sources_only: Whether to enforce public sources only
        
    Raises:
        ValueError: If domain is not allowlisted
    """
    if not public_sources_only:
        return
        
    allowed, host = is_url_allowlisted(url)
    if not allowed:
        raise ValueError(f"Domain '{host}' is not allowlisted for public ingestion")


def _create_text_splitter(chunk_size: int = DEFAULT_CHUNK_SIZE, 
                         chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """Create text splitter with configured chunk size."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def _load_resource_pack(path: str) -> dict[str, Any]:
    """Load resource pack YAML file."""
    pack_path = Path(path)
    if not pack_path.exists():
        raise FileNotFoundError(f"Resource pack not found: {pack_path}")
    
    with pack_path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    
    if not isinstance(parsed, dict):
        raise ValueError("Resource pack must be a YAML object")
    
    return parsed


def _resolve_report_paths(json_path: str) -> tuple[Path, Path]:
    """Resolve JSON and markdown report paths."""
    json_file = Path(json_path)
    if not json_file.is_absolute():
        json_file = Path.cwd() / json_file
    md_file = json_file.with_suffix(".md")
    return json_file, md_file


# --- Ingestion functions ---

def ingest_web_page(
    url: str,
    vector_store: LanceDBVectorStore,
    knowledge_base: str,
    text_splitter: RecursiveCharacterTextSplitter,
    public_sources_only: bool = True,
) -> IngestionStats:
    """
    Ingest a web page into LanceDB.
    
    Args:
        url: Web page URL
        vector_store: LanceDB vector store instance
        knowledge_base: Knowledge base name/identifier
        text_splitter: Text splitter for chunking
        public_sources_only: Whether to enforce domain allowlist
        
    Returns:
        IngestionStats with processing results
    """
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "errors": [],
    }
    
    try:
        _ensure_allowed_domain(url, public_sources_only)
        
        # Fetch and parse web page
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        # Extract title and text
        title = (soup.title.string or url).strip() if soup.title else url
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        
        if not text:
            stats["errors"].append("No text content extracted")
            return stats
        
        # Create document with metadata
        document = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "web",
                "title": title,
                "url": url,
                "knowledge_base": knowledge_base,
                "ingested_at": _timestamp(),
            },
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([document])
        
        # Add to vector store
        vector_store.add_documents(chunks, knowledge_base=knowledge_base)
        stats["chunks_added"] = len(chunks)
        
        logger.info(f"Ingested {url} -> {len(chunks)} chunks")
        
    except Exception as exc:
        stats["errors"].append(str(exc))
        logger.error(f"Failed to ingest {url}: {exc}")
    
    return stats


def ingest_pdf(
    file_path: str,
    vector_store: LanceDBVectorStore,
    knowledge_base: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> IngestionStats:
    """
    Ingest a PDF file into LanceDB.
    
    Args:
        file_path: Path to PDF file
        vector_store: LanceDB vector store instance
        knowledge_base: Knowledge base name/identifier
        text_splitter: Text splitter for chunking
        
    Returns:
        IngestionStats with processing results
    """
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "errors": [],
    }
    
    path = Path(file_path)
    
    try:
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        # Use PyPDF2 or PyMuPDF for PDF extraction
        try:
            import pymupdf  # fitz
            
            doc = pymupdf.open(str(path))
            text_parts = []
            
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            
            doc.close()
            text = "\n\n".join(text_parts)
            
        except ImportError:
            # Fallback to PyPDF2
            from pypdf import PdfReader
            
            reader = PdfReader(str(path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text}")
            
            text = "\n\n".join(text_parts)
        
        if not text.strip():
            stats["errors"].append("No text content extracted from PDF")
            return stats
        
        # Create document with metadata
        document = Document(
            page_content=text,
            metadata={
                "source": path.name,
                "source_type": "pdf",
                "file_path": str(path),
                "knowledge_base": knowledge_base,
                "ingested_at": _timestamp(),
            },
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([document])
        
        # Add to vector store
        vector_store.add_documents(chunks, knowledge_base=knowledge_base)
        stats["chunks_added"] = len(chunks)
        
        logger.info(f"Ingested {path.name} -> {len(chunks)} chunks")
        
    except Exception as exc:
        stats["errors"].append(str(exc))
        logger.error(f"Failed to ingest {file_path}: {exc}")
    
    return stats


def ingest_text_file(
    file_path: str,
    vector_store: LanceDBVectorStore,
    knowledge_base: str,
    text_splitter: RecursiveCharacterTextSplitter,
) -> IngestionStats:
    """
    Ingest a text file into LanceDB.
    
    Args:
        file_path: Path to text file
        vector_store: LanceDB vector store instance
        knowledge_base: Knowledge base name/identifier
        text_splitter: Text splitter for chunking
        
    Returns:
        IngestionStats with processing results
    """
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "errors": [],
    }
    
    path = Path(file_path)
    
    try:
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")
        
        # Read file content
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            stats["errors"].append("No text content in file")
            return stats
        
        # Create document with metadata
        document = Document(
            page_content=text,
            metadata={
                "source": path.name,
                "source_type": "document",
                "file_path": str(path),
                "file_type": path.suffix,
                "knowledge_base": knowledge_base,
                "ingested_at": _timestamp(),
            },
        )
        
        # Split into chunks
        chunks = text_splitter.split_documents([document])
        
        # Add to vector store
        vector_store.add_documents(chunks, knowledge_base=knowledge_base)
        stats["chunks_added"] = len(chunks)
        
        logger.info(f"Ingested {path.name} -> {len(chunks)} chunks")
        
    except Exception as exc:
        stats["errors"].append(str(exc))
        logger.error(f"Failed to ingest {file_path}: {exc}")
    
    return stats


# --- Report generation ---

def _write_report(report: IngestionReport, report_json_path: str) -> None:
    """Write ingestion report to JSON and Markdown files."""
    json_path, md_path = _resolve_report_paths(report_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Write Markdown
    lines = [
        "# LanceDB Ingestion Report",
        "",
        f"Run timestamp: {report['run_timestamp']}",
        f"Resource pack: {report['resource_pack_name']}",
        f"Resource pack path: {report['resource_pack_path']}",
        f"Knowledge base: {report['knowledge_base']}",
        f"Total duration: {report['total_duration_seconds']:.2f}s",
        "",
        "## Summary",
        "",
        f"- Documents processed: {report['documents_processed']}",
        f"- Chunks added: {report['chunks_added']}",
        f"- Success count: {report['success_count']}",
        f"- Failed count: {report['failed_count']}",
        "",
        "## Processed URLs",
    ]
    
    lines.extend([f"- {u}" for u in report["processed_urls"]] or ["- none"])
    lines.append("")
    lines.append("## Processed PDFs")
    lines.extend([f"- {p}" for p in report["processed_pdfs"]] or ["- none"])
    lines.append("")
    lines.append("## Processed Documents")
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
                f"- {item.get('source_type')} | {item.get('status')} | "
                f"chunks={item.get('chunks_added', 0)} | source={item.get('source')}"
            )
            if item.get("error"):
                lines.append(f"  error: {item['error']}")
    else:
        lines.append("- none")
    
    lines.append("")
    
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report written to {json_path} and {md_path}")


# --- CLI handling ---

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into LanceDB vector store"
    )
    
    # Data sources
    parser.add_argument("--urls", nargs="*", default=None, help="List of public URLs")
    parser.add_argument("--pdf-urls", nargs="*", default=None, help="List of PDF URLs")
    parser.add_argument("--pdf-dir", default="", help="Directory containing PDFs")
    parser.add_argument("--doc-paths", nargs="*", default=None, help="List of local documents")
    
    # Configuration
    parser.add_argument(
        "--knowledge-base",
        default="default",
        help="Knowledge base identifier (default: default)",
    )
    parser.add_argument(
        "--lancedb-path",
        default=DEFAULT_LANCEDB_PATH,
        help=f"Path to LanceDB directory (default: {DEFAULT_LANCEDB_PATH})",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"HuggingFace embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Text chunk size (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap size (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    
    # Resource pack
    parser.add_argument(
        "--resource-pack",
        default="backend/resources/resource_pack.yaml",
        help="Path to resource pack YAML",
    )
    parser.add_argument(
        "--use-pack",
        action="store_true",
        help="Use resource pack when CLI sources not provided",
    )
    
    # Actions
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset knowledge base before ingestion (WARNING: deletes existing data)",
    )
    parser.add_argument(
        "--save-report",
        default="backend/resources/lancedb_ingestion_report.json",
        help="Path to write ingestion report",
    )
    parser.add_argument(
        "--public-sources-only",
        action="store_true",
        default=True,
        help="Enforce domain allowlist (default: True)",
    )
    
    return parser.parse_args()


def _pick_sources(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str], str, str]:
    """Determine which sources to ingest based on CLI args and resource pack."""
    cli_urls = args.urls or []
    cli_pdfs: list[str] = []
    cli_pdf_urls = args.pdf_urls or []
    
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        cli_pdfs = [str(p) for p in sorted(pdf_dir.glob("*.pdf"))] if pdf_dir.exists() else []
    
    cli_docs = args.doc_paths or []
    
    # CLI sources take precedence
    if cli_urls or cli_pdfs or cli_pdf_urls or cli_docs:
        return cli_urls, cli_pdfs, cli_pdf_urls, cli_docs, "cli_overrides", ""
    
    # Use resource pack if requested
    if args.use_pack:
        pack = _load_resource_pack(args.resource_pack)
        pack_base = Path(args.resource_pack).resolve().parent
        
        urls = [str(u) for u in pack.get("web_urls", []) if isinstance(u, str)]
        pdfs = [str((pack_base / p).resolve()) for p in pack.get("pdf_paths", []) if isinstance(p, str)]
        pdf_urls = [str(u) for u in pack.get("pdf_urls", []) if isinstance(u, str)]
        docs = [str((pack_base / p).resolve()) for p in pack.get("doc_paths", []) if isinstance(p, str)]
        
        return urls, pdfs, pdf_urls, docs, str(pack.get("name", "unknown_pack")), str(Path(args.resource_pack))
    
    # Default URLs
    return list(DEFAULT_URLS), [], [], [], "built_in_defaults", ""


# --- Main execution ---

def main() -> None:
    """Main ingestion orchestration."""
    args = _parse_args()
    start_time = time.perf_counter()
    
    logger.info("=== LanceDB Ingestion Script ===")
    logger.info(f"Knowledge base: {args.knowledge_base}")
    logger.info(f"LanceDB path: {args.lancedb_path}")
    logger.info(f"Embedding model: {args.embedding_model}")
    
    # Resolve sources
    try:
        urls, pdfs, pdf_urls, docs, pack_name, pack_path = _pick_sources(args)
    except Exception as exc:
        logger.error(f"Failed to resolve sources: {exc}")
        sys.exit(1)
    
    logger.info(f"Sources - URLs: {len(urls)}, PDFs: {len(pdfs)}, PDF URLs: {len(pdf_urls)}, Docs: {len(docs)}")
    
    # Initialize embedding model
    try:
        logger.info("Loading embedding model...")
        embedding_function = HuggingFaceEmbeddings(model_name=args.embedding_model)
        logger.info("Embedding model loaded")
    except Exception as exc:
        logger.error(f"Failed to load embedding model: {exc}")
        sys.exit(1)
    
    # Initialize vector store
    try:
        logger.info("Initializing LanceDB vector store...")
        vector_store = LanceDBVectorStore(
            db_path=Path(args.lancedb_path),
            embedding_function=embedding_function,
            table_name="documents",
        )
        logger.info("Vector store initialized")
    except Exception as exc:
        logger.error(f"Failed to initialize vector store: {exc}")
        sys.exit(1)
    
    # Reset if requested
    if args.reset:
        logger.warning(f"Resetting knowledge base: {args.knowledge_base}")
        try:
            vector_store.delete_by_knowledge_base(args.knowledge_base)
            logger.info("Knowledge base reset complete")
        except Exception as exc:
            logger.error(f"Failed to reset knowledge base: {exc}")
            sys.exit(1)
    
    # Create text splitter
    text_splitter = _create_text_splitter(args.chunk_size, args.chunk_overlap)
    
    # Ingestion tracking
    totals = {
        "documents_processed": 0,
        "chunks_added": 0,
        "errors": 0,
    }
    report_errors: list[ReportError] = []
    success_count = 0
    source_results: list[dict[str, Any]] = []
    
    # Ingest web URLs
    for url in urls:
        logger.info(f"Ingesting URL: {url}")
        stats = ingest_web_page(url, vector_store, args.knowledge_base, text_splitter, args.public_sources_only)
        
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["errors"] += len(stats["errors"])
        
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": url, "error": err})
            source_results.append({
                "source": url,
                "source_type": "web",
                "domain": urlparse(url).netloc.lower(),
                "status": "failed",
                "chunks_added": stats["chunks_added"],
                "error": "; ".join(stats["errors"]),
            })
        else:
            success_count += 1
            source_results.append({
                "source": url,
                "source_type": "web",
                "domain": urlparse(url).netloc.lower(),
                "status": "ok",
                "chunks_added": stats["chunks_added"],
                "error": None,
            })
    
    # Ingest local PDFs
    for pdf in pdfs:
        logger.info(f"Ingesting PDF: {pdf}")
        stats = ingest_pdf(str(pdf), vector_store, args.knowledge_base, text_splitter)
        
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["errors"] += len(stats["errors"])
        
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": pdf, "error": err})
            source_results.append({
                "source": pdf,
                "source_type": "pdf_local",
                "domain": "local",
                "status": "failed",
                "chunks_added": stats["chunks_added"],
                "error": "; ".join(stats["errors"]),
            })
        else:
            success_count += 1
            source_results.append({
                "source": pdf,
                "source_type": "pdf_local",
                "domain": "local",
                "status": "ok",
                "chunks_added": stats["chunks_added"],
                "error": None,
            })
    
    # Ingest PDF URLs
    for pdf_url in pdf_urls:
        logger.info(f"Ingesting PDF URL: {pdf_url}")
        
        try:
            _ensure_allowed_domain(pdf_url, args.public_sources_only)
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            stats = ingest_pdf(tmp_path, vector_store, args.knowledge_base, text_splitter)
            
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
                
        except Exception as exc:
            stats = {
                "documents_processed": 1,
                "chunks_added": 0,
                "errors": [str(exc)],
            }
        
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["errors"] += len(stats["errors"])
        
        domain = urlparse(pdf_url).netloc.lower()
        
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": pdf_url, "error": err})
            source_results.append({
                "source": pdf_url,
                "source_type": "pdf_url",
                "domain": domain,
                "status": "failed",
                "chunks_added": stats["chunks_added"],
                "error": "; ".join(stats["errors"]),
            })
        else:
            success_count += 1
            source_results.append({
                "source": pdf_url,
                "source_type": "pdf_url",
                "domain": domain,
                "status": "ok",
                "chunks_added": stats["chunks_added"],
                "error": None,
            })
    
    # Ingest text documents
    for doc in docs:
        logger.info(f"Ingesting document: {doc}")
        stats = ingest_text_file(str(doc), vector_store, args.knowledge_base, text_splitter)
        
        totals["documents_processed"] += stats["documents_processed"]
        totals["chunks_added"] += stats["chunks_added"]
        totals["errors"] += len(stats["errors"])
        
        if stats["errors"]:
            for err in stats["errors"]:
                report_errors.append({"source": doc, "error": err})
            source_results.append({
                "source": doc,
                "source_type": "document",
                "domain": "local",
                "status": "failed",
                "chunks_added": stats["chunks_added"],
                "error": "; ".join(stats["errors"]),
            })
        else:
            success_count += 1
            source_results.append({
                "source": doc,
                "source_type": "document",
                "domain": "local",
                "status": "ok",
                "chunks_added": stats["chunks_added"],
                "error": None,
            })
    
    # Calculate duration
    duration = round(time.perf_counter() - start_time, 3)
    
    # Generate report
    report: IngestionReport = {
        "run_timestamp": _timestamp(),
        "resource_pack_name": pack_name,
        "resource_pack_path": pack_path,
        "knowledge_base": args.knowledge_base,
        "processed_urls": urls,
        "processed_pdfs": [*pdfs, *pdf_urls],
        "processed_docs": docs,
        "documents_processed": totals["documents_processed"],
        "chunks_added": totals["chunks_added"],
        "errors": report_errors,
        "total_duration_seconds": duration,
        "success_count": success_count,
        "failed_count": len(report_errors),
        "source_results": source_results,
    }
    
    _write_report(report, args.save_report)
    
    # Print summary
    logger.info("=== Ingestion Complete ===")
    logger.info(f"Documents processed: {totals['documents_processed']}")
    logger.info(f"Chunks added: {totals['chunks_added']}")
    logger.info(f"Success count: {success_count}")
    logger.info(f"Failed count: {len(report_errors)}")
    logger.info(f"Duration: {duration:.2f}s")
    
    # Exit with appropriate code
    total_sources = len(urls) + len(pdfs) + len(pdf_urls) + len(docs)
    if total_sources > 0 and success_count == 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
