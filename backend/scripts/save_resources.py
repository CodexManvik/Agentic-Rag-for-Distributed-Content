import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", value)


def _extension_from_content_type(content_type: str) -> str:
    content_type = content_type.lower()
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    if "json" in content_type:
        return ".json"
    return ".bin"


def _save_url(url: str, output_dir: Path) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return False, f"request_failed: {exc}"

    content_type = response.headers.get("content-type", "")
    parsed = urlparse(url)
    base = _sanitize_name(f"{parsed.netloc}{parsed.path}".strip("/")) or "resource"

    if url.lower().endswith(".pdf") or "pdf" in content_type.lower():
        file_path = output_dir / f"{base}.pdf"
        file_path.write_bytes(response.content)
        return True, f"saved_pdf: {file_path}"

    if "html" in content_type.lower() or response.text:
        file_path = output_dir / f"{base}.html"
        file_path.write_text(response.text, encoding="utf-8", errors="ignore")
        return True, f"saved_html_snapshot: {file_path}"

    ext = _extension_from_content_type(content_type)
    file_path = output_dir / f"{base}{ext}"
    file_path.write_bytes(response.content)
    return True, f"saved_binary: {file_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Save optional local resource snapshots from URLs")
    parser.add_argument("--urls", nargs="+", required=True, help="Resource URLs")
    parser.add_argument(
        "--output-dir",
        default="backend/resources/pdfs",
        help="Output directory for saved resources",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for url in args.urls:
        ok, message = _save_url(url, output_dir)
        if ok:
            saved += 1
            print(f"OK {url} -> {message}")
        else:
            skipped += 1
            print(f"SKIP {url} -> {message}")

    print(f"Finished. saved={saved}, skipped={skipped}")


if __name__ == "__main__":
    main()
