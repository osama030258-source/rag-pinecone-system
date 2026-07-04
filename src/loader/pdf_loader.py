"""
PDF Loader Module
------------------
Responsible for:
  - Validating uploaded PDF files
  - Extracting raw text page by page
  - Cleaning formatting artifacts (extra whitespace, broken lines, etc.)

Returns a list of dicts, one per page, so downstream modules
(chunker) can keep page-number metadata for source attribution.
"""

import re
from pathlib import Path
from pypdf import PdfReader


MAX_FILE_SIZE_MB = 20


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or is invalid."""
    pass


def validate_pdf(file_path: str) -> None:
    """
    Validates that the file exists, is a PDF, and is within the size limit.
    Raises PDFLoadError if any check fails.
    """
    path = Path(file_path)

    if not path.exists():
        raise PDFLoadError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Invalid file type: {path.suffix}. Only .pdf files are supported.")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PDFLoadError(
            f"File too large: {size_mb:.2f}MB. Maximum allowed is {MAX_FILE_SIZE_MB}MB."
        )

    if size_mb == 0:
        raise PDFLoadError("File is empty.")


def clean_text(text: str) -> str:
    """
    Removes common PDF extraction artifacts:
      - Excess whitespace / repeated newlines
      - Hyphenation at line breaks (e.g. "informa-\ntion" -> "information")
      - Stray page-number-only lines
    """
    if not text:
        return ""

    # Fix hyphenated line breaks: "exam-\nple" -> "example"
    text = re.sub(r"-\n\s*", "", text)

    # Collapse multiple newlines into a single space
    text = re.sub(r"\n+", " ", text)

    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)

    # Remove stray lines that are just numbers (likely page numbers)
    text = re.sub(r"\b\d{1,4}\b(?=\s*$)", "", text)

    return text.strip()


def load_pdf(file_path: str, document_name: str = None) -> list[dict]:
    """
    Loads a PDF and returns a list of page-level records:
        [
            {
                "document_name": "sample.pdf",
                "page_number": 1,
                "text": "cleaned page text..."
            },
            ...
        ]

    Pages with no extractable text (e.g. scanned images) are skipped
    but logged via the returned 'skipped_pages' info in the caller if needed.
    """
    validate_pdf(file_path)

    document_name = document_name or Path(file_path).name

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise PDFLoadError(f"Failed to read PDF: {e}")

    if len(reader.pages) == 0:
        raise PDFLoadError("PDF has no pages.")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        cleaned = clean_text(raw_text)

        if cleaned:  # skip pages with no usable text (e.g. pure images)
            pages.append({
                "document_name": document_name,
                "page_number": i,
                "text": cleaned
            })

    if not pages:
        raise PDFLoadError(
            "No extractable text found in PDF. It may be a scanned/image-only document."
        )

    return pages


if __name__ == "__main__":
    # Quick manual test — run: python src/loader/pdf_loader.py path/to/file.pdf
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    result = load_pdf(sys.argv[1])
    print(f"Extracted {len(result)} pages.")
    print("--- Preview of page 1 ---")
    print(result[0]["text"][:500])
