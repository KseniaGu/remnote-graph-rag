import io
import mimetypes
import re
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import fitz
import requests
from trafilatura import fetch_url, extract

from configs.constants import FILENAME_LENGTH_MAX, DOCUMENT_PARSER_LOGGING
from configs.enums import PDFParsingStatus, ImageParsingStatus, TextParsingStatus
from src.utils.common_funcs import write_file
from src.utils.helpers import get_logger

logger = get_logger(DOCUMENT_PARSER_LOGGING)


def clean_text(text: str) -> str:
    """Runs all text normalization and cleaning steps.
    
    Args:
        text: Input text to clean.
        
    Returns:
        Cleaned and normalized text.
    """
    if not text:
        return ""

    text = normalize_whitespace(text)
    text = clean_remnote_references(text)
    text = clean_clozes(text)
    text = strip_markdown_formatting(text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Replaces multiple spaces/tabs with a single space.
    
    Args:
        text: Input text.
        
    Returns:
        Text with normalized whitespace.
    """
    return " ".join(text.split())


def clean_remnote_references(text: str) -> str:
    """Cleans RemNote specific reference marking.
    
    Removes RemNote reference syntax like [[Page Name]] or [[Page Name|rem-id-123]].
    
    Args:
        text: Input text with RemNote references.
        
    Returns:
        Text with references cleaned.
    """
    # [[Text|ID]] -> Replace with Text
    text = re.sub(r'\[\[(.*?)\|.*?\]\]', r'\1', text)
    # [[Text]] -> Replace with Text
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    return text


def clean_clozes(text: str) -> str:
    """Processes patterns used in RemNote Flashcards.
    
    Removes cloze deletion syntax like {{c1::hidden text}}.
    
    Args:
        text: Input text with cloze patterns.
        
    Returns:
        Text with cloze patterns cleaned.
    """
    # Pattern: {{c(number)::(text)}}
    text = re.sub(r'\{\{c\d+::(.*?)(::.*?)?\}\}', r'\1', text)
    return text


def strip_markdown_formatting(text: str) -> str:
    """Removes Markdown formatting markers.
    
    Args:
        text: Input text with Markdown formatting.
        
    Returns:
        Text with formatting markers removed.
    """
    # Remove bold/italic stars
    text = text.replace("**", "").replace("__", "")
    # Remove images usually formatted as ![alt](url) -> keep alt
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    return text


def safe_unpack(zip_path: Path, extract_to: Path):
    """Unpacks a zip file while truncating long filenames to avoid OS errors.
    
    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract files to.
    """
    zip_path = Path(zip_path)
    extract_base = Path(extract_to)
    extract_base.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            original_path = Path(member.filename)
            filename = original_path.name
            extension = original_path.suffix

            if len(filename) > 100:
                truncated_name = filename[:90] + "_truncated" + extension
            else:
                truncated_name = filename

            target_path = extract_base / truncated_name

            with zf.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())

    logger.info(f"Successfully unpacked {zip_path.name} to {extract_to}")


def search_for_url(line: str) -> tuple[Optional[str], Optional[str]]:
    """Finds a Markdown link and returns (name, url).

    If no Markdown link is found, it falls back to finding a raw URL.
    
    Args:
        line: Text line to search for URLs.
        
    Returns:
        Tuple of (link name, URL). Name may be None if not found.
    """
    markdown_link_pattern = re.compile(r'\[(?P<name>[^\]]+)\]\((?P<url>[^)]+)\)')
    match = markdown_link_pattern.search(line)

    if match:
        name, url = match.group('name'), match.group('url')
        if name == url:
            name = None
        return name, url

    # Fallback: find raw URL if no [name](url) structure exists
    raw_url_match = re.search(r'(https?://[^\s)]+)', line)
    if raw_url_match:
        return None, raw_url_match.group(0)

    return None, None


def save_pdf_by_url(
        response: requests.Response,
        name: Optional[str],
        url: str,
        output_dir: Path
) -> tuple[PDFParsingStatus, Path]:
    """Saves a PDF from HTTP response to disk.
    
    Args:
        response: HTTP response containing PDF content.
        name: Optional name for the file.
        url: Source URL of the PDF.
        output_dir: Directory to save the PDF.
        
    Returns:
        Tuple of (parsing status, file path).
    """
    if name:
        clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
        if not clean_name:
            clean_name = str(uuid.uuid4())
        filename = f"{clean_name}.pdf"
    else:
        filename = url.split('/')[-1].split('?')[0]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

    filename = filename[-FILENAME_LENGTH_MAX:] if len(filename) > FILENAME_LENGTH_MAX else filename

    pdf_path = output_dir / filename
    if pdf_path.exists():
        return PDFParsingStatus.file_exists, pdf_path

    try:
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            doc.save(pdf_path)
    except Exception:
        logger.error(f"Exception while parsing {url}", exc_info=True)
        return PDFParsingStatus.failed, Path()

    return PDFParsingStatus.success, pdf_path


def save_image_by_url(
        response: requests.Response,
        content_type: str,
        name: Optional[str],
        url: str,
        output_dir: Path
) -> tuple[ImageParsingStatus, Path]:
    """Saves an image from HTTP response to disk.
    
    Args:
        response: HTTP response containing image content.
        content_type: MIME type of the image.
        name: Optional name for the file.
        url: Source URL of the image.
        output_dir: Directory to save the image.
        
    Returns:
        Tuple of (parsing status, file path).
    """
    if name:
        clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    else:
        clean_name = url.split('/')[-1].split('?')[0]
    if not clean_name:
        clean_name = str(uuid.uuid4())
    else:
        clean_name = clean_name[-FILENAME_LENGTH_MAX:] if len(clean_name) > FILENAME_LENGTH_MAX else clean_name
    try:
        ext = mimetypes.guess_extension(content_type) or ".jpg"
        filename = f"{clean_name}{ext}" if not clean_name.endswith(ext) else clean_name
        image_path = output_dir / filename

        if image_path.exists():
            return ImageParsingStatus.file_exists, image_path

        # Save the file in binary chunks (Memory efficient for high-res images)
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return ImageParsingStatus.success, image_path

    except Exception:
        logger.error(f"Exception while parsing {url}", exc_info=True)
        return ImageParsingStatus.failed, Path()


def save_text_by_url(name: Optional[str], url: str, output_dir: Path) -> tuple[TextParsingStatus, Path]:
    """Saves text content from URL to Markdown file.
    
    Args:
        name: Optional name for the file.
        url: Source URL of the text content.
        output_dir: Directory to save the Markdown file.
        
    Returns:
        Tuple of (parsing status, file path).
    """
    if name:
        clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    else:
        clean_name = url.split('/')[-1].split('?')[0]
    if not clean_name:
        clean_name = str(uuid.uuid4())
    else:
        clean_name = clean_name[-FILENAME_LENGTH_MAX:] if len(clean_name) > FILENAME_LENGTH_MAX else clean_name
    file_path = output_dir / f"{clean_name}.md"
    if file_path.exists():
        return TextParsingStatus.file_exists, file_path

    try:
        downloaded = fetch_url(url)
        result = extract(downloaded, output_format="markdown", with_metadata=True)
        write_file(result, file_path)
        return TextParsingStatus.success, file_path
    except Exception:
        logger.error(f"Exception while parsing {url}", exc_info=True)
        return TextParsingStatus.failed, Path()
