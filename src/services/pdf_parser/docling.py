import logging
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.exceptions import PDFValidationError

logger = logging.getLogger(__name__)


class DoclingParser:
    """Docling PDF parser for pdf document processing."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize DocumentConverter.

        max_pages: Maximum number of pages to process
        max_file_size_mb: Maximum file size in MB
        do_ocr: Enable OCR for scanned PDFs (default: False, very slow)
        do_table_structure: Extract table structures (default: True)
        """
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,  
        )

        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _warm_up_models(self):
        
        if not self._warmed_up:
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """PDF validation including size and page limits.

        :pdf_path: Path to PDF file
        """
        try:
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            actual_pages = len(pdf_doc)
            pdf_doc.close()

            if actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    async def parse_pdf(self, pdf_path: Path) -> str:
        """Parse PDF using Docling parser.
        Limited to 20 pages to avoid memory issues with large papers.

        :param pdf_path: Path to PDF file
        :returns: Text from sections of PDF
        """
        try:

            self._validate_pdf(pdf_path)
            self._warm_up_models()

            result = self._converter.convert(str(pdf_path), max_num_pages=self.max_pages, max_file_size=self.max_file_size_bytes)

            # Extract structured content
            doc = result.document

            # Direct markdown export
            section_text = doc.export_to_markdown()
            return section_text
             
        except PDFValidationError as e:
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            else:
                raise

        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            
