"""Document extraction and text processing flow."""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, cast

# Third-party libraries with missing type stubs
import ebooklib  # type: ignore[import-untyped]
import markdown  # type: ignore[import-untyped]
from bs4 import BeautifulSoup
from docx import Document  # type: ignore[import-not-found]
from ebooklib import epub
from langdetect import detect  # type: ignore[import-untyped]

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.knowledge.models import (
    DocumentContent,
    DocumentExtractionInput,
    DocumentExtractionOutput,
    DocumentMetadata,
    DocumentType,
    KnowledgeExtractionRequest,
    ProcessingStatus,
    TextChunk,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes different document types into text."""

    def __init__(self) -> None:
        self.supported_processors = {
            DocumentType.TXT: self._process_txt,
            DocumentType.PDF: self._process_pdf,
            DocumentType.EPUB: self._process_epub,
            DocumentType.DOCX: self._process_docx,
            DocumentType.HTML: self._process_html,
            DocumentType.MARKDOWN: self._process_markdown,
        }

    async def process_single_document(self, file_path: str) -> DocumentContent:
        """Process a single document (optimized for streaming)."""

        path = Path(file_path)
        logger.info(f"Processing single document: {path.name}")

        # Detect document type
        doc_type = self._detect_document_type(path)
        if not doc_type:
            return DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=self._create_metadata(path, None),
                full_text="",
                chunks=[],
                status=ProcessingStatus.FAILED,
                error_message=f"Unsupported file type: {path.suffix}"
            )

        try:
            # Extract metadata
            metadata = self._create_metadata(path, doc_type)

            # Process document content
            processor = self.supported_processors.get(doc_type)
            if not processor:
                raise ValueError(f"No processor available for {doc_type}")

            full_text = await processor(path)

            # Create basic chunks (will be enhanced by smart chunking later)
            chunks = self._create_chunks(
                full_text,
                self._generate_doc_id(file_path),
                chunk_size=1000,
                overlap=200
            )

            # Create document content
            doc_content = DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=metadata,
                full_text=full_text,
                chunks=chunks,
                status=ProcessingStatus.COMPLETED,
                language_detected=self._detect_language(full_text),
                reading_time_minutes=self._estimate_reading_time(full_text)
            )

            logger.debug(f"Successfully processed {path.name}: {len(full_text)} chars, {len(chunks)} chunks")
            return doc_content

        except Exception as e:
            logger.error(f"Failed to process document {path.name}: {e}")
            return DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=self._create_metadata(path, doc_type),
                full_text="",
                chunks=[],
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )

    async def process_document(self, file_path: str) -> DocumentContent:
        """Process a single document."""
        path = Path(file_path)

        # Detect document type
        doc_type = self._detect_document_type(path)
        if not doc_type:
            return DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=self._create_metadata(path, None),
                full_text="",
                chunks=[],
                status=ProcessingStatus.FAILED,
                error_message=f"Unsupported file type: {path.suffix}"
            )

        try:
            # Extract metadata
            metadata = self._create_metadata(path, doc_type)

            # Process document content
            processor = self.supported_processors.get(doc_type)
            if not processor:
                raise ValueError(f"No processor available for {doc_type}")

            full_text = await processor(path)

            # Create chunks
            chunks = self._create_chunks(
                full_text,
                self._generate_doc_id(file_path),
                chunk_size=1000,
                overlap=200
            )

            # Create document content
            doc_content = DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=metadata,
                full_text=full_text,
                chunks=chunks,
                status=ProcessingStatus.COMPLETED,
                language_detected=self._detect_language(full_text),
                reading_time_minutes=self._estimate_reading_time(full_text)
            )

            logger.info(f"Processed {path.name}: {len(full_text)} chars, {len(chunks)} chunks")
            return doc_content

        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")
            return DocumentContent(
                document_id=self._generate_doc_id(file_path),
                metadata=self._create_metadata(path, doc_type),
                full_text="",
                chunks=[],
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )

    def _detect_document_type(self, path: Path) -> Optional[DocumentType]:
        """Detect document type from file extension."""
        extension = path.suffix.lower()

        type_mapping = {
            '.txt': DocumentType.TXT,
            '.pdf': DocumentType.PDF,
            '.epub': DocumentType.EPUB,
            '.mobi': DocumentType.MOBI,
            '.docx': DocumentType.DOCX,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN,
        }

        return type_mapping.get(extension)

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID."""
        return hashlib.md5(file_path.encode()).hexdigest()

    def _create_metadata(self, path: Path, doc_type: Optional[DocumentType]) -> DocumentMetadata:
        """Create document metadata."""
        try:
            stat = path.stat()

            return DocumentMetadata(
                file_path=str(path),
                file_name=path.name,
                file_size=stat.st_size,
                file_type=doc_type or DocumentType.TXT,
                created_date=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified_date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                title=path.stem,  # Use filename as title for now
            )
        except Exception as e:
            logger.warning(f"Error creating metadata for {path}: {e}")
            return DocumentMetadata(
                file_path=str(path),
                file_name=path.name,
                file_size=0,
                file_type=doc_type or DocumentType.TXT,
                title=path.stem,
            )

    def _create_chunks(
        self,
        text: str,
        document_id: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the next 100 characters
                sentence_ends = []
                for i, char in enumerate(text[end:end+100]):
                    if char in '.!?':
                        sentence_ends.append(end + i + 1)

                if sentence_ends:
                    end = sentence_ends[0]

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    document_id=document_id,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)

            if start >= len(text):
                break

        return chunks

    async def _process_txt(self, path: Path) -> str:
        """Process plain text file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode text file with any supported encoding")

    async def _process_pdf(self, path: Path) -> str:
        """Process PDF file."""
        try:
            import PyPDF2

            text = ""
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            return text.strip()

        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Error processing PDF {path}: {e}")
            return ""

    async def _process_epub(self, path: Path) -> str:
        """Process EPUB file."""
        try:

            book = epub.read_epub(str(path))
            text_content = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    text_content.append(text)

            return '\n'.join(text_content)

        except ImportError:
            logger.error("ebooklib not installed. Install with: pip install ebooklib beautifulsoup4")
            return ""
        except Exception as e:
            logger.error(f"Error processing EPUB {path}: {e}")
            return ""

    async def _process_docx(self, path: Path) -> str:
        """Process DOCX file."""
        try:
            doc = Document(str(path))
            text_content = []

            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)

            return '\n'.join(text_content)

        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"Error processing DOCX {path}: {e}")
            return ""

    async def _process_html(self, path: Path) -> str:
        """Process HTML file."""
        try:
            from bs4 import BeautifulSoup

            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()

            soup = BeautifulSoup(content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            return soup.get_text()

        except ImportError:
            logger.error("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            return ""
        except Exception as e:
            logger.error(f"Error processing HTML {path}: {e}")
            return ""

    async def _process_markdown(self, path: Path) -> str:
        """Process Markdown file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                md_content = file.read()

            # Convert to HTML then extract text
            html = markdown.markdown(md_content)

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            except ImportError:
                # Fallback: return markdown as-is
                return md_content

        except ImportError:
            logger.warning("markdown not installed. Install with: pip install markdown")
            # Fallback: treat as plain text
            return await self._process_txt(path)
        except Exception as e:
            logger.error(f"Error processing Markdown {path}: {e}")
            return ""

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect text language."""
        if not text:
            return None

        try:
            return cast(str, detect(text[:1000]))  # Use first 1000 chars for detection
        except ImportError:
            logger.warning("langdetect not installed. Install with: pip install langdetect")
            return None
        except Exception:
            return None

    def _estimate_reading_time(self, text: str) -> int:
        """Estimate reading time in minutes (assuming 200 WPM)."""
        if not text:
            return 0

        word_count = len(text.split())
        return max(1, word_count // 200)


@flow(name="document-extraction-flow", description="Extract and process documents into structured content")  # type: ignore[arg-type]
class DocumentExtractionFlow:
    """Flow for extracting content from various document types."""

    async def process_single_document(self, doc_path: str) -> DocumentContent:
        """Process a single document (optimized for streaming)."""

        logger.info(f"Processing single document: {doc_path}")

        # Create processor locally
        processor = DocumentProcessor()

        # Process the document
        return await processor.process_single_document(doc_path)

    async def stream_document_processing(self, input_directory: str) -> AsyncGenerator[DocumentContent, None]:
        """Stream documents one by one for processing (generator)."""

        from pathlib import Path

        from ..models import DocumentType

        input_path = Path(input_directory)
        supported_extensions = {
            f".{doc_type.value}" for doc_type in DocumentType
        }

        processor = DocumentProcessor()

        # Stream through files
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc_content = await processor.process_single_document(str(file_path))
                    yield doc_content
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    # Yield failed document
                    yield DocumentContent(
                        document_id=processor._generate_doc_id(str(file_path)),
                        metadata=processor._create_metadata(file_path, None),
                        full_text="",
                        chunks=[],
                        status=ProcessingStatus.FAILED,
                        error_message=str(e)
                    )

    async def _extract_documents(self, input_data: DocumentExtractionInput) -> DocumentExtractionOutput:
        """Extract content from all provided documents."""
        logger.info(f"Processing {len(input_data.file_paths)} documents")

        # Create processor locally
        processor = DocumentProcessor()

        documents: List[DocumentContent] = []
        failed_files: List[Dict[str, str]] = []

        # Process documents concurrently (in batches to avoid overwhelming system)
        batch_size = 5
        for i in range(0, len(input_data.file_paths), batch_size):
            batch = input_data.file_paths[i:i+batch_size]

            # Process batch concurrently
            tasks = [processor.process_document(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for file_path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {file_path}: {result}")
                    failed_files.append({
                        "file_path": file_path,
                        "error": str(result)
                    })
                elif isinstance(result, DocumentContent) and result.status == ProcessingStatus.COMPLETED:
                    documents.append(result)
                elif isinstance(result, DocumentContent):
                    failed_files.append({
                        "file_path": file_path,
                        "error": result.error_message or "Unknown error"
                    })
                else:
                    # Should not reach here due to prior exception handling
                    failed_files.append({
                        "file_path": file_path,
                        "error": "Unexpected result type"
                    })

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(input_data.file_paths) + batch_size - 1)//batch_size}")

        logger.info(f"Successfully processed {len(documents)} documents, {len(failed_files)} failed")

        return DocumentExtractionOutput(
            documents=documents,
            failed_files=failed_files
        )

    @pipeline(input_model=KnowledgeExtractionRequest, output_model=DocumentExtractionOutput)
    async def run_pipeline(self, request: KnowledgeExtractionRequest) -> DocumentExtractionOutput:
        """Process all documents in a directory."""
        logger.info(f"Scanning directory: {request.input_directory}")

        # Discover files
        file_paths = self._discover_files(
            request.input_directory,
            request.supported_formats,
            request.max_files
        )

        if not file_paths:
            logger.warning(f"No supported files found in {request.input_directory}")
            return DocumentExtractionOutput(documents=[], failed_files=[])

        logger.info(f"Found {len(file_paths)} files to process")

        # Create extraction input
        extraction_input = DocumentExtractionInput(
            request=request,
            file_paths=file_paths
        )

        # Process documents
        return await self._extract_documents(extraction_input)

    def _discover_files(
        self,
        directory: str,
        supported_formats: List[DocumentType],
        max_files: Optional[int] = None
    ) -> List[str]:
        """Discover files in directory that match supported formats."""
        path = Path(directory)

        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Build extension list
        extensions = []
        for doc_type in supported_formats:
            if doc_type == DocumentType.TXT:
                extensions.extend(['.txt'])
            elif doc_type == DocumentType.PDF:
                extensions.extend(['.pdf'])
            elif doc_type == DocumentType.EPUB:
                extensions.extend(['.epub'])
            elif doc_type == DocumentType.MOBI:
                extensions.extend(['.mobi'])
            elif doc_type == DocumentType.DOCX:
                extensions.extend(['.docx'])
            elif doc_type == DocumentType.HTML:
                extensions.extend(['.html', '.htm'])
            elif doc_type == DocumentType.MARKDOWN:
                extensions.extend(['.md', '.markdown'])

        # Find matching files
        file_paths = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            matching_files = list(path.glob(pattern))
            file_paths.extend([str(f) for f in matching_files if f.is_file()])

        # Remove duplicates and sort
        file_paths = sorted(list(set(file_paths)))

        # Apply limit if specified
        if max_files and len(file_paths) > max_files:
            file_paths = file_paths[:max_files]
            logger.info(f"Limited to first {max_files} files")

        return file_paths
