"""Document processing utilities."""

import uuid
import logging
from pathlib import Path
from typing import List, Union
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents and split into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self, file_path: Union[str, Path]) -> List:
        """
        Load a document from file.

        Args:
            file_path: Path to document file

        Returns:
            List of document objects

        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif suffix == '.txt':
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported types: .pdf, .txt"
            )

        logger.info(f"Loading document from {file_path}")
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages/sections")

        return documents

    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks.

        Args:
            documents: List of document objects

        Returns:
            List of document chunks
        """
        chunks = self.splitter.split_documents(documents)
        logger.info(
            f"Split documents into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def process_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Process a file and return a dataframe of chunks.

        Args:
            file_path: Path to document file

        Returns:
            DataFrame with text chunks and metadata
        """
        # Load and split document
        documents = self.load_document(file_path)
        chunks = self.split_documents(documents)

        # Convert to dataframe
        rows = []
        for chunk in chunks:
            row = {
                "text": chunk.page_content,
                "chunk_id": uuid.uuid4().hex,
                **chunk.metadata,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Created dataframe with {len(df)} chunks")

        return df

    def save_chunks(self, df: pd.DataFrame, output_path: Path):
        """
        Save chunks dataframe to file.

        Args:
            df: DataFrame with chunks
            output_path: Path to save file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep="|", index=False)
        logger.info(f"Saved chunks to {output_path}")
