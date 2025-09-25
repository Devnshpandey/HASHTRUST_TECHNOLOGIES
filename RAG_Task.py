# """ RAG Policy Reviewer
# A retrieval-augmented generation system that answers questions based on
# company policy documents stored in a designated folder."""

import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import PyPDF2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#-----------------------------------#
@dataclass
class DocumentChunk:
    """Represents a chunk of text from a policy document."""
    text: str
    source: str  # Name of the source document
    page_number: int
    chunk_index: int

#--------------------------------------#
class PolicyProcessor:
    """Handles processing and management of policy documents."""

    def __init__(self, documents_path: str = "./policy_documents/"):
        """
        Initialize the Policy Processor.

        Args:
            documents_path: Path to the directory containing policy documents
        """
        self.documents_path = Path(documents_path)
        self._ensure_documents_directory_exists()

    def _ensure_documents_directory_exists(self) -> None:
        """Ensure the documents directory exists, create if it doesn't."""
        try:
            self.documents_path.mkdir(exist_ok=True)
            logger.info(f"Documents directory: {self.documents_path}")
        except OSError as e:
            logger.error(f"Error creating documents directory: {str(e)}")
            raise

    def get_policy_files(self) -> List[Path]:
        """
        Get all PDF policy files in the documents directory.

        Returns:
            List of Path objects for PDF files
        """
        pdf_files = list(self.documents_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.documents_path}")
        return pdf_files

    def extract_text_from_pdf(self, file_path: Path) -> List[Dict[str, str]]:
        """
        Extract text content from a PDF file with page information.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of dictionaries containing text and page number for each page

        Raises:
            PyPDF2.errors.PdfReadError: If the PDF cannot be read
            IOError: If the file cannot be opened
        """
        text_pages = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        text_pages.append({
                            'text': text,
                            'page_number': page_num
                        })
            logger.info(f"Extracted text from {file_path.name} ({len(text_pages)} pages)")
        except (PyPDF2.errors.PdfReadError, IOError) as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            raise

        return text_pages

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.

        Args:
            text: The text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters

        Returns:
            List of text chunks
        """
        # Clean the text by removing extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence end if possible
            if end < len(text):
                sentence_enders = ['.', '!', '?', '\n']
                for i in range(end, min(end + 50, len(text))):
                    if text[i] in sentence_enders:
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - overlap if (end - overlap) > start else end

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks

#-----------------------------------------#
class EmbeddingManager:
    """Manages text embeddings for document chunks."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Embedding Manager.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        logger.info(f"Initialized embedding model: {model_name}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(texts)
        logger.info(f"Generated embeddings for {len(texts)} text chunks")
        return embeddings

    def find_similar_chunks(self, query: str, chunk_embeddings: np.ndarray,
                           all_chunks: List[DocumentChunk], top_k: int = 3,
                           similarity_threshold: float = 0.5) -> List[DocumentChunk]:
        """
        Find the most relevant document chunks for a query.

        Args:
            query: The query string
            chunk_embeddings: Embeddings for all document chunks
            all_chunks: List of all document chunks
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score to include

        Returns:
            List of relevant document chunks
        """
        if not all_chunks or chunk_embeddings.size == 0:
            return []

        # Encode the query
        query_embedding = self.model.encode([query])

        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Filter by similarity threshold and return results
        results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                results.append(all_chunks[idx])

        logger.debug(f"Found {len(results)} relevant chunks for query: {query}")
        return results

#------------------------------------------------#
class PolicyReviewer:
    """
    A RAG system for querying company policy documents.

    Attributes:
        documents_path: Path to the directory containing policy documents
        policy_processor: Processor for policy documents
        embedding_manager: Manager for text embeddings
        document_chunks: List of processed document chunks
        chunk_embeddings: Embeddings for each document chunk
    """

    def __init__(self, documents_path: str = "./policy_documents/"):
        """
        Initialize the Policy Reviewer.

        Args:
            documents_path: Path to the directory containing policy documents
        """
        self.documents_path = documents_path
        self.policy_processor = PolicyProcessor(documents_path)
        self.embedding_manager = EmbeddingManager()
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_embeddings: np.ndarray = np.array([])

        # Process documents on initialization
        self.process_documents()

    def process_documents(self) -> None:
        """Process all PDF documents in the documents directory."""
        self.document_chunks = []

        # Get all PDF files
        pdf_files = self.policy_processor.get_policy_files()

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_path}")
            return

        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                text_pages = self.policy_processor.extract_text_from_pdf(pdf_file)

                for page_data in text_pages:
                    chunks = self.policy_processor.chunk_text(page_data['text'])

                    for i, chunk in enumerate(chunks):
                        self.document_chunks.append(DocumentChunk(
                            text=chunk,
                            source=pdf_file.name,
                            page_number=page_data['page_number'],
                            chunk_index=i
                        ))
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue

        # Generate embeddings for all chunks
        if self.document_chunks:
            texts = [chunk.text for chunk in self.document_chunks]
            self.chunk_embeddings = self.embedding_manager.generate_embeddings(texts)
            logger.info(f"Processed {len(self.document_chunks)} chunks from policy documents")
        else:
            logger.warning("No text chunks were extracted from the documents")

    def answer_policy_question(self, question: str) -> str:
        """
        Answer a question based on the policy documents.

        Args:
            question: The question to answer

        Returns:
            An answer based on the policy documents
        """
        if not self.document_chunks or self.chunk_embeddings.size == 0:
            return "No policy documents are available for querying. Please ensure PDF files are placed in the policy_documents directory."

        # Find relevant document chunks
        relevant_chunks = self.embedding_manager.find_similar_chunks(
            question, self.chunk_embeddings, self.document_chunks
        )

        if not relevant_chunks:
            return "I couldn't find relevant information in the policy documents. Please try rephrasing your question or ask a human resources representative."

        # Format response
        response = self._format_response(question, relevant_chunks)

        return response

    @staticmethod
    def _format_response(question: str, relevant_chunks: List[DocumentChunk]) -> str:
        """
        Format a response based on the retrieved context.

        Args:
            question: The original question
            relevant_chunks: The relevant document chunks

        Returns:
            A formatted response
        """
        # Group chunks by source for better organization
        chunks_by_source: Dict[str, List[DocumentChunk]] = {}
        for chunk in relevant_chunks:
            if chunk.source not in chunks_by_source:
                chunks_by_source[chunk.source] = []
            chunks_by_source[chunk.source].append(chunk)

        # Build response
        response = f"Regarding your question '{question}', here's what I found in company policies:\n\n"

        for source, chunks in chunks_by_source.items():
            response += f"From {source}:\n"
            for chunk in chunks:
                response += f"- Page {chunk.page_number}: {chunk.text}\n"
            response += "\n"

        response += "Please consult the full policy documents or HR for complete information."

        return response

    def list_available_policies(self) -> List[str]:
        """
        List all available policy documents.

        Returns:
            List of policy document names
        """
        pdf_files = self.policy_processor.get_policy_files()
        return [file.name for file in pdf_files]


def run_policy_reviewer() -> None:
    """Run a simple command-line interface for the Policy Reviewer."""
    reviewer = PolicyReviewer()

    policies = reviewer.list_available_policies()
    if not policies:
        print("No policy documents found. Please add PDF files to the policy_documents directory.")
        return

    print(f"Policy Reviewer initialized with {len(policies)} policy documents.")
    print("Type 'quit' to exit or ask a question about company policies.")

    while True:
        try:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            response = reviewer.answer_policy_question(user_input)
            print(f"\n{response}")

        except KeyboardInterrupt:
            print("\n\nSession ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")


if __name__ == "__main__":
    run_policy_reviewer()