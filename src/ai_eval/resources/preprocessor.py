# import os
import re
from typing import List, Optional

# from langchain_google_vertexai import VertexAIEmbeddings
# from langchain_ollama import OllamaEmbeddings
from chonkie import RecursiveChunker, SemanticChunker, SentenceChunker
from langchain.docstore.document import Document

# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from ai_eval.config import global_config as glob
from ai_eval.config.config import model_list
from ai_eval.services.clients import GCPClient
from ai_eval.services.file_gcp import stream_gcs_pdf
from ai_eval.services.logger import LoggerFactory
from ai_eval.utils.utils import list_objects, timer

my_logger = LoggerFactory().create_module_logger()


class Preprocessor:
    def __init__(self, bucket_name: str = glob.GCP_CS_BUCKET, verbose: bool = True):
        """
        Initialize the Preprocessor with a GCP bucket name.

        Args:
            bucket_name (str): The name of the GCP bucket.
        """
        self.bucket_name = bucket_name
        self.gcp_client = GCPClient(bucket_name)
        self.verbose = verbose

    @timer
    def fetch_documents(
        self, blob_path: Optional[str] = None, source: str = "gcp"
    ) -> List[Document]:
        """
        Fetch and process documents from GCP Cloud Storage or local file using PyPDFLoader for local files.

        Args:
            blob_path (Optional[str]): Path to the blob (GCP) or absolute file path (local).
            source (str): "gcp" or "local".

        Returns:
            List[Document]: List of LangChain Document objects.
        """
        documents = []
        if source == "gcp":
            if blob_path is not None:
                blobs = [blob_path]
            else:
                blobs = list_objects(self.bucket_name)
            for prefix in blobs:
                if not prefix.endswith(".pdf"):
                    continue
                streamed_text = stream_gcs_pdf(prefix)
                content = [doc for doc in streamed_text]
                documents.extend(content)
                my_logger.info(f"Processed {prefix}")
        elif source == "local":
            if blob_path is not None:
                loader = PyPDFLoader(blob_path)
                documents = loader.load()
                my_logger.info(f"Processed {blob_path}")
            else:
                raise ValueError(
                    "Provide blob_path as absolute file path for local source."
                )
        else:
            raise ValueError("Source must be 'gcp' or 'local'.")

        return documents

    @timer
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller, semantically meaningful chunks.

        This method uses a semantic chunking approach to divide the input documents
        into smaller parts based on semantic similarity and a specified breakpoint
        threshold. The chunking process is powered by an embedding model and a
        semantic chunker.

        Args:
            documents (List[Document]): A list of `Document` objects to be chunked.

        Returns:
            List[Document]: A list of chunked `Document` objects, where each chunk
            represents a semantically meaningful portion of the original documents.
        """
        texts = [page.page_content for page in documents]
        # texts = " ".join(texts).replace("\n", "")

        # Remove multiple consecutive dots (quick fix for ... issues)
        cleaned_texts = [re.sub(r'\.{2,}', '.', t) for t in texts]
        # Remove multiple consecutive whitespaces
        cleaned_texts = [re.sub(r'\s+', ' ', t) for t in cleaned_texts]

        preproc_config = model_list["preprocessor"]
        chunk_size = preproc_config.get("chunk_size", 1000)
        min_sentences = preproc_config.get("min_sentences", 5)
        embedding_model_name = preproc_config.get("embedding_model", "all-MiniLM-L6-v2")

        match preproc_config["type"]:
            case "semantic_chunker":
                my_logger.info("Using SemanticChunker for document chunking.")
                chunker = SemanticChunker(
                    embedding_model=embedding_model_name,
                    chunk_size=chunk_size,
                    min_sentences=min_sentences,
                )

            case "recursive_chunker":

                my_logger.info("Using RecursiveChunker for document chunking.")
                chunker = RecursiveChunker(chunk_size=chunk_size)

            case "sentence_chunker":
                my_logger.info("Using SentenceChunker for document chunking.")
                chunker = SentenceChunker(
                    tokenizer="gpt2",
                    chunk_size=chunk_size,
                    chunk_overlap=50,
                    min_sentences_per_chunk=1,
                )
            case _:
                raise ValueError(
                    f"Preprocessor type {model_list['preprocessor']['type']} not supported."
                )

        docs = chunker(cleaned_texts)
        flat_chunks = [chunk for chunk_list in docs for chunk in chunk_list]

        chunked_documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_index": i,
                    "chunk_token_count": chunk.token_count,
                    "source": "",
                },
            )
            for i, chunk in enumerate(flat_chunks)
        ]

        my_logger.info(
            f"Chunked {len(documents)} pages into {len(chunked_documents)} chunks."
        )
        return chunked_documents
