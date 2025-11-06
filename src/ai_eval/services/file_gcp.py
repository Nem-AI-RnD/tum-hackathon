import os
from io import StringIO, BytesIO
from typing import Generator, List, Optional, Any, Dict
import pandas as pd
from pypdf import PdfReader
from google.api_core.exceptions import NotFound
from langchain.docstore.document import Document
from ai_eval.services.logger import LoggerFactory
from ai_eval.services.clients import GCPClient
from ai_eval.services.blueprint_file import BaseService
from ai_eval.config import global_config as glob
import json

my_logger = LoggerFactory(handler_type="Stream", verbose=True).create_module_logger()


def stream_gcs_pdf(
    blob_path: str, **meta_info_fields: Optional[Dict]
) -> Generator[Document, None, None]:
    """
    Reads the PDF and returns a generator of LangChain Documents with extracted text and metadata.

    Args:
        blob_path (str): The path to the blob representing the PDF file.
        meta_info_fields (Optional[dict]): Additional metadata fields to include in the Document.

    Yields:
        Document: A LangChain Document containing extracted text and page metadata.
    """
    client = GCPClient(bucket_name=glob.GCP_CS_BUCKET)
    blob = client.get_blob(blob_path)

    with blob.open("rb") as pdf_stream:
        pdf_reader = PdfReader(pdf_stream)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            yield Document(
                page_content=text,
                metadata={"page": i, "source": blob_path, **meta_info_fields},
            )


class PDFService(BaseService):
    """Service for reading and writing PDF files in GCP Cloud Storage."""

    def __init__(
        self,
        path: str = "",
        root_path: str = "",
        verbose: bool = True,
    ):
        """
        Initialize the PDF service.

        Args:
            path (str): Path/Prefix (subfolder + filename) to the PDF file in the bucket.
            root_path (str): Root path (parent folder) for the file.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.client = GCPClient(glob.GCP_CS_BUCKET)
        self.blob = self.client.get_blob(self.path)
        self.verbose = verbose

    def doRead(self, local_pdf_path: str) -> None:
        """
        Download a PDF file from GCP Cloud Storage to a local file path.

        Args:
            local_pdf_path (str): The local path where the PDF file will be saved.

        Returns:
            None
        """
        try:
            my_logger.info(f"Downloading PDF file from path: {self.path}")
            with open(local_pdf_path, "wb") as pdf_file:
                self.blob.download_to_file(pdf_file)
            if self.verbose:
                my_logger.info(f"PDF file downloaded to: {local_pdf_path}")
        except NotFound:
            my_logger.error(f"Blob not found in bucket: {self.path}")
            raise
        except Exception as e:
            my_logger.error(
                f"Failed to download PDF file from path: {self.path}. Error: {e}"
            )
            raise

    def doWrite(self, X: BytesIO) -> str:
        """
        Upload a PDF file to GCP Cloud Storage using an in-memory file-like object.

        Args:
            X (BytesIO): Byte stream representing the PDF file. The in-memory file-like object containing the PDF data.

        Returns:
            str: A success message.
        """
        try:
            assert isinstance(X, BytesIO), "Input must be a BytesIO object."
            # Ensure the stream is at the beginning
            X.seek(0)
            self.blob.upload_from_file(X, content_type="application/pdf")
            if self.verbose:
                my_logger.info(f"PDF Service Output to File: {self.path}")
            my_logger.info(f"Successfully uploaded PDF file to path: {self.path}")
            return f"Successfully uploaded PDF file to path: {self.path}"
        except Exception as e:
            my_logger.error(
                f"Failed to upload PDF file to path: {self.path}. Error: {e}"
            )
            raise

    # def doWrite(self, local_pdf_path: str) -> str:
    #     """
    #     Upload a locally stored PDF file to GCP Cloud Storage.

    #     Args:
    #         local_pdf_path (str): The local path to the PDF file to upload.

    #     Returns:
    #         str: A success message.
    #     """
    #     try:
    #         my_logger.info(f"Uploading PDF file to path: {self.path}")
    #         with open(local_pdf_path, "rb") as pdf_file:
    #             self.blob.upload_from_file(pdf_file, content_type="application/pdf")
    #         if self.verbose:
    #             my_logger.info(f"PDF Service Output to File: {self.path}")
    #         my_logger.info(f"Successfully uploaded PDF file to path: {self.path}")
    #         return f"Successfully uploaded PDF file to path: {self.path}"
    #     except FileNotFoundError:
    #         my_logger.error(f"Local PDF file not found: {local_pdf_path}")
    #         raise
    #     except Exception as e:
    #         my_logger.error(
    #             f"Failed to upload PDF file to path: {self.path}. Error: {e}"
    #         )
    #         raise


class CSVService(BaseService):
    """Service for reading and writing CSV files in GCP Cloud Storage."""

    def __init__(
        self,
        path: str = "",
        root_path: str = "",
        delimiter: str = "\t",
        encoding: str = "UTF-8",
        schema_map: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Initialize the CSV service.

        Args:
            path (Optional[str]): The name of the GCP bucket.
            root_path (Optional[str]): Root path for the file.
            delimiter (str): Delimiter used in the CSV file.
            encoding (str): Encoding of the CSV file.
            schema_map (Optional[dict]): Mapping of column names for renaming.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.client = GCPClient(glob.GCP_CS_BUCKET)
        self.blob = self.client.get_blob(self.path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.schema_map = schema_map
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> pd.DataFrame:
        try:
            my_logger.info(f"Reading CSV file from path: {self.path}")
            data = self.blob.download_as_text(encoding=self.encoding)
            df = pd.read_csv(StringIO(data), delimiter=self.delimiter, **kwargs)
            if self.schema_map:
                df.rename(columns=self.schema_map, inplace=True)
            if self.verbose:
                my_logger.info(f"CSV Service Read from File: {self.path}")
            my_logger.info(f"Successfully read CSV file from path: {self.path}")
            return df
        except Exception as e:
            my_logger.error(
                f"Failed to read CSV file from path: {self.path}. Error: {e}"
            )
            raise

    def doWrite(self, X: pd.DataFrame, **kwargs: Any) -> None:
        try:
            my_logger.info(f"Writing CSV file to path: {self.path}")
            csv_buffer = StringIO()
            X.to_csv(
                csv_buffer,
                encoding=self.encoding,
                sep=self.delimiter,
                index=False,
                **kwargs,
            )
            self.blob.upload_from_string(csv_buffer.getvalue())
            if self.verbose:
                my_logger.info(f"CSV Service Output to File: {self.path}")
            my_logger.info(f"Successfully wrote CSV file to path: {self.path}")
        except Exception as e:
            my_logger.error(
                f"Failed to write CSV file to path: {self.path}. Error: {e}"
            )
            raise


class JSONService(BaseService):
    """Service for reading and writing JSON files in GCP Cloud Storage."""

    def __init__(
        self,
        path: str = "",
        root_path: str = "",
        verbose: bool = False,
    ):
        """
        Initialize the JSON service.

        Args:
            path (str): Path to the JSON file in the bucket.
            root_path (Optional[str]): Root path for the file.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.client = GCPClient(glob.GCP_CS_BUCKET)
        self.blob = self.client.get_blob(self.path)
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> Dict:
        try:
            my_logger.info(f"Reading JSON file from path: {self.path}")
            content = self.blob.download_as_text()
            json_data = json.loads(content)
            if self.verbose:
                my_logger.info(f"JSON Service Read from File: {self.path}")
            my_logger.info(f"Successfully read JSON file from path: {self.path}")
            return json_data
        except Exception as e:
            my_logger.error(
                f"Failed to read JSON file from path: {self.path}. Error: {e}"
            )
            raise e

    def doWrite(self, X: Dict, **kwargs: Any) -> None:
        try:
            my_logger.info(f"Writing JSON file to path: {self.path}")
            json_data = json.dumps(X, **kwargs)
            self.blob.upload_from_string(json_data)
            if self.verbose:
                my_logger.info(f"JSON Service Output to File: {self.path}")
            my_logger.info(f"Successfully wrote JSON file to path: {self.path}")
        except Exception as e:
            my_logger.error(
                f"Failed to write JSON file to path: {self.path}. Error: {e}"
            )
            raise e


class TXTService(BaseService):
    """Service for reading and writing text files in GCP Cloud Storage."""

    def __init__(
        self,
        path: str = "",
        root_path: str = "",
        verbose: bool = False,
    ):
        """
        Initialize the TXT service.

        Args:
            bucket_name (str): The name of the GCP bucket.
            path (Optional[str]): Path to the text file in the bucket.
            root_path (Optional[str]): Root path for the file.
            verbose (bool): Enable verbose logging.
        """
        super().__init__(path, root_path)
        self.path = os.path.join(root_path, path)
        self.client = GCPClient(glob.GCP_CS_BUCKET)
        self.blob = self.client.get_blob(self.path)
        self.verbose = verbose

    def doRead(self, **kwargs: Any) -> List[str]:
        """
        Read the text file from GCP Cloud Storage as a byte stream and decode it.

        Returns:
            List[str]: A list of lines from the text file.
        """
        try:
            my_logger.info(f"Reading TXT file from path: {self.path}")
            content = self.blob.download_as_text()
        except NotFound:
            print("File not found in GCS")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []
        return content.splitlines()

    def doWrite(self, X: List[str]) -> None:
        """
        Write the input list of strings to a text file in GCP Cloud Storage.

        Args:
            X (List[str]): The content to write to the text file. Must be a list of strings.
        Returns:
            None
        """
        assert isinstance(X, list), "Input must be a list of strings."
        try:
            my_logger.info(f"Writing text file to path: {self.path}")
            content = "\n".join(X)
            blob = self.client.get_blob(self.path)
            blob.upload_from_string(content)  # Upload the joined string as text
            if self.verbose:
                my_logger.info(f"TXT Service Output to File: {self.path}")
            my_logger.info(f"Successfully wrote text file to path: {self.path}")
        except Exception as e:
            my_logger.error(
                f"Failed to write text file to path: {self.path}. Error: {e}"
            )
            raise e
