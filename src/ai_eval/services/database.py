from typing import Any, Dict, List, Optional

from google.cloud.firestore import Client

from ai_eval.config import global_config as glob
from ai_eval.services.logger import LoggerFactory
from ai_eval.utils.utils import timer

my_logger = LoggerFactory().create_module_logger()


class FirestoreService:
    def __init__(
        self,
        project: str = glob.GCP_PROJECT,
        database: str = glob.GCP_FIRESTORE,
        collection_name: str = glob.GCP_FIRESTORE_COLLECTION,
        timeout: int = 600,  # Default timeout of 600 seconds
        verbose: bool = True,
        **client_kwargs: Any,
    ):
        """
        Initializes the database service with the specified Firestore configuration.

        Args:
            project (str): The Google Cloud project ID. Defaults to `glob.GCP_PROJECT`.
            database (str): The Firestore database name. Defaults to `glob.GCP_FIRESTORE`.
            collection_name (str): The name of the Firestore collection to interact with.
                Defaults to `glob.GCP_FIRESTORE_COLLECTION`.
            timeout (int): The timeout for Firestore operations in seconds. Defaults to 600.
            verbose (bool): If True, logs initialization details. Defaults to True.
            **client_kwargs (Any): Additional keyword arguments to pass to the Firestore client.

        Attributes:
            db (Client): The Firestore client instance.
            collection_name (str): The name of the Firestore collection.
            collection (CollectionReference): Reference to the Firestore collection.
            verbose (bool): Indicates whether verbose logging is enabled.
            _doc_ids_cache (list): Cached list of all document IDs in the collection.

        Logs:
            Logs the database string and initialized collection name if `verbose` is True.
        """
        self.verbose = verbose
        self.timeout = timeout
        my_logger.info("Initializing Firestore client...")
        my_logger.info(f"Project: {project}")
        try:
            self.db = Client(project=project, database=database, **client_kwargs)
            self.collection_name = collection_name
            self.collection = self.db.collection(collection_name)
            self._doc_ids_cache = self._get_all_document_ids()
        except Exception as e:
            my_logger.error(f"Error initializing Firestore client: {str(e)}")
            raise e

        my_logger.info(f"\nDatabase: {self.db._database_string}")
        my_logger.info(f"Initialized Firestore collection: {collection_name}")

    @property
    def document_ids(self) -> List[str]:
        """
        Property to get the cached document IDs.

        Returns:
            List[str]: List of document IDs.
        """
        return self._doc_ids_cache

    def create(self, document_data: Dict[str, Any]) -> str:
        """
        Create a new document in the collection.

        :param document_data: Dictionary containing the document data
        :return: The ID of the newly created document
        """
        try:
            doc_ref = self.collection.add(document_data)[1]
            return doc_ref.id
        except Exception as e:
            my_logger.error(f"Error creating document: {str(e)}")
            return ""

    def read(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a document from the collection by its ID.

        :param document_id: The ID of the document to read
        :return: The document data as a dictionary, or None if not found
        """
        try:
            doc_ref = self.collection.document(document_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            my_logger.error(f"Error reading document: {str(e)}")
            return None

    def update(self, document_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing document in the collection.

        :param document_id: The ID of the document to update
        :param update_data: Dictionary containing the fields to update
        :return: True if the update was successful, False otherwise
        """
        try:
            docs = self.collection.where("uid", "==", document_id).stream()
            for doc in docs:
                doc.reference.update(update_data)
                return True
            return False
        except Exception as e:
            my_logger.error(f"Error updating document: {str(e)}")
            return False

    def delete(self, document_id: str) -> bool:
        """
        Delete a document from the collection by its ID.

        :param document_id: The ID of the document to delete
        :return: True if the deletion was successful, False otherwise
        """
        try:
            doc_ref = self.collection.document(document_id)
            if doc_ref.get().exists:
                doc_ref.delete()
                return True
            return False
        except Exception as e:
            my_logger.error(f"Error deleting document: {str(e)}")
            return False

    def list_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List documents in the collection.

        :param limit: Maximum number of documents to return
        :return: A list of document data dictionaries
        """
        docs = self.collection.limit(limit).stream(timeout=self.timeout)
        return [doc.to_dict() for doc in docs]

    def query_documents(
        self, field: str, operator: str, value: Any
    ) -> List[Dict[str, Any]]:
        """
        Query documents in the collection based on a field, operator, and value.

        :param field: The field to query
        :param operator: The comparison operator (e.g., '==', '>', '<', '>=', '<=')
        :param value: The value to compare against
        :return: A list of matching document data dictionaries
        """
        try:
            query = self.collection.where(field, operator, value)
            return [doc.to_dict() for doc in query.stream(timeout=self.timeout)]
        except Exception as e:
            my_logger.error(f"Error querying documents: {str(e)}")
            return []

    @timer
    def _get_all_document_ids(self) -> List[str]:
        """
        Retrieve all document IDs from the Firestore collection.

        Returns:
            List[str]: List of document IDs.
        """
        docs = self.collection.stream(timeout=self.timeout)
        doc_ids = [doc.id for doc in docs]
        # if self.verbose:
        #     print(f"Retrieved {len(doc_ids)} document IDs.")
        return doc_ids

    def delete_collection(self, batch_size: int = 500) -> bool:
        """
        Delete the entire collection in batches.

        Args:
            batch_size (int): Number of documents to delete in each batch.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            docs = self.collection.limit(batch_size).stream(timeout=self.timeout)
            deleted = 0

            for doc in docs:
                doc.reference.delete()
                deleted += 1
                if self.verbose:
                    my_logger.info(f"Deleted document {deleted}: {doc.id}")

            if deleted >= batch_size:
                return self.delete_collection(
                    batch_size
                )  # Recursive call for remaining documents

            if self.verbose:
                my_logger.info(
                    f"Successfully deleted collection '{self.collection_name}' ({deleted} documents)"
                )

            # Clear the document IDs cache
            self._doc_ids_cache = []
            return True

        except Exception as e:
            my_logger.error(f"Error deleting collection: {str(e)}")
            return False
