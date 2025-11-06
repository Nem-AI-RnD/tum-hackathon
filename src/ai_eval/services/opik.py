import os
from typing import Dict, Any, List, Optional, Literal
from opik import Opik
from ai_eval.config import global_config as glob
from ai_eval.services.logger import LoggerFactory

my_logger = LoggerFactory(handler_type="Stream").create_module_logger()


class OpikService:
    """
    A service class for performing CRUD operations on Opik datasets.
    """

    def __init__(self, project_name: Optional[str] = None, verbose: bool = True):
        """
        Initializes the OpikService with the project name.

        Args:
            project_name (str, optional): The name of the Opik project.
                Defaults to the OPIK_PROJECT_NAME environment variable.
            verbose (bool): Flag to enable verbose output. Default is True.
        """
        self.verbose = verbose
        self.project_name = project_name or glob.OPIK_PROJECT_NAME
        if not self.project_name:
            raise ValueError(
                "Opik project name must be provided either as an argument or"
                " through the OPIK_PROJECT_NAME environment variable."
            )
        self.client = Opik(project_name=self.project_name)
        if self.verbose:
            my_logger.info(
                f"Opik client initialized with project '{self.project_name}'"
            )

    def create(self, name: str) -> Any:  # Replace Any with actual Opik Dataset type
        """
        Create dataset or creates if it doesn't exist.

        Args:
            name (str): The name of the dataset.

        Returns:
            Any: The Opik dataset object.
        """
        if self.verbose:
            my_logger.info(f"Creating/Getting Opik dataset '{name}'")
        try:
            return self.client.get_or_create_dataset(name=name)
        except Exception as e:
            my_logger.error(f"Error creating dataset '{name}': {e}")
            raise
        return None

    def update_from_items(self, dataset: Any, items: List[Dict[str, Any]]) -> None:
        """
        Inserts multiple items into the specified dataset.

        Args:
            dataset (Any): The Opik dataset object.
            items (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                represents an item to be inserted into the dataset.
        """
        if self.verbose:
            my_logger.info(
                f"Inserting {len(items)} items into Opik dataset '{dataset.name}'"
            )
        try:
            dataset.insert(items)
            my_logger.info(
                f"Inserted {len(items)} items into Opik dataset '{dataset.name}'"
            )
        except Exception as e:
            my_logger.error(f"Error inserting items into dataset '{dataset.name}': {e}")

    def update_from_json(self, dataset: Any, file_path: str) -> None:
        """
        Reads items from a JSONL file and inserts them into the specified dataset.

        Args:
            dataset (Any): The Opik dataset object.
            file_path (str): The path to the JSONL file.
        """
        if self.verbose:
            my_logger.info(f"Reading items from JSONL file '{file_path}'")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        if self.verbose:
            my_logger.info(
                f"Inserting items from JSONL file '{file_path}' into Opik dataset '{dataset.name}'"
            )
        try:
            dataset.insert_from_jsonl(file_path=file_path)
        except Exception as e:
            my_logger.error(
                f"Error inserting items from JSONL file '{file_path}' into dataset '{dataset.name}': {e}"
            )
            raise

    def update_from_pandas(
        self,
        dataset: Any,
        dataframe: Any,
        keys_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Inserts items from a Pandas DataFrame into the specified dataset.

        Args:
            dataset (Any): The Opik dataset object.
            dataframe (Any): The Pandas DataFrame containing the data to be inserted.
            keys_mapping (Dict[str, str], optional): A dictionary mapping DataFrame
                column names to Opik dataset field names. Defaults to None.
        """
        if self.verbose:
            my_logger.info(
                f"Inserting items from DataFrame into Opik dataset '{dataset.name}'"
            )
        try:
            if keys_mapping is None:
                keys_mapping = {}
            # Ensure the DataFrame is not empty
            if dataframe.empty:
                raise ValueError("The DataFrame is empty.")
            # Ensure the DataFrame has the required columns

            dataset.insert_from_pandas(dataframe=dataframe, keys_mapping=keys_mapping)
        except Exception as e:
            my_logger.error(
                f"Error inserting items from DataFrame into dataset '{dataset.name}': {e}"
            )
            raise

    def delete_items(self, dataset: Any, items_ids: List[str]) -> None:
        """
        Deletes items from the specified dataset.

        Args:
            dataset (Any): The Opik dataset object.
            items_ids (List[str]): A list of item IDs to delete.
        """
        if self.verbose:
            my_logger.info(
                f"Deleting {len(items_ids)} items from Opik dataset '{dataset.name}'"
            )
        try:
            dataset.delete(items_ids=items_ids)
        except Exception as e:
            my_logger.error(f"Error deleting items from dataset '{dataset.name}': {e}")
            raise

    def read(self, name: str, output_format: Literal["json", "pandas"]) -> Any:
        """
        Retrieves an existing dataset.

        Args:
            name (str): The name of the dataset.
            output_format (Literal["json", "pandas"]): The format in which to return the dataset.
                Can be either "json" or "pandas".

        Returns:
            Any: The Opik dataset object, or None if the dataset does not exist.
        """
        try:
            dataset = self.client.get_dataset(name=name)
            match output_format:
                case "json":
                    data = dataset.to_json()
                case "pandas":
                    data = dataset.to_pandas()
                case _:
                    raise ValueError(f"Unsupported output format: {output_format}")
            return data
        except Exception as e:  # raise an exception if the dataset doesn't exist
            print(f"Error getting dataset '{name}': {e}")
            return None

    def clear(self, name: str) -> None:
        """
        Clears the specified dataset.

        Args:
            name (str): The name of the dataset to clear.
        """
        try:
            dataset = self.client.get_dataset(name=name)
            dataset.clear()
            if self.verbose:
                my_logger.info(f"Cleared Opik dataset '{name}'")
        except Exception as e:
            my_logger.error(f"Error clearing dataset '{name}': {e}")
            raise
