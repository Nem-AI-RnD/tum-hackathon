from ai_eval.services.llm_engineering import LLMEngineering
from ai_eval.services.logger import LoggerFactory
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
from langfuse import Langfuse
import json

my_logger = LoggerFactory(handler_type="Stream").create_module_logger()

# Load environment variables from .env file
load_dotenv()

# Access variables
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

class LangfuseService(LLMEngineering):
    def __init__(self, verbose: bool = True):
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self.verbose = verbose
        if self.verbose:
            my_logger.info("Initializing Langfuse client...")
        # Check if the client is authenticated
        if self.client.auth_check():
            my_logger.info("Langfuse client is authenticated and ready!")
        else:
            my_logger.info(
                "Authentication failed. Please check your credentials and host."
            )

    def createDataset(self, name: str) -> None:
        """Create dataset"""
        try:
            if self.verbose:
                my_logger.info(f"Creating dataset with name: {name}")
            if name != '':
                dataset = self.client.get_dataset(name)
                if dataset:
                    my_logger.info(f"Dataset '{name}' already exists.")
                    return
                self.client.create_dataset(name)
                my_logger.info(f"{name} dataset created")
            else:
                my_logger.info("please provide proper name for the dataset")
        except Exception as e:
            # Check if the exception is due to dataset not found (404)
            if "404" in str(e):
                my_logger.info(f"Dataset '{name}' not found. Creating a new one.")
                dataset = self.client.create_dataset(
                    name=name,
                    description="Description of your dataset",
                    metadata={"created_by": "your_name"},
                )
                my_logger.info(f"Successfully created dataset '{name}'.")
            else:
                my_logger.info(f"Error occurred: {e}")
                raise

    def read(self, dataset_name: str) -> Any:
        """Read dataset"""
        try:
            dataset = self.client.get_dataset(dataset_name)
            return dataset
        except Exception as e:  # raise an exception if the dataset doesn't exist
            my_logger.error(f"Error getting dataset '{dataset_name}': {e}")
            raise

    def clean_null_bytes(self, s: Any) -> Any:
        """Remove null bytes from string values while preserving non-string types.

        Args:
            s: Input value of any type

        Returns:
            Cleaned string if input is string, otherwise original value
        """
        if isinstance(s, str):
            return s.replace("\x00", "")
        return s

    def update_from_pandas(
        self,
        dataset: Any,
        dataframe: Any,
        keys_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Inserts items from a Pandas DataFrame into the specified dataset.

        Args:
            dataset (Any): The Langfuse dataset object.
            dataframe (Any): The Pandas DataFrame containing the data to be inserted.
            keys_mapping (Dict[str, str], optional): A dictionary mapping DataFrame
                column names to Langfuse dataset field names. Defaults to None.
        """
        if self.verbose:
            my_logger.info(
                f"Inserting items from DataFrame into Langfuse dataset '{dataset}'"
            )
        try:
            if keys_mapping is None:
                keys_mapping = {}
            # Ensure the DataFrame is not empty
            if dataframe.empty:
                raise ValueError("The DataFrame is empty.")
            # Ensure the DataFrame has the required columns
            for i, row in dataframe.iterrows():
                data = {"question": row["question"], "context": row["context"]}
                input_data = {k: self.clean_null_bytes(v) for k, v in data.items()}

                json_string = json.dumps(input_data, ensure_ascii=False)
                size_in_bytes = len(json_string.encode('utf-8'))
                my_logger.info(f"Size of input data for row {i}: {size_in_bytes} bytes")
                self.client.create_dataset_item(
                    dataset_name=dataset,
                    input=input_data,
                    expected_output=row["answer"],
                )
        except Exception as e:
            my_logger.error(
                f"Error inserting items from DataFrame into dataset '{dataset}': {e}"
            )
            raise

    def insertItems(self, dataset_name: str, items: List[Dict[str, Any]]) -> None:
        """Insert items into dataset"""
        try:
            for item in items:
                if "metadata" in item.keys():
                    self.client.create_dataset_item(
                        dataset_name=dataset_name,
                        input=item["input"],
                        expected_output=item["expected_output"],
                        metadata=item["metadata"],
                        testing="data",
                    )
                else:
                    self.client.create_dataset_item(
                        dataset_name=dataset_name,
                        input=item["input"],
                        expected_output=item["expected_output"],
                    )
        except Exception as e:
            my_logger.info(f"Error inserting items into dataset: {e}")
        my_logger.info(
            f"Items {len(items)} inserted successfully in the dataset {dataset_name}"
        )

    def deleteDataset(self, name: str) -> None:
        """Delete dataset"""
        try:
            if name != "":
                self.client.delete_dataset(name)
                my_logger.info(f"{name} dataset deleted")
            else:
                my_logger.info("please provide the proper name for the dataset")
        except Exception as e:
            my_logger.info(f"Error deleting dataset '{name}': {e}")
            raise e

    def createPrompt(self, name: str, prompt: str) -> None:
        """Create a prompt"""
        try:
            if self.verbose:
                my_logger.info(f"Creating prompt with name: {name}")
            if name and prompt:
                self.client.create_prompt(name=name, prompt=prompt)
                my_logger.info(f"{name} prompt created")
            else:
                my_logger.info("Please provide proper name and prompt")
        except Exception as e:
            my_logger.info(f"Error creating prompt '{name}': {e}")
            raise e
