import json
import time
from functools import wraps
from typing import Any, Callable, List

from deepeval.models.base_model import DeepEvalBaseLLM
from google.cloud import storage
from langchain.docstore.document import Document

# from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import JsonOutputParser

from ai_eval.config import global_config as glob
from ai_eval.services.logger import LoggerFactory

logger = LoggerFactory().create_module_logger()


def timer(func: Callable) -> Callable:
    """
    A decorator that measures the execution time of the decorated function.

    Args:
        func (Callable): The function to be wrapped and timed.

    Returns:
        Callable: The wrapped function that prints its execution time upon completion.

    Example:
        @timer
        def example_function():
            # Function implementation
            pass
    """

    @wraps(func)
    def wrapper_timer(*args: tuple, **kwargs: dict) -> Any:
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def retry(func: Callable) -> Callable:
    """
    A decorator that retries the decorated function up to 3 times in case of an exception.

    Args:
        func (Callable): The function to be retried.

    Returns:
        Callable: The wrapped function that retries upon failure.

    Example:
        @retry
        def example_function():
            # Function implementation
            pass
    """

    @wraps(func)
    def wrapper_retry(*args: tuple, **kwargs: dict) -> Any:
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}.\nRetrying...")
                if attempt == 2:
                    raise

    return wrapper_retry


def list_objects(bucket_name: str) -> List[str]:
    """Lists all the blobs/objects in the specified GCP bucket.

    Args:
        bucket_name (str): The name of the GCP bucket.

    Returns:
        List[str]: A list of blob names in the bucket.

    Example:
        list_objects('my-bucket')
    """
    try:
        storage_client = storage.Client(project=glob.GCP_PROJECT)
        blobs = storage_client.list_blobs(bucket_name)
        blob_names = [blob.name for blob in blobs]
        return blob_names
    except Exception as e:
        print(f"Error listing objects in bucket {bucket_name}: {e}")
        return []


class LangChainDeepEvalModel(DeepEvalBaseLLM):
    """
    A class to evaluate language models using the LangChain framework.

    Attributes:
        model (LLM): The language model to be evaluated.

    Methods:
        load_model() -> LLM:
            Loads and returns the language model.

        generate(prompt: str) -> str:
            Generates a response from the language model based on the given prompt.

        async a_generate(prompt: str) -> str:
            Asynchronously generates a response from the language model based on the given prompt.

        get_model_name() -> str:
            Returns the name of the model.
    """

    def __init__(self, model: LLM):
        self.model = model

    def load_model(self) -> LLM:
        return self.model

    def generate(self, prompt: str) -> str:

        chat_model = self.load_model()
        output = chat_model.predict(prompt)
        parser = JsonOutputParser()
        # Try to parse as JSON using LangChain's parser
        try:
            parsed = parser.parse(output)
            return json.dumps(parsed)
        except Exception:
            logger.warning("Failed to parse output as JSON using LangChain parser.")
            # Fallback: try to extract JSON substring with regex
            import re

            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try:
                    parsed = parser.parse(match.group(0))
                    return json.dumps(parsed)
                except Exception:
                    logger.warning("Failed to parse extracted JSON substring.")
                    pass
            # Final fallback: return error and raw output
            return json.dumps(
                {"error": "Invalid JSON from model", "raw_output": output}
            )

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.apredict(prompt)
        return res

    def get_model_name(self) -> str:
        return "LangChain Model"


def validate_documents(docs: List[Document]) -> None:
    """Validate that all items in the list are instances of Document.

    Args:
        docs: List of Document objects to validate.

    Raises:
        AssertionError: If any item is not a Document.
    """
    for doc in docs:
        assert isinstance(doc, Document), f"Expected Document, got {type(doc)}"
