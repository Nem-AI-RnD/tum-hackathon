from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from typing_extensions import Literal


class OutputStructure(BaseModel):
    """
    Structured data for model output.
    The output is a JSON object with two fields:

    Evaluation: Your reasoning for the rating, e.g. citing specific examples of how the question is or is not useful to the user
    Score: Your rating as an integer number between 1 and 5, e.g. with 1 being the lowest and 5 being the highest (very useful)
    """

    Evaluation: str = Field(
        description="Your reasoning for the rating, citing specific examples of how the question is or is not useful to the user",
        examples=[
            "The question is not useful because it is not relevant to the context of the document."
        ],
    )
    Score: Literal["1", "2", "3", "4", "5"] = Field(
        description="Your rating as an integer number between 1 and 5 with 1 being the lowest and 5 being the highest (very useful)",
        examples=["5"],
    )


class DataIn(BaseModel):
    """
    DataIn is a Pydantic model that defines the schema for input data required for data generation.

    Attributes:
        blob_input_path (Optional[str]):
            Prefix (folder/name) to the input blob(s) to be used for data generation.
            This is an optional field.
        blob_output_path (str):
            Prefix (folder/name) to the dataset output CSV file uploaded to GCP.
            This is a required field.
        n_generations (int):
            Number of records to generate in the dataset.
            This is a required field.
    """

    blob_input_path: Optional[str] = Field(
        description="Prefix (folder/name) to the input blob(s) to be used for data generation",
        default="raw_documents/20240731_Nemetschek SE_Mitarbeiterhandbuch_Employee Handbook.pdf",
    )
    blob_output_path: str = Field(
        description="Prefix (folder/name) to the dataset output CSV file uploaded to GCP",
        default="generated_test_data.csv",
    )
    n_generations: int = Field(
        description="Number of records to generate in the dataset",
        default=5,
    )


class DataOut(BaseModel):
    """
    DataOut is a Pydantic model that represents the output data schema.

    Attributes:
        request_id (str): A unique identifier for the request.
        outputs (List[Dict[str, Union[str, int]]]): A list of dictionaries containing the generated data.
            Each dictionary can have string or integer values.
    """

    # success: bool = Field(description="True if data generation was successful")
    request_id: str = Field(description="Unique ID for the request")
    outputs: List[Dict[str, Union[str, int]]] = Field(description="Generated data")


class DataInRAG(BaseModel):
    """
    DataInRAG

    A Pydantic model representing the schema for data input in a Retrieval-Augmented Generation (RAG) pipeline.

    Attributes:
        prefix (Optional[str]):
            Prefix (folder name) to the input blob(s) to be used for data generation.
            Defaults to "raw_documents".
        test_data_filename (str):
            Filename of the CSV test data set uploaded to GCP.
            Defaults to "generated_test_data.csv".
        deepeval_data_filename (str):
            Filename of the DeepEval dataset saved as a JSON file to GCP.
            Defaults to "deepeval_test_data.json".
    """

    prefix: Optional[str] = Field(
        description="Prefix (folder name) to the input blob(s) to be used for data generation",
        default="raw_documents",
    )
    test_data_filename: str = Field(
        description="Filename of the CSV test data set uploaded to GCP",
        default="generated_test_data.csv",
    )
    deepeval_data_filename: str = Field(
        description="Filename of the DeepEval dataset saved as JSON file to GCP",
        default="deepeval_test_data.json",
    )


class DataOutRAG(BaseModel):
    """
    DataOutRAG is a Pydantic model that represents the output data structure for a RAG (Retrieval-Augmented Generation) process.

    Attributes:
        request_id (str): A unique identifier for the request.
        deepeval_test_set (List[Dict[str, str]]): A list of dictionaries containing the DeepEval test set data.
    """

    request_id: str = Field(description="Unique ID for the request")
    deepeval_test_set: List[Dict] = Field(
        description="List of dictionaries containing the DeepEval test set data"
    )
    # success: bool = Field(description="True if data generation was successful")


class DataInExp(BaseModel):
    """
    DataInExp is a Pydantic model that represents the input data for experiments.

    Attributes:
        experiment (str): Name of the experiment to be executed.
        status (Optional[str]): Current status of the experiment.
        request_id (Optional[str]): A unique identifier for the request.
        outputs (Optional[List[str]]): List of output results from the experiment.
    """
    experiment: str
    status: Optional[str] = None
    request_id: Optional[str] = None
    outputs: Optional[List[str]] = None


class DataOutExp(BaseModel):
    """
    DataOutExp is a Pydantic model that represents the output data for experiments.

    Attributes:
        request_id (str): A unique identifier for the request.
        experiment_name (str): Name of the experiment.
        status (Optional[str]): Current status of the experiment.
        outputs (Optional[List[str]]): List of output results from the experiment.
    """
    request_id: str 
    experiment_name: str
    status: Optional[str] = None
    outputs: Optional[List[str]] = None