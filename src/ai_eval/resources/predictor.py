import os
from typing import List, Dict, Optional
from tqdm.auto import tqdm
import pandas as pd
from langchain.docstore.document import Document
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from ai_eval.resources.llm_aaj import (
    build_local_vectorstore,
    answer_with_rag,
)
from ai_eval.config import global_config as glob
from ai_eval.services.logger import LoggerFactory
from ai_eval.resources.get_models import InitModels
from ai_eval.services.file import JSONService
from ai_eval.services.file_gcp import CSVService, JSONService as JSONService_GCP

# from ai_eval.services.opik import OpikService

my_logger = LoggerFactory(handler_type="Stream").create_module_logger()

model = InitModels(load_models=["rag_generator"])


class Predictor:
    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        self.vectorstore = None
        # self.opik_service = OpikService()
        self.rag_generator = model.rag_generator
        self.docs_processed: List[Document] = []
        self.data = pd.DataFrame()

    def load_test_data(
        self,
        prefix: str = "raw_documents",
        filename: str = "generated_test_data.csv",
        source: str = "gcp",
    ) -> List[Document]:
        """
        Loads test documents from cloud storage or local file and processes them into a list of Document objects.

        Args:
            filename (str): The name of the CSV file to load test data from. Defaults to "generated_test_data.csv".
            prefix (str): The prefix for the blob path (GCP) or local directory. Defaults to "raw_documents".
            source (str): "gcp" to load from cloud storage, "local" to load from local file.

        Returns:
            List[Document]: A list of chunked Document objects created from the loaded test data.
        """
        if source == "gcp":
            my_logger.info("Loading prepared test cases from cloud storage...")
            self.data = CSVService(
                root_path=prefix, path=filename, verbose=self.verbose
            ).doRead()

        elif source == "local":
            my_logger.info("Loading prepared test cases from local file...")
            if filename.endswith(".json"):
                # Use JSONService to read local JSON file and convert to DataFrame
                json_service = JSONService(
                    path=filename, root_path=prefix, verbose=self.verbose
                )
                self.data = pd.DataFrame(json_service.doRead())
            else:
                raise ValueError("Only JSON files are supported for local source.")
        else:
            raise ValueError("Source must be 'gcp' or 'local'.")

        # Convert the list of strings to a list of Document objects
        self.docs_processed = [
            Document(page_content=context, metadata={"source": filename})
            for context in self.data["context"].tolist()
        ]
        my_logger.info(f"Loaded {len(self.docs_processed)} documents.")
        return self.docs_processed

    def create_test_cases(self) -> EvaluationDataset:
        """
        Creates test cases for DeepEval from the evaluated QA pairs.

        Returns:
            EvaluationDataset: The dataset containing the test cases.
        """
        # Extract questions from the data if needed elsewhere
        assert hasattr(self, "vectorstore"), "Vector store not created yet."
        sample_queries = self.data["question"].tolist()
        expected_responses = self.data["answer"].tolist()
        input_contexts = self.data["context"].tolist()

        test_cases = []
        predicted_contexts, predicted_answers = [], []
        for query, ground_truth_answer, ground_truth_context in tqdm(
            zip(sample_queries, expected_responses, input_contexts),
            total=len(sample_queries),
            desc="Building test dataset",
        ):
            # Answer the question with RAG ('predictions')
            # ---------------------------------------------
            # Replace this part in the future with brand-spec. RAG node!
            try:
                predicted_answer, predicted_context = answer_with_rag(
                    question=query, llm=self.rag_generator, vectorstore=self.vectorstore
                )

                predicted_answers.append(predicted_answer.content)
                predicted_contexts.append(predicted_context[0])

                test_case = LLMTestCase(
                    input=query,
                    actual_output=predicted_answer.content,
                    expected_output=ground_truth_answer,
                    retrieval_context=predicted_context,
                    context=[ground_truth_context],
                )
            except Exception as e:
                my_logger.error(f"Error processing query '{query}': {e}")
                test_case = LLMTestCase(
                    input=query,
                    actual_output="",
                    expected_output=ground_truth_answer,
                    retrieval_context=[],
                    context=ground_truth_context,
                )
            test_cases.append(test_case)
        my_logger.info(f"Created {len(test_cases)} test cases.")

        # Save the test cases to the dataset
        self.data["predicted_answer"] = predicted_answers
        self.data["predicted_context"] = predicted_contexts
        return EvaluationDataset(test_cases=test_cases)

    def predict(self, **kwargs: Dict[str, str]) -> pd.DataFrame:
        """
        Main method to load test data, create vectorstore, and generate test cases.

        Args:
            **kwargs: Additional keyword arguments for loading test data.

        Returns:
            Pandas dataframe: The dataset containing the test cases.
        """
        docs_processed = self.load_test_data(**kwargs)
        # Replace this part in the future with brand-spec. vectorsearch!
        self.vectorstore = build_local_vectorstore(docs_processed)
        self.evaluation_dataset = self.create_test_cases()
        # return self.evaluation_dataset
        return self.data

    def save_deepeval_dataset(
        self,
        dataset: Optional[EvaluationDataset] = None,
        filename: str = "deepeval_test_data.json",
    ) -> None:
        """
        Saves the DeepEval dataset to a JSON file.

        Args:
            dataset (EvaluationDataset, optional): The dataset to save. If None, uses the current dataset.
                Defaults to None.
            filename (str): The name of the file to save the dataset to. Defaults to "deepeval_test_data.json".
        """
        if dataset is None and not hasattr(self, "evaluation_dataset"):
            my_logger.error("No dataset provided to save.")
            raise ValueError("Dataset cannot be None when saving DeepEval dataset.")
        elif dataset is None:
            dataset = self.evaluation_dataset

        my_logger.info("Saving DeepEval dataset...")
        try:
            dataset.save_as(
                file_type="json",
                directory=os.path.join(glob.DATA_PKG_DIR, "tmp"),
                include_test_cases=True,
            )
        except Exception as e:
            my_logger.error(f"Error saving dataset: {e}")
            return

        # Grab the local file name
        local_tmp_json = os.listdir(os.path.join(glob.DATA_PKG_DIR, "tmp"))[0]
        my_logger.info(f"Saving DeepEval dataset to {local_tmp_json}...")

        # Read the local file
        js = JSONService(
            path=local_tmp_json, root_path=os.path.join(glob.DATA_PKG_DIR, "tmp")
        )
        self.my_set = js.doRead()

        # Upload the local file to GCP
        json_service = JSONService_GCP(
            root_path="raw_documents",
            path=filename,
        )
        json_service.doWrite(X=self.my_set)

        my_logger.info(f"Saved DeepEval dataset to {filename}.")
        # And finally delete the local file
        os.remove(os.path.join(glob.DATA_PKG_DIR, "tmp", local_tmp_json))
        my_logger.info(f"Deleted local file {local_tmp_json}.")

    def upload_to_opik(self, dataset_name: str) -> None:
        """
        Uploads the generated pandas dataframe to Opik.

        Args:
            dataset_name (str): The name of the dataset in Opik.
        """
        try:
            # Create or get the dataset
            dataset = self.opik_service.create(name=dataset_name)
            # Upload the dataframe to Opik
            self.opik_service.update_from_pandas(dataset, self.data.fillna(0.0))
            if self.verbose:
                my_logger.info(f"Dataset uploaded to Opik: {dataset_name} 📁")
        except Exception as e:
            if self.verbose:
                my_logger.error(f"Error uploading to Opik: {e}")
            raise e
