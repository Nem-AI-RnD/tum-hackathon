from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.language_models.llms import LLM

# from ai_eval.services.langfuse import LangfuseService
from ai_eval.config import global_config as glob
from ai_eval.resources import prompts as pr
from ai_eval.resources.llm_aaj import (
    eval_qa_couples,
    generate_qa_couples,
    generate_qa_memory,
)
from ai_eval.resources.preprocessor import Preprocessor
from ai_eval.services.database import FirestoreService
from ai_eval.services.file import XLSXService
from ai_eval.services.file_gcp import CSVService
from ai_eval.services.logger import LoggerFactory

my_logger = LoggerFactory().create_module_logger()


class TestDatasetCreator(Preprocessor):
    """
    A class to create a DeepEval test dataset.
    """

    def __init__(
        self,
        blob_path: Optional[str],
        qa_generator_llm: LLM,
        eval_llm: LLM,
        source: str = "gcp",
        verbose: bool = True,
    ):
        """
        Initializes the DeepEvalTestDatasetCreator with the given parameters.

        Args:
            blob_path (Optional[str]): The path to the blob representing the PDF file. Can be None.
            qa_generator_llm (LLM): The LLM to use for QA generation.
            eval_llm (LLM): The LLM to use for evaluation.
            verbose (bool): Flag to enable verbose output. Default is True.
        """
        super().__init__()
        self.blob_path = blob_path
        self.qa_generator_llm = qa_generator_llm
        self.eval_llm = eval_llm
        self.docs_processed: Optional[List[Document]] = None
        self.sampled_contexts: Optional[List[Document]] = None
        self.generated_questions: Optional[pd.DataFrame] = None

        self.verbose = verbose
        self.source = source
        if self.verbose:
            print("\nStarting Test Dataset Creation...")

    def preprocess_documents(self) -> None:
        """
        Loads and chunks the raw documents from the specified GCP bucket.
        """
        if self.verbose:
            my_logger.info("Loading and chunking documents...")

        # Download the PDF files from GCP
        raw_data = self.fetch_documents(blob_path=self.blob_path, source=self.source)

        # Chunk the loaded documents
        self.docs_processed = self.chunk_documents(raw_data)

    def generate_qa_pairs(
        self, n_generations: int = 20, with_replacement: bool = False
    ) -> None:
        """
        Generates question/answer pairs from the processed documents.

        Args:
            n_generations (int): Number of QA pairs to generate. Default is 20.
        """
        assert self.docs_processed is not None, "Documents not loaded yet."

        prompt = PromptTemplate(
            # template=pr.QA_generation_prompt,
            # input_variables=["context"],
            template=pr.QA_generation_prompt_with_memory,
            input_variables=["history", "context"],
        )

        # Without memory:
        # self.generated_questions = generate_qa_couples(
        #     self.docs_processed,
        #     prompt,
        #     self.qa_generator_llm,
        #     n_generations,
        #     with_replacement=with_replacement,
        # )

        # With memory:
        self.generated_questions = generate_qa_memory(
            self.docs_processed,
            prompt,
            self.qa_generator_llm,
            n_generations,
            self.sampled_contexts,
            with_replacement=with_replacement,
        )

    def evaluate_qa_pairs(self) -> None:
        """
        Evaluates the generated question/answer pairs.
        """
        assert self.generated_questions is not None, "QA pairs not generated yet."

        prompt_groundedness = PromptTemplate(
            template=pr.question_groundedness_critique_prompt,
            input_variables=["question", "context"],
        )
        prompt_relevancy = PromptTemplate(
            template=pr.question_relevance_critique_prompt, input_variables=["question"]
        )

        prompt_faithfulness = PromptTemplate(
            template=pr.answer_faithfulness_critique_prompt,
            input_variables=["context", "question", "answer"],
        )

        prompt_answer_evaluator = PromptTemplate(
            template=pr.answer_location_dependency_prompt,
            input_variables=["context", "question", "answer"],
        )
        # print("Count", len(self.generated_questions))
        self.outputs = eval_qa_couples(
            self.generated_questions,
            prompt_groundedness,
            prompt_relevancy,
            prompt_faithfulness,
            prompt_answer_evaluator,
            self.eval_llm,
        )

        self.generated_qa = pd.DataFrame.from_dict(self.outputs)

        self.generated_qa = self.generated_qa[
            [
                "index",
                "question",
                "answer",
                "location_dependency_evaluator_target_answer",
                "context",
                "groundedness_score",
                "groundedness_eval",
                "question_relevancy_score",
                "question_relevancy_eval",
                "faithfulness_score",
                "faithfulness_eval",
                # "location_dependency_evaluator_score",
                # "location_dependency_evaluator_eval",
                # "location_dependency_evaluator_question",
                # "location_dependency_evaluator_answer",
            ]
        ]

        print(f"Filtered QA pairs: {len(self.generated_qa)} out of {len(self.outputs)}")


class Synthesizer(TestDatasetCreator):
    def __init__(
        self,
        qa_generator: LLM,
        eval_model: LLM,
        blob_path: Optional[str] = None,
        source: str = "gcp",
        verbose: bool = True,
    ):
        """
        Initialize the Synthesizer with the given LLMs.

        Args:
            blob_path (Optional[str]): The cloud storage prefix/path to the blob representing the PDF file. Can be None.
            qa_generator (LLM): The LLM to use for QA generation.
            source (str): "gcp" or "local".
            verbose (bool): Flag to enable verbose output. Default is True.
            eval_model (LLM): The LLM to use for evaluation.
        """
        super().__init__(
            blob_path=blob_path,
            qa_generator_llm=qa_generator,
            eval_llm=eval_model,
            source=source,
            verbose=verbose,
        )
        self.db = FirestoreService()
        # self.langfuse_service = LangfuseService()
        self.request_id = str(uuid4())
        self.qa_generator = qa_generator
        self.eval_model = eval_model
        self.source = source
        self.verbose = verbose
        if self.verbose:
            my_logger.info("Synthesizer initialized.")

    def create_qa_data(
        self, n_generations: int = 20
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Creates test dataset of Question/Answer pairs given a context for use case (here: RAG) evaluation.
        Contexts are extracted from the provided document blob.

        Args:
            n_generations (int, optional): The number of QA pairs to generate. Defaults to 20.
        Return: Dict object containing the generated QA pairs.
        """
        if self.verbose:
            my_logger.info(f"Creating {n_generations} Q/A test cases 🙃 ...")

        try:
            # Track request in Firestore
            request_data = {
                "uid": self.request_id,
                "blob_path": self.blob_path,
                "created_at": datetime.utcnow().isoformat(),
                "status": "processing",
                "n_generations": n_generations,
            }
            self.db.create(request_data)

            my_logger.info(
                f"Document for request {self.request_id} created in Firestore."
            )

            # Download raw PDFs from Cloud storage
            # and chunk into LangChain compatible documents
            self.preprocess_documents()

            # Generate QA pairs
            self.generate_qa_pairs(n_generations=n_generations)

            # Evaluate the generated QA pairs
            # and create dataset + save CSV to Cloud storage
            self.evaluate_qa_pairs()

            # Update status on success
            success = self.db.update(self.request_id, {"status": "completed"})

            if self.verbose and success:
                my_logger.info(
                    f"Status updated to 'completed' for request {self.request_id}"
                )
                my_logger.info("Question/Answer pairs successfully generated!! 🥳 🎉")

            return self.outputs

        except Exception as e:
            # Update status on failure
            self.db.update(self.request_id, {"status": "failed", "error": str(e)})
            my_logger.error(f"Error: {e}")
            return []

    def save_data(self, file_name: str = "generated_test_data.csv") -> None:
        """
        Saves the generated dataset to a CSV blob on Google Cloud Storage.

        file_name (str): The name of the file to save the dataset to. Defaults to "generated_test_data.csv".
        """
        assert hasattr(self, "generated_questions"), "Dataset not created yet."
        try:
            if self.source == "gcp":
                service = CSVService(
                    root_path="raw_documents", path=file_name, verbose=True
                )
            else:
                service = XLSXService(
                    path="generated_qa_data.xlsx",
                    root_path=glob.DATA_PKG_DIR,
                    verbose=True,
                )
            service.doWrite(X=self.generated_qa)

            if self.verbose:
                my_logger.info(f"Dataset saved to {service.path} 📁")
        except Exception as e:
            if self.verbose:
                my_logger.error(f"Error saving dataset: {e}")
            raise e

    def upload_to_langfuse(self, dataset_name: str) -> None:
        """
        Uploads the generated pandas dataframe to Langfuse.

        Args:
            dataset_name (str): The name of the dataset in Langfuse.
        """
        try:
            # Create or get the dataset
            dataset = self.langfuse_service.createDataset(name=dataset_name)
            # Upload the dataframe to Langfuse
            self.langfuse_service.update_from_pandas(dataset_name, self.generated_qa)
            if self.verbose:
                my_logger.info(f"Dataset uploaded to Langfuse: {dataset_name} 📁")
        except Exception as e:
            if self.verbose:
                my_logger.error(f"Error uploading to Langfuse: {e}")
            raise e
