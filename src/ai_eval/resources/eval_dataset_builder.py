from typing import Any, Dict, List

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from langchain.docstore.document import Document
from tqdm.auto import tqdm

from ai_eval.resources.rag_template import RAG
from ai_eval.services.logger import LoggerFactory

my_logger = LoggerFactory().create_module_logger()


class EvalDatasetBuilder:
    """
    Utility class to build an EvaluationDataset for a single-turn RAG pipeline
    (query → answer + retrieved contexts) using DeepEval.
    """

    def __init__(self, rag_instance: RAG) -> None:
        self.rag_instance = rag_instance
        self.data: Dict[str, Any] = {}

    def build_evaluation_dataset(
        self,
        input_contexts: List[str],
        sample_queries: List[str],
        expected_responses: List[str],
    ) -> EvaluationDataset:
        """
        Build the dataset.

        Args:
            input_contexts: list of ground-truth context strings (one per query).
            sample_queries: list of queries.
            expected_responses: list of expected answers.

        Returns:
            EvaluationDataset populated with goldens and test cases.
        """
        docs_processed = [Document(page_content=ctx) for ctx in input_contexts]
        if hasattr(self.rag_instance, "documents"):
            self.rag_instance.documents = docs_processed

        predicted_answers: List[str] = []
        predicted_retrievals: List[List[str]] = []
        goldens: List[Golden] = []

        for query, gt_answer, gt_context in tqdm(
            zip(sample_queries, expected_responses, input_contexts),
            total=len(sample_queries),
            desc="Building test dataset",
        ):
            try:
                answer, relevant_docs = self.rag_instance.answer(query)
            except Exception as e:
                my_logger.warning(f"RAG error for query='{query}': {e}")
                answer = ""
                relevant_docs = []

            # Convert retrieved docs into list of strings
            retrieval_list = [
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in relevant_docs
            ]

            predicted_answers.append(answer)
            predicted_retrievals.append(retrieval_list)

            # Create golden (input, expected output, ground truth context)
            goldens.append(
                Golden(input=query, expected_output=gt_answer, context=[gt_context])
            )

        dataset = EvaluationDataset(goldens=goldens)

        # Add test cases: actual_output + retrieval_context
        for idx, golden in enumerate(dataset.goldens):
            test_case = LLMTestCase(
                input=golden.input,
                expected_output=golden.expected_output,
                actual_output=predicted_answers[idx],
                context=golden.context,
                retrieval_context=predicted_retrievals[idx],
            )
            dataset.add_test_case(test_case)

        self.data["predicted_answers"] = predicted_answers
        self.data["predicted_retrievals"] = predicted_retrievals

        my_logger.info("Evaluation dataset created successfully.")
        return dataset
