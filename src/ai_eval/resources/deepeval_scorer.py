from collections import defaultdict
from typing import Dict, List

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from ai_eval.config import config
from ai_eval.config import global_config as glob
from ai_eval.resources.get_models import InitModels
from ai_eval.services.file import JSONService
from ai_eval.services.logger import LoggerFactory
from ai_eval.utils.utils import LangChainDeepEvalModel, retry

my_logger = LoggerFactory().create_module_logger()


class DeepEvalScorer:
    def __init__(self, evaluation_dataset: EvaluationDataset):
        """
        Initialize the DeepEvalScorer with an evaluation dataset.

        :param evaluation_dataset: The dataset to evaluate, created using the Predictor class.
        """
        self.evaluation_dataset = evaluation_dataset
        # Do NOT rebuild the evaluation dataset here; just use the provided one.
        
        # Get the first configured output service (Opik or Langfuse)
        output_services = config.io["output"]["service"]
        service_name = next(iter(output_services.keys()))  # Get first service
        config_out = output_services[service_name]
        
        self.metric_config = config_out["metrics"]
        self.models = InitModels()
        self.deepeval_model = LangChainDeepEvalModel(model=self.models.eval_model)
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> List[BaseMetric]:
        """
        Dynamically initialize the DeepEval metrics based on the configuration in input_output.yaml.

        :return: A list of initialized metrics.
        """
        metrics = []

        # Mapping metric names to their corresponding classes
        metric_classes = {
            "AnswerRelevancyMetric": AnswerRelevancyMetric,
            "FaithfulnessMetric": FaithfulnessMetric,
            "ContextualRecallMetric": ContextualRecallMetric,
            # "ContextualRelevanceMetric": ContextualRelevancyMetric,
            "ContextualPrecisionMetric": ContextualPrecisionMetric,
        }

        for metric_name, params in self.metric_config.items():
            if metric_name in metric_classes:
                metric_class = metric_classes[metric_name]
                metrics.append(metric_class(model=self.deepeval_model, **params))

        return metrics

    @retry
    def _retryable_evaluate(self) -> dict:
        """
        Helper function to call the evaluate function with retry logic.

        :return: The evaluation results.
        """
        # Use test_cases directly from the provided EvaluationDataset
        test_cases = getattr(self.evaluation_dataset, "test_cases", [])
        return evaluate(
            test_cases,
            metrics=self.metrics,
        )

    # @track(name=glob.OPIK_SPAN_NAME, project_name=glob.OPIK_PROJECT_NAME)
    def calculate_scores(self) -> dict:
        """
        Evaluate the dataset using the initialized metrics and log the results to Opik.

        :return: The evaluation results.
        """
        try:
            # Evaluate the dataset
            my_logger.info("Starting evaluation...")
            self.results = self._retryable_evaluate()
            my_logger.info("Evaluation completed successfully!")
            return self.results
        except Exception as e:
            my_logger.error(f"Evaluation failed: {str(e)}")
            return {}

    def get_overall_metrics(self) -> Dict[str, float]:
        """
        Get the overall metric scores from the evaluation results.
        Calculates the average score for each metric across all test cases.

        :return: A dictionary mapping metric names to their overall scores.
        """
        assert (
            self.results is not None
        ), "Results have not been calculated yet. Run calculate_scores() first."

        # Collect all metric scores by metric name using defaultdict
        metric_scores = defaultdict(list)

        for test_result in self.results.test_results:
            for metric_data in test_result.metrics_data:
                metric_scores[metric_data.name].append(metric_data.score)

        # Calculate average for each metric in a single comprehension
        overall_metrics = {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in metric_scores.items()
        }

        # Calculate overall performance as the average of all metrics
        overall_metrics['Average Performance'] = sum(overall_metrics.values()) / len(
            overall_metrics
        )

        return overall_metrics

    def get_summary(self, save_to_file: bool = False) -> Dict[int, dict]:
        """
        Generates a summary of test results, optionally saving it to a JSON file.

        Args:
            save_to_file (bool, optional): If True, saves the summary to a JSON file. Defaults to False.

        Returns:
            Dict[int, dict]: A dictionary where each key is the test index and the value is a dictionary containing:
                - "query": The input query for the test.
                - "actual_output": The actual output produced.
                - "expected_output": The expected output for the test.
                - "actual_context": The context used during the test.
                - "retrieval_context": The context retrieved for the test.
                - "metrics": A dictionary of metric names and their corresponding scores.
        """
        assert self.results is not None, "Results have not been calculated yet."
        test_summary = dict()
        for i in range(len(self.results.test_results)):
            scores = {
                metric.name: metric.score
                for metric in self.results.test_results[i].metrics_data
            }
            test_summary[i] = {
                "query": self.results.test_results[i].input,
                "actual_output": self.results.test_results[i].actual_output,
                "expected_output": self.results.test_results[i].expected_output,
                "actual_context": self.results.test_results[i].context,
                "retrieval_context": self.results.test_results[i].retrieval_context,
                "metrics": scores,
            }
        my_logger.info("Summary of test results generated successfully!")

        if save_to_file:
            json = JSONService(
                path="deepeval_results.json", root_path=glob.DATA_PKG_DIR, verbose=True
            )
            json.doWrite(test_summary)

        return test_summary
