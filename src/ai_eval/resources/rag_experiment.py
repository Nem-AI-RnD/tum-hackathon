import logging
from typing import List
from uuid import uuid4
from ai_eval.config import global_config as glob
from ai_eval.config.config import rag_model_list
from ai_eval.resources.rag_models import RAGModels
from langchain.docstore.document import Document
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    BaseMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from ai_eval.services.langfuse import LangfuseService
from langchain_ollama import OllamaEmbeddings
from langchain_google_vertexai import ChatVertexAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAGExperiments:
    """
    A class to create and manage Langfuse experiments for RAG retriever or generation evaluation.

    Attributes:
        experiment_type (str): The type of experiment ('retriever' or 'generation').
        metrics (List[BaseMetric]): List of evaluation metrics.
    """

    def __init__(self, experiment_type: str) -> None:
        """
        Initialize the RAGExperiment with the specified type.

        Args:
            experiment_type (str): Type of experiment, must be 'retriever' or 'generation'.

        Raises:
            ValueError: If experiment_type is not valid.
        """
        if experiment_type not in ["retriever", "generation"]:
            raise ValueError("Type must be 'retriever' or 'generation'")
        self.request_id: str = str(uuid4())
        self.experiment_type: str = experiment_type
        self.dataset = glob.LANGFUSE_DATASET_NAME
        self.metrics: List[BaseMetric] = []
        self.langfuse = LangfuseService()
        self.groups = [
            key
            for key in rag_model_list[self.experiment_type].keys()
            if key not in ("metrics", "eval_model", "embedding_model")
        ]
        self.context = self._initialize_context()
        self.eval_model = self._initialize_model()
        self.metrics = self._initialize_metrics()

    def _initialize_context(self) -> List[Document]:
        """Initialize the context from the dataset."""
        data = self.langfuse.read(self.dataset)
        return [
            Document(
                page_content=item.input["context"].strip(), metadata={"source": "test"}
            )
            for item in data.items
        ]

    def _initialize_model(self) -> DeepEvalBaseLLM:
        """Initialize the evaluation model based on the provider."""
        eval_model_name = rag_model_list[self.experiment_type]["eval_model"]
        match glob.MODEL_PROVIDER:
            case "google":
                model = ChatVertexAI(
                    project=glob.GCP_PROJECT,
                    model_name=eval_model_name[glob.MODEL_PROVIDER],
                )
            case "ollama":
                model = OllamaEmbeddings(model=eval_model_name)
            case _:
                raise ValueError(f"Model provider {glob.MODEL_PROVIDER} not supported.")
        return CustomAIModel(model)

    def _initialize_metrics(self) -> List[BaseMetric]:
        """Initialize the metrics for evaluation."""
        metrics_config = rag_model_list[self.experiment_type]["metrics"]
        metric_classes = {
            "AnswerRelevancyMetric": AnswerRelevancyMetric,
            "FaithfulnessMetric": FaithfulnessMetric,
            "ContextualRecallMetric": ContextualRecallMetric,
            "ContextualRelevanceMetric": ContextualRelevancyMetric,
            "ContextualPrecisionMetric": ContextualPrecisionMetric,
        }
        return [
            metric_classes[metric_name](model=self.eval_model)
            for metric_name in metrics_config.keys()
            if metric_name in metric_classes
        ]

    def _evaluate_and_log_metrics(
        self, item: object, actual_output: str, predicted_context: str, trace_id: str
    ) -> None:
        """
        Evaluate the test case and log metrics.
        """
        try:
            test_case = LLMTestCase(
                input=item.input["question"],
                actual_output=actual_output,
                expected_output=item.expected_output,
                context=[item.input["context"]],
                retrieval_context=[predicted_context],
            )
            results = evaluate([test_case], self.metrics)

            for test in results.test_results:
                for metric in test.metrics_data:
                    try:
                        self.langfuse.client.score(
                            name=metric.name,
                            value=metric.score,
                            trace_id=trace_id,
                            data_type="NUMERIC",
                            metadata=metric.reason,
                        )
                    except Exception as e:
                        logger.error(f"Failed to log metric {metric.name}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def create_experiment(self, enable_retriever_generation: bool) -> None:
        """
        Create and run the Langfuse experiment for RAG evaluation.

        This method:
        1. Initializes the experiment with the configured dataset
        2. For each group, creates and runs experiments with different models
        3. Logs context retrieval and generation results
        4. Evaluates and logs metrics for each test case

        Args:
            enable_retriever_generation (bool): Flag to enable retriever generation mode.

        Raises:
            ValueError: If the dataset is not available
        """
        try:
            logger.info(f"Initialized {self.experiment_type} experiment in Langfuse")
            if not self.dataset:
                raise ValueError("Dataset name is not initialized or invalid.")
            dataset = self.langfuse.read(self.dataset)
            if dataset is None:
                raise ValueError("Dataset not available")
            context_model = None

            for group in self.groups:
                try:
                    # Initialize model configuration
                    key = list(rag_model_list[self.experiment_type][group].keys())[0]
                    model_name = rag_model_list[self.experiment_type][group][key]
                    if enable_retriever_generation:
                        experiment_name = f"retriever_generation-{group}-{model_name}"
                    else:
                        experiment_name = f"{self.experiment_type}-{group}-{model_name}"

                    # Initialize models
                    try:
                        model = RAGModels(self.experiment_type, group, self.context)
                        models = rag_model_list[self.experiment_type]
                        if self.experiment_type == "generation":
                            if enable_retriever_generation:
                                if (
                                    "embedding_model" in models
                                    and models['embedding_model'] is not None
                                ):
                                    key = list(models['embedding_model'].keys())[0]
                                    if models['embedding_model'][key] is not None:
                                        context_model = RAGModels(
                                            self.experiment_type,
                                            'embedding_model',
                                            self.context,
                                        )
                    except Exception as model_error:
                        logger.error(
                            f"Failed to initialize models for group {group}: {str(model_error)}"
                        )
                        continue

                    # Process dataset items
                    for item in dataset.items:
                        try:
                            with item.observe(run_name=experiment_name) as trace_id:
                                trace = self.langfuse.client.trace(
                                    name=experiment_name, id=trace_id
                                )
                                retriever_span = trace.span(
                                    name="context_retrieval",
                                    input=item.input["question"],
                                    metadata={"model": experiment_name},
                                )

                                if self.experiment_type == 'retriever':
                                    # Log generation
                                    generation = trace.generation(
                                        name="generation",
                                        model=experiment_name,
                                        input=item.input["context"],
                                    )
                                    predicted_context = model.callRetrieverModel(
                                        item.input["question"]
                                    )
                                    retriever_span.end(output=predicted_context)
                                    generation.end(output=predicted_context)
                                    trace.update(
                                        input=item.input["question"],
                                        output=item.expected_output,
                                    )
                                    # Evaluate metrics
                                    self._evaluate_and_log_metrics(
                                        item, "NA", predicted_context, trace_id
                                    )
                                else:
                                    if enable_retriever_generation:
                                        if context_model is not None:
                                            logger.info("Embedding model is provided")
                                            context = context_model.callRetrieverModel(
                                                item.input["question"]
                                            )
                                            predicted_context = context
                                            response = model.callLLMModel(
                                                item.input["question"],
                                                predicted_context,
                                            )
                                        else:
                                            logger.info(
                                                "No embedding model for generation"
                                            )

                                    else:
                                        logger.info("No embedding model")
                                        response = model.callLLMModel(
                                            item.input["question"],
                                            item.input["context"],
                                        )
                                        predicted_context = item.input["context"]

                                    retriever_span.end(output=predicted_context)
                                    generation = trace.generation(
                                        name="generation",
                                        model=f"{experiment_name}",
                                        input=response.content,
                                    )
                                    generation.end(output=response.content)
                                    trace.update(
                                        input=item.input["question"],
                                        output=response.content,
                                    )
                                    self._evaluate_and_log_metrics(
                                        item,
                                        response.content,
                                        predicted_context,
                                        trace_id,
                                    )
                        except Exception as item_error:
                            logger.error(f"Failed to process item: {str(item_error)}")
                            continue
                except Exception as group_error:
                    logger.error(
                        f"Failed to process group '{group}' during experiment '{self.experiment_type}': {str(group_error)}"
                    )
                    continue
        except ValueError as ve:
            logger.error(f"Dataset error: {str(ve)}")
            raise


class CustomAIModel(DeepEvalBaseLLM):
    """
    A custom AI model wrapper for DeepEvalBaseLLM.
    """

    def __init__(self, model: object) -> None:
        try:
            self.model = model
        except Exception as e:
            logger.error(f"Failed to initialize CustomAIModel: {str(e)}")
            raise

    def get_model_name(self) -> str:
        return "Langchain model"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate response from the model with error handling.

        Args:
            prompt (str): Input prompt for the model

        Returns:
            str: Generated response content

        Raises:
            Exception: If model invocation fails
        """
        try:
            return self.model.invoke(prompt).content
        except Exception as e:
            logger.error(f"Model generation failed: {str(e)}")
            raise

    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate response from the model with error handling.

        Args:
            prompt (str): Input prompt for the model

        Returns:
            str: Generated response content

        Raises:
            Exception: If async model prediction fails
        """
        try:
            chat_model = self.load_model()
            res = await chat_model.apredict(prompt)
            return res
        except Exception as e:
            logger.error(f"Async model generation failed: {str(e)}")
            raise
