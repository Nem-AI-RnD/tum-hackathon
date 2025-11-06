from typing import Any, List, Literal, Optional, Union

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

from ai_eval.config import global_config as glob
from ai_eval.config.config import model_list
from ai_eval.services.logger import LoggerFactory

my_logger = LoggerFactory().create_module_logger()


class InitModels:
    """
    A class to initialize and manage models for QA generation, RAG generation,
    evaluation, and embeddings based on the specified model provider.

    Attributes:
        qa_generator (Union[ChatVertexAI, ChatOllama]): The model used for question-answer generation.
        rag_generator (Union[ChatVertexAI, ChatOllama]): The model used for retrieval-augmented generation.
        eval_model (Union[ChatVertexAI, ChatOllama]): The model used for evaluation purposes.
        embedding_model (Union[VertexAIEmbeddings, OllamaEmbeddings]): The embedding model for the provider.

    Methods:
        __init__(model_provider: str, load_models: List[str]): Initializes the specified models based on the provider and configuration.

    Raises:
        ValueError: If the specified model provider or load_models is not supported.
    """

    def __init__(
        self,
        model_provider: str = glob.MODEL_PROVIDER,
        load_models: Optional[
            List[
                Literal[
                    "qa_generator", "rag_generator", "eval_model", "embedding_model"
                ]
            ]
        ] = None,
        embedding_params: Optional[dict] = None,
    ) -> None:
        """
        Initialize the models based on the specified provider and configuration.

        Args:
            model_provider (str): The model provider to use (e.g., "google", "ollama").
            load_models (List[Literal["qa_generator", "rag_generator", "eval_model", "embedding_model"]]):
                A list of models to load. Options are "qa_generator", "rag_generator", "eval_model", "embedding_model", or None to load all models.
            embedding_params (dict, optional): Additional parameters for embedding model initialization.
        """
        if load_models is None:
            load_models = [
                "qa_generator",
                "rag_generator",
                "eval_model",
                "embedding_model",
            ]

        self.qa_generator: Optional[Union[ChatVertexAI, ChatOllama]] = None
        self.rag_generator: Optional[Union[ChatVertexAI, ChatOllama]] = None
        self.eval_model: Optional[Union[ChatVertexAI, ChatOllama]] = None
        self.embedding_model: Optional[Union[VertexAIEmbeddings, OllamaEmbeddings]] = (
            None
        )

        # Models available
        selected_models = model_list["chat_model"][glob.MODEL_PROVIDER]
        chat_model = judge_model = selected_models[list(selected_models.keys())[0]]

        # Embedding models available
        embedding_models = model_list["embedding_model"][glob.MODEL_PROVIDER]
        embedding_model_name = embedding_models[list(embedding_models.keys())[0]]

        # Select the chat/generation and judge models
        match model_provider:
            case "google":
                # judge_model = selected_models[list(selected_models.keys())[1]]

                if "qa_generator" in load_models:
                    self.qa_generator = ChatVertexAI(
                        project=glob.GCP_PROJECT,
                        model_name=chat_model,
                        temperature=0.1,
                        max_retries=2,
                    )
                if "rag_generator" in load_models:
                    self.rag_generator = ChatVertexAI(
                        project=glob.GCP_PROJECT,
                        model_name=chat_model,
                        temperature=0.1,
                        max_retries=2,
                    )
                if "eval_model" in load_models:
                    self.eval_model = ChatVertexAI(
                        project=glob.GCP_PROJECT,
                        model_name=judge_model,
                        temperature=0.2,
                        max_retries=2,
                    )
                if "embedding_model" in load_models:
                    params = embedding_params or {}
                    self.embedding_model = VertexAIEmbeddings(
                        project=glob.GCP_PROJECT,
                        model_name=embedding_model_name,
                        **params,
                    )
            case "ollama":
                if "qa_generator" in load_models:
                    self.qa_generator = ChatOllama(
                        model=chat_model,
                        temperature=0.1,
                    )
                if "rag_generator" in load_models:
                    self.rag_generator = ChatOllama(
                        model=chat_model,
                        temperature=0.1,
                    )
                if "eval_model" in load_models:
                    self.eval_model = ChatOllama(
                        model=judge_model,
                        temperature=0.2,
                    )
                if "embedding_model" in load_models:
                    params = embedding_params or {}
                    self.embedding_model = OllamaEmbeddings(
                        model=embedding_model_name,
                        **params,
                    )
            case _:
                raise ValueError(f"Model provider {glob.MODEL_PROVIDER} not supported.")

        my_logger.info(f"Using chat model: {chat_model}")
        my_logger.info(f"Using judge model: {judge_model}")
        if self.embedding_model:
            my_logger.info(f"Using embedding model: {embedding_model_name}")

        my_logger.info("Models loaded successfully")
