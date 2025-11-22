"""
RAG Prototype: Production-Grade RAG with Multimodal Support

This module provides a complete RAG implementation with:
- Pydantic-based configuration with validation
- Text-only and multimodal retrieval modes
- Support for images, diagrams, and tables
- LlamaIndex + Qdrant + Jina embeddings + LangChain integration
- Claude vision capabilities for multimodal understanding
- Flexible mode switching
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import qdrant_client
from langchain.docstore.document import Document

try:  # pragma: no cover - optional dependency
    from langchain_anthropic import ChatAnthropic
except Exception as exc:  # noqa: BLE001 - surface downstream
    ChatAnthropic = None  # type: ignore[assignment]
    _CHAT_ANTHROPIC_IMPORT_ERROR = exc
else:
    _CHAT_ANTHROPIC_IMPORT_ERROR = None
from llama_index.core import (
    Document as LlamaIndexDocument,
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import ImageNode, TextNode
from llama_index.embeddings.jinaai import JinaEmbedding
try:  # pragma: no cover - optional dependency
    from llama_index.llms.langchain import LangChainLLM
except Exception as exc:  # noqa: BLE001 - report later
    LangChainLLM = None  # type: ignore[assignment]
    _LANGCHAIN_LLM_IMPORT_ERROR = exc
else:
    _LANGCHAIN_LLM_IMPORT_ERROR = None

if TYPE_CHECKING:
    from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal

try:
    from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
except ImportError:
    AnthropicMultiModal = None  # type: ignore[assignment, misc]

from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_eval.resources.rag_template import RAG

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    JINA_V4 = "jina-embeddings-v4"  # Multimodal support
    JINA_V3 = "jina-embeddings-v3"
    JINA_V2 = "jina-embeddings-v2-base-en"
    JINA_CLIP = "jina-clip-v1"  # For image embeddings


class LLMModel(str, Enum):
    """Supported LLM models."""

    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_OPUS_3_5 = "claude-3-5-opus-20241022"
    # Legacy models with vision
    CLAUDE_SONNET_3 = "claude-3-sonnet-20240229"
    CLAUDE_OPUS_3 = "claude-3-opus-20240229"


class ResponseModeType(str, Enum):
    """LlamaIndex response synthesis modes."""

    COMPACT = "compact"
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE_SUMMARIZE = "simple_summarize"
    REFINE = "refine"
    GENERATION = "generation"


class RetrievalMode(str, Enum):
    """RAG retrieval modes."""

    TEXT_ONLY = "text_only"
    MULTIMODAL = "multimodal"
    HYBRID = "hybrid"  # Both text and image retrieval


class ParseMode(str, Enum):
    """Document parsing modes."""

    BASIC = "basic"  # Standard text extraction
    MULTIMODAL_LVM = "parse_page_with_lvm"  # LlamaParse with vision model
    FAST = "fast"  # Quick parsing without vision


class RAGConfig(BaseSettings):
    """
    Production-grade RAG configuration with multimodal support.

    New Features:
    - Multimodal retrieval mode
    - Image processing settings
    - Vision model configuration
    - Flexible parsing modes

    Example:
        ```python
        # Text-only mode
        config = RAGConfig(retrieval_mode=RetrievalMode.TEXT_ONLY)

        # Multimodal mode
        config = RAGConfig(
            retrieval_mode=RetrievalMode.MULTIMODAL,
            enable_vision=True,
            vision_llm_model=LLMModel.CLAUDE_SONNET_3_5
        )
        ```
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # API Keys (loaded from environment)
    qdrant_url: str = Field(
        ...,
        description="Qdrant instance URL",
        examples=["https://xyz.qdrant.io:6333"],
    )
    qdrant_api_key: str = Field(..., description="Qdrant API key", repr=False)
    jina_api_key: str = Field(..., description="Jina AI API key", repr=False)
    anthropic_api_key: str = Field(...,
                                   description="Anthropic API key", repr=False)
    llamaparse_api_key: Optional[str] = Field(
        default=None,
        description="LlamaParse API key for multimodal parsing",
        repr=False,
    )

    # Collection settings
    collection_name: str = Field(
        default="documents_collection",
        description="Name of the Qdrant collection",
        min_length=1,
        max_length=255,
    )
    force_recreate: bool = Field(
        default=False,
        description="Force recreate collection if it exists",
    )

    # Retrieval mode settings
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.TEXT_ONLY,
        description="Retrieval mode: text_only, multimodal, or hybrid",
    )
    parse_mode: ParseMode = Field(
        default=ParseMode.BASIC,
        description="Document parsing mode",
    )
    enable_vision: bool = Field(
        default=False,
        description="Enable vision capabilities for multimodal LLM",
    )

    # Embedding settings
    embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.JINA_V4,
        description="Jina AI embedding model (v4 supports multimodal)",
    )
    embedding_dim: int = Field(
        default=2048,
        ge=128,
        le=4096,
        description="Embedding vector dimension",
    )
    embedding_task: str = Field(
        default="retrieval.passage",
        description="Jina AI task type",
    )

    # Image/Multimodal settings
    image_embedding_model: Optional[str] = Field(
        default=None,
        description="Separate model for image embeddings (if different)",
    )
    extract_images: bool = Field(
        default=False,
        description="Extract and process images from documents",
    )
    max_image_size: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Maximum image dimension (px) for processing",
    )
    image_quality: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Image processing quality",
    )

    # Chunking settings
    chunk_size: int = Field(
        default=1024,
        ge=128,
        le=8192,
        description="Maximum chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks in tokens",
    )
    paragraph_separator: str = Field(
        default="\n\n\n",
        description="Separator for paragraph boundaries",
    )

    # Retrieval settings
    top_k: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of documents to retrieve",
    )
    similarity_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity search",
    )
    min_similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )

    # Multimodal retrieval weights
    text_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for text similarity in hybrid mode",
    )
    image_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for image similarity in hybrid mode",
    )

    # LLM settings
    llm_model: LLMModel = Field(
        default=LLMModel.CLAUDE_HAIKU_4_5,
        description="Anthropic Claude model for text generation",
    )
    vision_llm_model: Optional[LLMModel] = Field(
        default=None,
        description="Separate model for vision tasks (if different)",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retries for LLM calls",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens for LLM response",
    )

    # Response settings
    response_mode: ResponseModeType = Field(
        default=ResponseModeType.COMPACT,
        description="Response synthesis mode",
    )
    streaming: bool = Field(
        default=False,
        description="Enable streaming responses",
    )

    # Advanced settings
    enable_metadata_extraction: bool = Field(
        default=True,
        description="Extract and enrich node metadata",
    )
    cache_embeddings: bool = Field(
        default=False,
        description="Cache embeddings for repeated queries",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: ValidationInfo) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1024)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("text_weight", "image_weight")
    @classmethod
    def validate_weights(cls, v: float, info: ValidationInfo) -> float:
        """Validate multimodal retrieval weights."""
        if "text_weight" in info.data and "image_weight" in info.data:
            total = info.data["text_weight"] + \
                info.data.get("image_weight", 0.3)
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                logger.warning(
                    "⚠️  text_weight + image_weight should equal 1.0, got %s", total
                )
        return v

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(
        cls, v: RetrievalMode, info: ValidationInfo
    ) -> RetrievalMode:
        """Validate retrieval mode compatibility."""
        if v in [RetrievalMode.MULTIMODAL, RetrievalMode.HYBRID]:
            # Check if required settings are enabled
            enable_vision = info.data.get("enable_vision", False)
            if not enable_vision:
                logger.warning(
                    "⚠️  %s mode works best with enable_vision=True", v.value
                )
        return v

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        if not v:
            raise ValueError("qdrant_url cannot be empty")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("qdrant_url must start with http:// or https://")
        return v

    @computed_field
    @property
    def effective_chunk_size(self) -> int:
        """Calculate effective chunk size after overlap."""
        return self.chunk_size - self.chunk_overlap

    @computed_field
    @property
    def is_multimodal(self) -> bool:
        """Check if multimodal features are enabled."""
        return self.retrieval_mode in [RetrievalMode.MULTIMODAL, RetrievalMode.HYBRID]

    @computed_field
    @property
    def effective_vision_model(self) -> str:
        """Get the vision LLM model (falls back to main LLM)."""
        if self.vision_llm_model:
            return self.vision_llm_model.value
        return self.llm_model.value

    def validate_config(self) -> None:
        """Perform additional validation checks."""
        # Check all required API keys
        required_keys = {
            "qdrant_url": self.qdrant_url,
            "qdrant_api_key": self.qdrant_api_key,
            "jina_api_key": self.jina_api_key,
            "anthropic_api_key": self.anthropic_api_key,
        }

        missing = [k for k, v in required_keys.items()
                   if not v or v.strip() == ""]
        if missing:
            raise ValueError(
                f"❌ Missing required configuration: {', '.join(missing)}")

        # Check multimodal-specific requirements
        if self.is_multimodal:
            if self.parse_mode == ParseMode.MULTIMODAL_LVM and not self.llamaparse_api_key:
                logger.warning(
                    "⚠️  Multimodal parsing requires llamaparse_api_key. "
                    "Consider setting it for better results."
                )

            # Jina v4 is recommended for multimodal
            if self.embedding_model != EmbeddingModel.JINA_V4:
                logger.warning(
                    "⚠️  %s mode works best with "
                    "jina-embeddings-v4 (current: %s)",
                    self.retrieval_mode.value,
                    self.embedding_model.value,
                )

        logger.info(
            "✅ Configuration validated\n"
            "   Mode: %s\n"
            "   Multimodal: %s\n"
            "   Vision: %s",
            self.retrieval_mode.value,
            self.is_multimodal,
            self.enable_vision,
        )

    def switch_mode(self, new_mode: RetrievalMode) -> None:
        """
        Switch retrieval mode dynamically.

        Args:
            new_mode: New retrieval mode to switch to
        """

        old_mode = self.retrieval_mode
        self.retrieval_mode = new_mode

        # Adjust related settings
        if new_mode == RetrievalMode.TEXT_ONLY:
            self.extract_images = False
        elif new_mode in [RetrievalMode.MULTIMODAL, RetrievalMode.HYBRID]:
            self.extract_images = True
            if not self.enable_vision:
                logger.warning(
                    "⚠️  Consider enabling vision with enable_vision=True")

        logger.info("🔄 Switched mode: %s → %s", old_mode.value, new_mode.value)

    def to_safe_dict(self) -> Dict[str, Any]:
        """Export configuration without sensitive data."""
        return self.model_dump(
            exclude={
                "qdrant_api_key",
                "jina_api_key",
                "anthropic_api_key",
                "llamaparse_api_key",
            }
        )

    def save_to_file(self, path: Union[str, Path], include_computed: bool = True) -> None:
        """
        Save configuration to JSON file (excluding secrets).

        Args:
            path: Path to save configuration
            include_computed: Include computed fields in output
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_safe_dict()

        # Add computed fields if requested
        if include_computed:
            config_dict["_computed"] = {
                "effective_chunk_size": self.effective_chunk_size,
                "is_multimodal": self.is_multimodal,
                "effective_vision_model": self.effective_vision_model,
            }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(config_dict, file, indent=2, ensure_ascii=False)
            file.write('\n')

        logger.info(
            "💾 Configuration saved to: %s (%.2f KB)",
            path,
            path.stat().st_size / 1024
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RAGConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        return cls(**config_dict)


class DeploymentMetadata(BaseModel):
    """Metadata about embedding deployment with multimodal support."""

    collection_name: str
    num_documents: int = Field(ge=0)
    num_nodes: int = Field(ge=0)
    num_text_nodes: int = Field(default=0, ge=0)
    num_image_nodes: int = Field(default=0, ge=0)
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    retrieval_mode: str
    is_multimodal: bool = False
    deployment_timestamp: str
    duration_seconds: float = Field(ge=0.0)
    success: bool = True
    error_message: Optional[str] = None

    @computed_field
    @property
    def nodes_per_document(self) -> float:
        """Average nodes per document."""
        if self.num_documents == 0:
            return 0.0
        return self.num_nodes / self.num_documents

    @computed_field
    @property
    def multimodal_ratio(self) -> Optional[float]:
        """Ratio of image nodes to total nodes."""
        if not self.is_multimodal or self.num_nodes == 0:
            return None
        return self.num_image_nodes / self.num_nodes

    def model_dump_json(self, **kwargs: Any) -> str:
        """Export as formatted JSON string."""
        return super().model_dump_json(indent=2, **kwargs)


class RetrievalResult(BaseModel):
    """Structured retrieval result with multimodal metadata."""

    query: str
    documents: List[Document]
    scores: Optional[List[float]] = None
    num_retrieved: int
    num_text_docs: int = 0
    num_image_docs: int = 0
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat())
    top_k_used: int
    retrieval_mode: str = "text_only"

    @computed_field
    @property
    def has_scores(self) -> bool:
        """Check if scores are available."""
        return self.scores is not None and len(self.scores) > 0

    @computed_field
    @property
    def avg_score(self) -> Optional[float]:
        """Average similarity score."""
        if not self.has_scores:
            return None
        return sum(self.scores) / len(self.scores)

    @computed_field
    @property
    def has_multimodal_content(self) -> bool:
        """Check if result contains multimodal content."""
        return self.num_image_docs > 0


class QADataItem(BaseModel):
    """Structured QA data item matching generated_qa_data_tum.json format."""

    index: int
    question: str
    answer: str
    location_dependency_evaluator_target_answer: str
    context: str
    groundedness_score: int = Field(ge=1, le=5)
    groundedness_eval: str
    question_relevancy_score: int = Field(ge=1, le=5)
    question_relevancy_eval: str
    faithfulness_score: int = Field(ge=1, le=5)
    faithfulness_eval: str

    def model_dump_json(self, **kwargs: Any) -> str:
        """Export as formatted JSON string."""
        return super().model_dump_json(indent=2, **kwargs)


class RAGPrototype(RAG):
    """
    Production-grade RAG with multimodal support.

    Key Features:
    - Switch between text-only and multimodal modes
    - Process images, diagrams, and tables
    - Multimodal embeddings with Jina v4
    - Vision-capable LLMs (Claude 3.5 Sonnet)
    - Hybrid retrieval (text + image)

    Example:
        ```python
        # Text-only mode
        config = RAGConfig(retrieval_mode=RetrievalMode.TEXT_ONLY)
        rag = RAGPrototype(config)
        rag.deploy_embeddings(documents)

        # Switch to multimodal
        rag.switch_to_multimodal()

        # Multimodal query
        answer, docs = rag.answer("Explain the diagram showing X")
        ```
    """

    def __init__(
        self,
        config: RAGConfig,
        llm: Optional[ChatAnthropic] = None,
        documents: Optional[List[Document]] = None,
    ):
        """Initialize RAG Prototype with multimodal support."""
        config.validate_config()
        self.config = config

        # Initialize LLMs
        if llm is None:
            llm = self._create_llm()
        self.llm = llm

        # Initialize vision LLM if needed
        self.vision_llm: Optional[AnthropicMultiModal] = None
        if config.enable_vision:
            if AnthropicMultiModal is None:
                raise ImportError(
                    "llama-index-multi-modal-llms-anthropic is required for vision support. "
                    "Install it with: pip install llama-index-multi-modal-llms-anthropic"
                )
            self.vision_llm = self._create_vision_llm()

        # Initialize parent RAG class
        super().__init__(
            llm=llm,
            documents=documents or [],
            k=config.top_k,
        )

        # Initialize embedding model
        self.embed_model = self._create_embedding_model()
        Settings.embed_model = self.embed_model

        # Initialize Qdrant client
        self.qdrant_client = self._create_qdrant_client()

        # Initialize vector store
        self._setup_vector_store()

        # Node parser
        self.node_parser = self._create_node_parser()

        # Index and retrievers
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None

        # Metadata tracking
        self.deployment_metadata: Optional[DeploymentMetadata] = None
        self._is_deployed: bool = False

        logger.info(
            "✅ RAGPrototype initialized\n"
            "   Mode: %s\n"
            "   Collection: %s\n"
            "   Embedding: %s\n"
            "   LLM: %s\n"
            "   Vision: %s",
            config.retrieval_mode.value,
            config.collection_name,
            config.embedding_model.value,
            config.llm_model.value,
            "Enabled" if config.enable_vision else "Disabled",
        )

    def _create_llm(self) -> ChatAnthropic:
        """Create ChatAnthropic instance from config."""
        if ChatAnthropic is None:
            raise ImportError(
                "langchain_anthropic is required to create ChatAnthropic instances"
            ) from _CHAT_ANTHROPIC_IMPORT_ERROR
        kwargs: Dict[str, Any] = {
            "model": self.config.llm_model.value,
            "temperature": self.config.temperature,
            "max_retries": self.config.max_retries,
            "api_key": self.config.anthropic_api_key,
        }
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens

        return ChatAnthropic(**kwargs)

    def _create_vision_llm(self) -> AnthropicMultiModal:
        """Create multimodal LLM for vision tasks."""
        if AnthropicMultiModal is None:
            raise ImportError(
                "llama-index-multi-modal-llms-anthropic is required for vision support. "
                "Install it with: pip install llama-index-multi-modal-llms-anthropic"
            )
        model = self.config.effective_vision_model
        return AnthropicMultiModal(
            model=model,
            api_key=self.config.anthropic_api_key,
            max_tokens=self.config.max_tokens or 1024,
            temperature=self.config.temperature,
        )

    def _create_embedding_model(self) -> JinaEmbedding:
        """Create JinaEmbedding instance from config."""
        return JinaEmbedding(
            api_key=self.config.jina_api_key,
            model=self.config.embedding_model.value,
            task=self.config.embedding_task,
        )

    def _create_qdrant_client(self) -> qdrant_client.QdrantClient:
        """Create Qdrant client from config."""
        return qdrant_client.QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
        )

    def _create_node_parser(self) -> SentenceSplitter:
        """Create node parser from config."""
        return SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            paragraph_separator=self.config.paragraph_separator,
        )

    def _setup_vector_store(self) -> None:
        """Setup Qdrant vector store."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(
                collection.name == self.config.collection_name for collection in collections
            )

            if collection_exists:
                if self.config.force_recreate:
                    logger.info(
                        "🔄 Deleting existing collection: %s",
                        self.config.collection_name,
                    )
                    self.qdrant_client.delete_collection(
                        self.config.collection_name)
                    collection_exists = False
                else:
                    logger.info(
                        "📦 Using existing collection: %s",
                        self.config.collection_name,
                    )
                    self._is_deployed = True

            self.vector_store = QdrantVectorStore(
                collection_name=self.config.collection_name,
                client=self.qdrant_client,
                vector_size=self.config.embedding_dim,
            )

            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            if not collection_exists:
                logger.info(
                    "📦 Created new collection: %s",
                    self.config.collection_name,
                )

        except Exception as exc:
            logger.error("❌ Error setting up vector store: %s", exc)
            raise

    def switch_to_multimodal(
        self,
        enable_vision: bool = True,
        parse_mode: ParseMode = ParseMode.MULTIMODAL_LVM,
    ) -> None:
        """
        Switch to multimodal retrieval mode.

        Args:
            enable_vision: Enable vision LLM
            parse_mode: Parsing mode to use
        """

        self.config.switch_mode(RetrievalMode.MULTIMODAL)
        self.config.enable_vision = enable_vision
        self.config.parse_mode = parse_mode
        self.config.extract_images = True

        if enable_vision and self.vision_llm is None:
            self.vision_llm = self._create_vision_llm()

        logger.info("✅ Switched to multimodal mode")

    def switch_to_text_only(self) -> None:
        """Switch to text-only retrieval mode."""
        self.config.switch_mode(RetrievalMode.TEXT_ONLY)
        self.config.extract_images = False
        logger.info("✅ Switched to text-only mode")

    def clear_collection(self) -> bool:
        """
        Clear all vectors from the collection without deleting it.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🧹 Clearing collection: %s",
                        self.config.collection_name)

            # Get collection info
            collection_info = self.qdrant_client.get_collection(
                self.config.collection_name
            )

            if collection_info.points_count > 0:
                # Delete all points from collection
                from qdrant_client.models import Filter, FilterSelector
                self.qdrant_client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[]  # Empty filter matches all points
                        )
                    )
                )
                logger.info("✅ Cleared %d points from collection",
                            collection_info.points_count)
            else:
                logger.info("✅ Collection was already empty")

            self._is_deployed = False
            self.index = None
            self.retriever = None
            self.query_engine = None
            self.deployment_metadata = None

            return True

        except Exception as exc:
            logger.error("❌ Error clearing collection: %s", exc)
            return False

    def delete_collection(self) -> bool:
        """
        Delete the entire collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🗑️  Deleting collection: %s",
                        self.config.collection_name)
            self.qdrant_client.delete_collection(self.config.collection_name)

            self._is_deployed = False
            self.index = None
            self.retriever = None
            self.query_engine = None
            self.deployment_metadata = None

            logger.info("✅ Collection deleted successfully")
            return True

        except Exception as exc:
            logger.error("❌ Error deleting collection: %s", exc)
            return False

    def reset_for_redeployment(self, delete_collection: bool = False) -> None:
        """
        Reset the RAG system for fresh deployment.

        Args:
            delete_collection: If True, delete collection. If False, just clear it.
        """
        logger.info("🔄 Resetting RAG system for redeployment...")

        if delete_collection:
            self.delete_collection()
            # Recreate the vector store
            self._setup_vector_store()
        else:
            self.clear_collection()

        logger.info("✅ RAG system reset complete")

    def deploy_embeddings(
        self,
        documents: Union[List[Document], List[LlamaIndexDocument]],
        force_redeploy: bool = False,
        clear_existing: bool = True,
    ) -> Optional[DeploymentMetadata]:
        """
        Deploy embeddings with multimodal support.

        Args:
            documents: List of documents (can include images)
            force_redeploy: Force redeployment even if already deployed
            clear_existing: Clear existing data before deployment (only if force_redeploy=True)

        Returns:
            DeploymentMetadata with statistics
        """

        if self._is_deployed and not force_redeploy:
            logger.warning(
                "⚠️  Already deployed. Use force_redeploy=True to recreate.")
            return self.deployment_metadata

        if not documents:
            raise ValueError("❌ No documents provided")

        # Handle redeployment
        if self._is_deployed and force_redeploy:
            if clear_existing:
                logger.info("🔄 Force redeploy: clearing existing data...")
                self.reset_for_redeployment(delete_collection=False)
            else:
                logger.info("🔄 Force redeploy: adding to existing data...")

        logger.info(
            "🚀 Starting %s embedding deployment for %s documents",
            "multimodal" if self.config.is_multimodal else "text",
            len(documents),
        )
        start_time = datetime.now()

        try:
            # Convert to LlamaIndex documents
            llama_docs = self._convert_to_llama_documents(documents)

            # Parse into nodes (may include image nodes)
            logger.info("📝 Chunking documents into nodes...")
            nodes = self.node_parser.get_nodes_from_documents(llama_docs)

            # Count node types
            text_nodes = sum(
                1 for node in nodes if isinstance(node, (TextNode, type(nodes[0])))
            )
            image_nodes = sum(
                1 for node in nodes if isinstance(node, ImageNode))

            # Enrich metadata
            if self.config.enable_metadata_extraction:
                self._enrich_node_metadata(nodes, start_time)

            avg_nodes_per_doc = len(
                nodes) / len(llama_docs) if llama_docs else 0
            logger.info(
                "✅ Created %s nodes\n"
                "   Text: %s, Images: %s\n"
                "   Avg/doc: %.1f",
                len(nodes),
                text_nodes,
                image_nodes,
                avg_nodes_per_doc,
            )

            # Build index
            logger.info("🔄 Building vector index...")
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                show_progress=True,
            )

            # Create retriever
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.config.top_k,
            )

            # Create query engine
            self._create_query_engine()

            # Metadata
            duration = (datetime.now() - start_time).total_seconds()
            self.deployment_metadata = DeploymentMetadata(
                collection_name=self.config.collection_name,
                num_documents=len(llama_docs),
                num_nodes=len(nodes),
                num_text_nodes=text_nodes,
                num_image_nodes=image_nodes,
                embedding_model=self.config.embedding_model.value,
                embedding_dim=self.config.embedding_dim,
                chunk_size=self.config.chunk_size,
                retrieval_mode=self.config.retrieval_mode.value,
                is_multimodal=self.config.is_multimodal,
                deployment_timestamp=start_time.isoformat(),
                duration_seconds=duration,
                success=True,
            )

            self._is_deployed = True

            logger.info(
                "✅ Deployment complete in %.2fs\n"
                "   Mode: %s\n"
                "   Nodes: %s (%s text, %s images)",
                duration,
                self.config.retrieval_mode.value,
                len(nodes),
                text_nodes,
                image_nodes,
            )

            return self.deployment_metadata

        except Exception as exc:
            error_msg = f"Deployment failed: {exc}"
            logger.error("❌ %s", error_msg)

            self.deployment_metadata = DeploymentMetadata(
                collection_name=self.config.collection_name,
                num_documents=len(documents),
                num_nodes=0,
                embedding_model=self.config.embedding_model.value,
                embedding_dim=self.config.embedding_dim,
                chunk_size=self.config.chunk_size,
                retrieval_mode=self.config.retrieval_mode.value,
                is_multimodal=self.config.is_multimodal,
                deployment_timestamp=start_time.isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=error_msg,
            )

            raise ValueError(error_msg) from exc

    def _create_query_engine(self) -> None:
        """Create query engine (uses vision LLM if available)."""
        llm_to_use: Union[ChatAnthropic, AnthropicMultiModal] = (
            self.vision_llm
            if (self.config.enable_vision and self.vision_llm)
            else self.llm
        )

        if isinstance(llm_to_use, ChatAnthropic):
            if LangChainLLM is None:
                raise ImportError(
                    "llama_index.llms.langchain is required to wrap ChatAnthropic instances"
                ) from _LANGCHAIN_LLM_IMPORT_ERROR
            llama_llm: Union[LangChainLLM, AnthropicMultiModal] = LangChainLLM(
                llm=llm_to_use
            )
        else:
            llama_llm = llm_to_use  # AnthropicMultiModal is already compatible

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=llama_llm,
            response_mode=self.config.response_mode.value,
        )

    def _convert_to_llama_documents(
        self,
        documents: Union[List[Document], List[LlamaIndexDocument]],
    ) -> List[LlamaIndexDocument]:
        """Convert documents to LlamaIndex format."""
        if not documents:
            raise ValueError("❌ No documents provided")

        if isinstance(documents[0], LlamaIndexDocument):
            return documents  # type: ignore[return-value]

        llama_docs = []
        for doc in documents:
            llama_docs.append(
                LlamaIndexDocument(
                    text=doc.page_content,
                    metadata=doc.metadata or {},
                )
            )

        logger.debug("🔄 Converted %s docs to LlamaIndex format",
                     len(documents))
        return llama_docs

    def _enrich_node_metadata(self, nodes: List[Any], deployment_time: datetime) -> None:
        """Enrich nodes with metadata including multimodal info."""
        for idx, node in enumerate(nodes):
            if node.metadata is None:
                node.metadata = {}

            source = node.metadata.get("source", self.config.collection_name)
            page = node.metadata.get("page", "NA")
            node_type = "image" if isinstance(node, ImageNode) else "text"

            node.metadata.update(
                {
                    "chunk_index": idx,
                    "chunk_id": f"{source}_p{page}_c{idx}",
                    "node_type": node_type,
                    "retrieval_mode": self.config.retrieval_mode.value,
                    "deployment_timestamp": deployment_time.isoformat(),
                    "chunk_size": self.config.chunk_size,
                    "embedding_model": self.config.embedding_model.value,
                }
            )

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Retrieve documents (supports multimodal)."""
        self._ensure_deployed()

        if not self.retriever:
            raise ValueError("❌ Retriever not initialized")

        k = top_k or self.config.top_k
        original_k = self.retriever._similarity_top_k
        self.retriever._similarity_top_k = k

        try:
            nodes_with_scores = self.retriever.retrieve(question)

            if return_scores:
                results: List[Tuple[Document, float]] = []
                for node in nodes_with_scores:
                    doc = Document(
                        page_content=node.get_content(),
                        metadata=getattr(node, "metadata", {}) or {},
                    )
                    score = node.score or 0.0

                    if score >= self.config.min_similarity_score:
                        results.append((doc, score))

                logger.debug(
                    "📊 Retrieved %s docs (%s mode)",
                    len(results),
                    self.config.retrieval_mode.value,
                )
                return results

            docs: List[Document] = []
            for node in nodes_with_scores:
                score = node.score or 0.0
                if score >= self.config.min_similarity_score:
                    doc = Document(
                        page_content=node.get_content(),
                        metadata=getattr(node, "metadata", {}) or {},
                    )
                    docs.append(doc)

            return docs

        finally:
            self.retriever._similarity_top_k = original_k

    def retrieve_structured(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> RetrievalResult:
        """Retrieve with structured multimodal metadata."""
        self._ensure_deployed()

        k = top_k or self.config.top_k
        results = self.retrieve(question, top_k=k, return_scores=return_scores)

        if return_scores:
            docs = [doc for doc, _ in results]  # type: ignore[assignment]
            # type: ignore[assignment]
            scores = [score for _, score in results]
        else:
            docs = results  # type: ignore[assignment]
            scores = None

        # Count multimodal content
        num_text = sum(1 for doc in docs if doc.metadata.get(
            "node_type") != "image")
        num_image = sum(1 for doc in docs if doc.metadata.get(
            "node_type") == "image")

        return RetrievalResult(
            query=question,
            documents=docs,
            scores=scores,
            num_retrieved=len(docs),
            num_text_docs=num_text,
            num_image_docs=num_image,
            top_k_used=k,
            retrieval_mode=self.config.retrieval_mode.value,
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate answer (uses vision LLM if enabled)."""
        if self.query_engine is not None:
            response = self.query_engine.query(question)
            return str(response)

        # Fallback
        prompt = (
            "Using the following context, answer the question:\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # Use vision LLM if available and enabled
        llm_to_use: Union[ChatAnthropic, AnthropicMultiModal] = (
            self.vision_llm
            if (self.config.enable_vision and self.vision_llm)
            else self.llm
        )

        answer = llm_to_use.invoke(prompt)
        if hasattr(answer, "content"):
            return answer.content  # type: ignore[return-value]
        return str(answer)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Tuple[str, List[Document]]:
        """Complete RAG pipeline with multimodal support."""
        self._ensure_deployed()

        relevant_docs = self.retrieve(question, top_k=top_k)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        answer = self.generate(question, context)

        return answer, relevant_docs

    def _ensure_deployed(self) -> None:
        """Ensure embeddings are deployed."""
        if not self._is_deployed or self.retriever is None:
            raise ValueError(
                "❌ Retriever not initialized. Call deploy_embeddings() first.")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive collection information.

        Returns:
            Dictionary with collection metadata and statistics
        """
        try:
            collection_info = self.qdrant_client.get_collection(
                self.config.collection_name
            )

            info = {
                "name": self.config.collection_name,
                "vectors_count": collection_info.points_count,
                "vectors_size": collection_info.config.params.vectors.size,
                "status": collection_info.status.value,
                "retrieval_mode": self.config.retrieval_mode.value,
                "is_multimodal": self.config.is_multimodal,
                "is_deployed": self._is_deployed,
            }

            # Add deployment metadata if available
            if self.deployment_metadata:
                info["deployment"] = {
                    "num_documents": self.deployment_metadata.num_documents,
                    "num_nodes": self.deployment_metadata.num_nodes,
                    "num_text_nodes": self.deployment_metadata.num_text_nodes,
                    "num_image_nodes": self.deployment_metadata.num_image_nodes,
                    "deployment_timestamp": self.deployment_metadata.deployment_timestamp,
                    "duration_seconds": self.deployment_metadata.duration_seconds,
                    "success": self.deployment_metadata.success,
                }

            return info

        except Exception as exc:
            logger.error("❌ Error getting collection info: %s", exc)
            return {
                "name": self.config.collection_name,
                "error": str(exc)
            }

    def print_deployment_info(self) -> None:
        """Print formatted deployment information."""
        info = self.get_collection_info()

        if "error" in info:
            logger.info("❌ Collection Error: %s", info["error"])
            return

        output = (
            "\n" + "=" * 60 + "\n"
            "📦 DEPLOYMENT INFORMATION\n"
            "=" * 60 + "\n"
            "\n🏷️  Collection:\n"
            "   Name: {name}\n"
            "   Status: {status}\n"
            "   Vector Count: {vectors_count}\n"
            "   Vector Size: {vectors_size}\n"
            "\n🔧 Configuration:\n"
            "   Retrieval Mode: {retrieval_mode}\n"
            "   Multimodal: {is_multimodal}\n"
            "   Deployed: {is_deployed}\n"
            "{deployment_info}"
            "=" * 60
        ).format(
            name=info["name"],
            status=info["status"],
            vectors_count=info["vectors_count"],
            vectors_size=info["vectors_size"],
            retrieval_mode=info["retrieval_mode"],
            is_multimodal="Yes" if info["is_multimodal"] else "No",
            is_deployed="Yes" if info["is_deployed"] else "No",
            deployment_info=(
                "\n📊 Deployment Stats:\n"
                "   Documents: {num_docs}\n"
                "   Total Nodes: {num_nodes}\n"
                "   Text Nodes: {num_text}\n"
                "   Image Nodes: {num_img}\n"
                "   Duration: {duration:.2f}s\n"
            ).format(
                num_docs=info["deployment"]["num_documents"],
                num_nodes=info["deployment"]["num_nodes"],
                num_text=info["deployment"]["num_text_nodes"],
                num_img=info["deployment"]["num_image_nodes"],
                duration=info["deployment"]["duration_seconds"],
            ) if "deployment" in info else ""
        )
        logger.info("%s", output)

    def export_deployment_report(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """
        Export a comprehensive deployment report.

        Args:
            output_path: Path to save the report (JSON format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "collection_info": self.get_collection_info(),
            "config": self.config.to_safe_dict(),
            "deployment_metadata": (
                self.deployment_metadata.model_dump()
                if self.deployment_metadata else None
            ),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')

        logger.info(
            "📋 Deployment report exported to: %s (%.2f KB)",
            output_path,
            output_path.stat().st_size / 1024
        )

    @property
    def is_deployed(self) -> bool:
        """Check if deployed."""
        return self._is_deployed

    @property
    def is_multimodal(self) -> bool:
        """Check if multimodal mode is active."""
        return self.config.is_multimodal

    @classmethod
    def from_env(
        cls,
        collection_name: str = "documents_collection",
        force_recreate: bool = False,
        retrieval_mode: RetrievalMode = RetrievalMode.TEXT_ONLY,
        **config_overrides: Any,
    ) -> "RAGPrototype":
        """Create from environment with mode selection."""
        config = RAGConfig(
            collection_name=collection_name,
            force_recreate=force_recreate,
            retrieval_mode=retrieval_mode,
            **config_overrides,
        )
        return cls(config=config)

    def generate_qa_data(
        self,
        question: str,
        index: int = 0,
    ) -> QADataItem:
        """
        Generate structured QA data with evaluation metrics.

        This method runs the RAG pipeline and generates evaluation scores
        (groundedness, question relevancy, faithfulness) matching the format
        of generated_qa_data_tum.json.

        Args:
            question: The question to answer
            index: Index for ordering in dataset

        Returns:
            QADataItem with complete evaluation metadata

        Example:
            ```python
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)
            qa_item = rag.generate_qa_data("What is the main topic?", index=0)
            print(qa_item.model_dump_json())
            ```
        """
        self._ensure_deployed()

        # Retrieve context
        relevant_docs = self.retrieve(question, top_k=self.config.top_k)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Generate answer
        answer = self.generate(question, context)

        # Generate evaluation metrics
        groundedness_eval = self._evaluate_groundedness(
            question, answer, context)
        question_relevancy_eval = self._evaluate_question_relevancy(question)
        faithfulness_eval = self._evaluate_faithfulness(answer, context)

        # Create QA data item
        qa_item = QADataItem(
            index=index,
            question=question,
            answer=answer,
            location_dependency_evaluator_target_answer=answer,
            context=context,
            groundedness_score=groundedness_eval["score"],
            groundedness_eval=groundedness_eval["explanation"],
            question_relevancy_score=question_relevancy_eval["score"],
            question_relevancy_eval=question_relevancy_eval["explanation"],
            faithfulness_score=faithfulness_eval["score"],
            faithfulness_eval=faithfulness_eval["explanation"],
        )

        logger.info(
            "✅ Generated QA data item #%d\n"
            "   Question: %s\n"
            "   Groundedness: %d/5\n"
            "   Relevancy: %d/5\n"
            "   Faithfulness: %d/5",
            index,
            question[:60],
            qa_item.groundedness_score,
            qa_item.question_relevancy_score,
            qa_item.faithfulness_score,
        )

        return qa_item

    def _evaluate_groundedness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Evaluate groundedness: whether the answer is supported by context.

        Uses LLM to assess if the answer is grounded in the provided context.
        """
        eval_prompt = PromptTemplate(
            """Evaluate how well the provided answer is grounded in the context.
Groundedness measures whether the answer is directly supported by and can be derived from the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Provide your evaluation as a JSON object with this structure:
{{
  "score": <integer 1-5>,
  "explanation": "<brief explanation of the groundedness assessment>"
}}

Where:
- 1 = Answer is not supported by context or contradicts it
- 2 = Answer has minimal support from context
- 3 = Answer is partially supported by context
- 4 = Answer is mostly supported by context
- 5 = Answer is fully and directly supported by context

Only return the JSON object, no additional text."""
        )

        formatted_prompt = eval_prompt.format(
            context=context,
            question=question,
            answer=answer,
        )

        response = self.llm.invoke(formatted_prompt)
        response_text = response.content if hasattr(
            response, "content") else str(response)

        try:
            result = json.loads(response_text)
            return {
                "score": max(1, min(5, result.get("score", 3))),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse groundedness evaluation: %s", response_text[:100])
            return {
                "score": 3,
                "explanation": response_text[:200],
            }

    def _evaluate_question_relevancy(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Evaluate question relevancy: whether the question is relevant to the domain.

        Assesses if the question is meaningful and relevant to the document domain.
        """
        eval_prompt = PromptTemplate(
            """Evaluate the relevancy and importance of the following question.
Question relevancy measures how well the question addresses meaningful topics within the domain,
and how valuable the answer would be to domain professionals.

Question: {question}

Provide your evaluation as a JSON object with this structure:
{{
  "score": <integer 1-5>,
  "explanation": "<brief explanation of the relevancy assessment>"
}}

Where:
- 1 = Question is not relevant or meaningful to the domain
- 2 = Question has minimal relevance
- 3 = Question is moderately relevant
- 4 = Question is quite relevant and valuable
- 5 = Question is highly relevant and addresses important domain topics

Only return the JSON object, no additional text."""
        )

        formatted_prompt = eval_prompt.format(question=question)
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content if hasattr(
            response, "content") else str(response)

        try:
            result = json.loads(response_text)
            return {
                "score": max(1, min(5, result.get("score", 3))),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse relevancy evaluation: %s", response_text)
            return {
                "score": 3,
                "explanation": response_text[:200],
            }

    def _evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness: whether the answer accurately reflects context facts.

        Assesses if the answer accurately and faithfully represents the information
        in the context without hallucination or distortion.
        """
        eval_prompt = PromptTemplate(
            """Evaluate the faithfulness of the provided answer.
Faithfulness measures whether the answer accurately reflects the facts and information in the context,
without hallucination, distortion, or misrepresentation.

Context:
{context}

Answer: {answer}

Provide your evaluation as a JSON object with this structure:
{{
  "score": <integer 1-5>,
  "explanation": "<brief explanation of the faithfulness assessment>"
}}

Where:
- 1 = Answer contains significant hallucinations or contradicts context
- 2 = Answer has some inaccuracies or misrepresentations
- 3 = Answer is mostly faithful with minor inaccuracies
- 4 = Answer is quite faithful with minimal issues
- 5 = Answer is completely faithful and accurately reflects context

Only return the JSON object, no additional text."""
        )

        formatted_prompt = eval_prompt.format(context=context, answer=answer)
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content if hasattr(
            response, "content") else str(response)

        try:
            result = json.loads(response_text)
            return {
                "score": max(1, min(5, result.get("score", 3))),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse faithfulness evaluation: %s", response_text)
            return {
                "score": 3,
                "explanation": response_text[:200],
            }

    def generate_qa_dataset(
        self,
        questions: List[str],
        start_index: int = 0,
        show_progress: bool = True,
        save_intermediate: Optional[Union[str, Path]] = None,
    ) -> List[QADataItem]:
        """
        Generate a complete QA dataset from multiple questions with progress tracking.

        Args:
            questions: List of questions to generate QA data for
            start_index: Starting index for dataset items
            show_progress: Show progress bar and statistics
            save_intermediate: Path to save intermediate results (useful for long runs)

        Returns:
            List of QADataItem objects

        Example:
            ```python
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)
            questions = ["What is X?", "How do I do Y?"]
            qa_dataset = rag.generate_qa_dataset(
                questions,
                show_progress=True,
                save_intermediate="data/qa_intermediate.json"
            )
            ```
        """
        qa_dataset = []
        total = len(questions)
        errors = 0

        logger.info("🚀 Starting QA dataset generation for %d questions", total)

        for idx, question in enumerate(questions):
            try:
                current_idx = start_index + idx

                if show_progress:
                    logger.info(
                        "📝 Processing %d/%d: %s...",
                        idx + 1,
                        total,
                        question[:60]
                    )

                qa_item = self.generate_qa_data(question, index=current_idx)
                qa_dataset.append(qa_item)

                # Save intermediate results if requested
                if save_intermediate and (idx + 1) % 10 == 0:
                    self.save_qa_dataset(qa_dataset, save_intermediate)
                    logger.info(
                        "💾 Saved intermediate results (%d items)", len(qa_dataset))

            except Exception as e:
                errors += 1
                logger.error(
                    "❌ Error generating QA data for question %d: %s\n   Question: %s",
                    idx,
                    e,
                    question[:100],
                )
                continue

        # Final summary
        success_rate = ((total - errors) / total * 100) if total > 0 else 0
        logger.info(
            "✅ QA dataset generation complete\n"
            "   Total: %d items\n"
            "   Successful: %d\n"
            "   Errors: %d\n"
            "   Success rate: %.1f%%",
            len(qa_dataset),
            len(qa_dataset),
            errors,
            success_rate,
        )

        return qa_dataset

    def save_qa_dataset(
        self,
        qa_dataset: List[QADataItem],
        output_path: Union[str, Path],
        validate: bool = True,
    ) -> None:
        """
        Save QA dataset to JSON file in generated_qa_data_tum.json format.

        Args:
            qa_dataset: List of QADataItem objects to save
            output_path: Path to save the JSON file
            validate: Validate data structure before saving

        Raises:
            ValueError: If validation fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if validate and qa_dataset:
            # Validate that all items have required fields
            required_fields = {
                "index", "question", "answer",
                "location_dependency_evaluator_target_answer",
                "context", "groundedness_score", "groundedness_eval",
                "question_relevancy_score", "question_relevancy_eval",
                "faithfulness_score", "faithfulness_eval"
            }

            for idx, item in enumerate(qa_dataset):
                item_dict = item.model_dump()
                missing = required_fields - set(item_dict.keys())
                if missing:
                    raise ValueError(
                        f"❌ QA item {idx} missing required fields: {missing}"
                    )

                # Validate scores are in range [1, 5]
                for score_field in ["groundedness_score", "question_relevancy_score", "faithfulness_score"]:
                    score = item_dict[score_field]
                    if not (1 <= score <= 5):
                        raise ValueError(
                            f"❌ QA item {idx}: {score_field}={score} must be in range [1, 5]"
                        )

        # Convert to dict format matching generated_qa_data_tum.json
        data = [item.model_dump() for item in qa_dataset]

        # Save with proper formatting (4 spaces indent, no trailing whitespace)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write('\n')  # Add newline at end of file

        logger.info(
            "💾 QA dataset saved to: %s\n"
            "   Items: %d\n"
            "   Size: %.2f KB",
            output_path,
            len(data),
            output_path.stat().st_size / 1024
        )

    @classmethod
    def load_qa_dataset(
        cls,
        input_path: Union[str, Path],
        validate: bool = True,
    ) -> List[QADataItem]:
        """
        Load QA dataset from JSON file.

        Args:
            input_path: Path to the JSON file
            validate: Validate loaded data

        Returns:
            List of QADataItem objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If validation fails
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"❌ File not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("❌ JSON file must contain a list of QA items")

        qa_dataset = []
        for idx, item_dict in enumerate(data):
            try:
                qa_item = QADataItem(**item_dict)
                qa_dataset.append(qa_item)
            except Exception as e:
                logger.error("❌ Error loading QA item %d: %s", idx, e)
                if validate:
                    raise ValueError(
                        f"❌ Invalid QA item at index {idx}: {e}") from e
                continue

        logger.info(
            "✅ Loaded QA dataset from: %s\n"
            "   Items: %d\n"
            "   Size: %.2f KB",
            input_path,
            len(qa_dataset),
            input_path.stat().st_size / 1024
        )

        return qa_dataset

    def append_to_qa_dataset(
        self,
        output_path: Union[str, Path],
        questions: List[str],
        show_progress: bool = True,
    ) -> List[QADataItem]:
        """
        Append new QA items to an existing dataset file.

        Args:
            output_path: Path to existing QA dataset JSON file
            questions: List of questions to generate and append
            show_progress: Show progress during generation

        Returns:
            Complete list of QADataItem objects (old + new)

        Example:
            ```python
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)

            # Append 10 new questions to existing dataset
            new_questions = ["Question 1?", "Question 2?"]
            complete_dataset = rag.append_to_qa_dataset(
                "data/generated_qa_data_tum.json",
                new_questions
            )
            ```
        """
        output_path = Path(output_path)

        # Load existing dataset if file exists
        if output_path.exists():
            logger.info("📂 Loading existing dataset from: %s", output_path)
            existing_dataset = self.load_qa_dataset(
                output_path, validate=False)
            start_index = max(
                [item.index for item in existing_dataset], default=-1) + 1
            logger.info("   Found %d existing items, starting at index %d",
                        len(existing_dataset), start_index)
        else:
            logger.info("📄 Creating new dataset at: %s", output_path)
            existing_dataset = []
            start_index = 0

        # Generate new QA items
        new_items = self.generate_qa_dataset(
            questions=questions,
            start_index=start_index,
            show_progress=show_progress,
        )

        # Combine and save
        complete_dataset = existing_dataset + new_items
        self.save_qa_dataset(complete_dataset, output_path)

        logger.info(
            "✅ Dataset updated\n"
            "   Previous: %d items\n"
            "   New: %d items\n"
            "   Total: %d items",
            len(existing_dataset),
            len(new_items),
            len(complete_dataset)
        )

        return complete_dataset

    def merge_qa_datasets(
        self,
        output_path: Union[str, Path],
        *input_paths: Union[str, Path],
        deduplicate: bool = True,
    ) -> List[QADataItem]:
        """
        Merge multiple QA dataset files into one.

        Args:
            output_path: Path to save merged dataset
            *input_paths: Paths to input dataset files
            deduplicate: Remove duplicate questions

        Returns:
            Merged list of QADataItem objects
        """
        all_items: List[QADataItem] = []
        seen_questions: set = set()

        logger.info("🔄 Merging %d QA datasets...", len(input_paths))

        for input_path in input_paths:
            items = self.load_qa_dataset(Path(input_path), validate=False)

            for item in items:
                if deduplicate:
                    q_normalized = item.question.strip().lower()
                    if q_normalized in seen_questions:
                        logger.debug("⏭️  Skipping duplicate: %s",
                                     item.question[:60])
                        continue
                    seen_questions.add(q_normalized)

                all_items.append(item)

        # Re-index items
        for idx, item in enumerate(all_items):
            item.index = idx

        # Save merged dataset
        self.save_qa_dataset(all_items, output_path)

        logger.info(
            "✅ Merged %d datasets into %d items\n"
            "   Saved to: %s",
            len(input_paths),
            len(all_items),
            output_path
        )

        return all_items

    def get_qa_statistics(
        self,
        qa_dataset: List[QADataItem],
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a QA dataset.

        Args:
            qa_dataset: List of QADataItem objects

        Returns:
            Dictionary with statistics
        """
        if not qa_dataset:
            return {"total_items": 0}

        scores = {
            "groundedness": [item.groundedness_score for item in qa_dataset],
            "question_relevancy": [item.question_relevancy_score for item in qa_dataset],
            "faithfulness": [item.faithfulness_score for item in qa_dataset],
        }

        stats = {
            "total_items": len(qa_dataset),
            "score_statistics": {},
            "quality_metrics": {},
        }

        # Calculate statistics for each score type
        for score_name, score_list in scores.items():
            avg_score = sum(score_list) / len(score_list)
            stats["score_statistics"][score_name] = {
                "average": round(avg_score, 2),
                "min": min(score_list),
                "max": max(score_list),
                "distribution": {
                    str(i): score_list.count(i) for i in range(1, 6)
                }
            }

        # Overall quality metrics
        avg_all = sum(
            item.groundedness_score +
            item.question_relevancy_score +
            item.faithfulness_score
            for item in qa_dataset
        ) / (len(qa_dataset) * 3)

        high_quality = sum(
            1 for item in qa_dataset
            if all([
                item.groundedness_score >= 4,
                item.question_relevancy_score >= 4,
                item.faithfulness_score >= 4,
            ])
        )

        stats["quality_metrics"] = {
            "overall_average": round(avg_all, 2),
            "high_quality_items": high_quality,
            "high_quality_percentage": round(high_quality / len(qa_dataset) * 100, 1),
        }

        # Content statistics
        stats["content_statistics"] = {
            "avg_question_length": round(
                sum(len(item.question)
                    for item in qa_dataset) / len(qa_dataset), 1
            ),
            "avg_answer_length": round(
                sum(len(item.answer)
                    for item in qa_dataset) / len(qa_dataset), 1
            ),
            "avg_context_length": round(
                sum(len(item.context)
                    for item in qa_dataset) / len(qa_dataset), 1
            ),
        }

        return stats

    def print_qa_statistics(
        self,
        qa_dataset: List[QADataItem],
    ) -> None:
        """
        Print formatted QA dataset statistics.

        Args:
            qa_dataset: List of QADataItem objects
        """
        stats = self.get_qa_statistics(qa_dataset)

        if stats["total_items"] == 0:
            print("Dataset is empty")
            return

        output = (
            "\n" + "=" * 60 + "\n"
            "📊 QA DATASET STATISTICS\n"
            "=" * 60 + "\n"
            "\n📈 Overall Metrics:\n"
            "   Total Items: {total_items}\n"
            "   Overall Quality: {overall_avg}/5.0\n"
            "   High Quality Items: {high_quality} ({high_quality_pct}%)\n"
            "\n📊 Score Averages:\n"
            "   Groundedness: {groundedness_avg}/5.0\n"
            "   Question Relevancy: {relevancy_avg}/5.0\n"
            "   Faithfulness: {faithfulness_avg}/5.0\n"
            "\n📝 Content Statistics:\n"
            "   Avg Question Length: {q_len} chars\n"
            "   Avg Answer Length: {a_len} chars\n"
            "   Avg Context Length: {c_len} chars\n"
            "=" * 60
        ).format(
            total_items=stats["total_items"],
            overall_avg=stats["quality_metrics"]["overall_average"],
            high_quality=stats["quality_metrics"]["high_quality_items"],
            high_quality_pct=stats["quality_metrics"]["high_quality_percentage"],
            groundedness_avg=stats["score_statistics"]["groundedness"]["average"],
            relevancy_avg=stats["score_statistics"]["question_relevancy"]["average"],
            faithfulness_avg=stats["score_statistics"]["faithfulness"]["average"],
            q_len=stats["content_statistics"]["avg_question_length"],
            a_len=stats["content_statistics"]["avg_answer_length"],
            c_len=stats["content_statistics"]["avg_context_length"],
        )
        print(output)

    def to_deepeval_format(
        self,
        qa_dataset: List[QADataItem],
    ) -> List[Dict[str, Any]]:
        """
        Convert QA dataset to DeepEval-compatible format.

        Args:
            qa_dataset: List of QADataItem objects

        Returns:
            List of dictionaries formatted for DeepEval

        Example:
            ```python
            rag = RAGPrototype.from_env()
            qa_dataset = rag.load_qa_dataset("data/generated_qa_data_tum.json")
            deepeval_data = rag.to_deepeval_format(qa_dataset)
            ```
        """
        deepeval_cases = []

        for item in qa_dataset:
            case = {
                "input": item.question,
                "expected_output": item.answer,
                "actual_output": item.answer,  # Same as expected for generated data
                "context": [item.context],
                "retrieval_context": [item.context],
                # Include evaluation scores as metadata
                "metadata": {
                    "index": item.index,
                    "groundedness_score": item.groundedness_score,
                    "question_relevancy_score": item.question_relevancy_score,
                    "faithfulness_score": item.faithfulness_score,
                    "target_answer": item.location_dependency_evaluator_target_answer,
                }
            }
            deepeval_cases.append(case)

        return deepeval_cases

    def build_deepeval_dataset(
        self,
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        run_rag: bool = True,
        show_progress: bool = True,
    ) -> tuple[List[Dict[str, Any]], Optional[List[QADataItem]]]:
        """
        Build a DeepEval-compatible dataset by running the RAG pipeline.

        This is the main method for integration with the setup.ipynb workflow.

        Args:
            questions: List of questions to evaluate
            expected_answers: Optional list of expected answers (ground truth)
            ground_truth_contexts: Optional list of ground truth contexts
            run_rag: Whether to run the RAG pipeline to get actual outputs
            show_progress: Show progress during processing

        Returns:
            Tuple of (deepeval_cases, qa_items) where:
            - deepeval_cases: List ready for DeepEval evaluation
            - qa_items: Optional list of QADataItem objects (if generated)

        Example:
            ```python
            # For use with setup.ipynb evaluation pipeline
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)

            deepeval_cases, qa_items = rag.build_deepeval_dataset(
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
                run_rag=True
            )
            ```
        """
        self._ensure_deployed()

        deepeval_cases = []
        qa_items = [] if run_rag else None

        total = len(questions)
        if show_progress:
            logger.info(
                "🔄 Building DeepEval dataset for %d questions...", total)

        for idx, question in enumerate(questions):
            if show_progress and (idx + 1) % 10 == 0:
                logger.info("   Progress: %d/%d (%.1f%%)", idx +
                            1, total, (idx + 1) / total * 100)

            try:
                # Get expected values
                expected_answer = expected_answers[idx] if expected_answers else ""
                gt_context = ground_truth_contexts[idx] if ground_truth_contexts else ""

                if run_rag:
                    # Run RAG pipeline
                    answer, relevant_docs = self.answer(question)
                    retrieval_contexts = [
                        doc.page_content for doc in relevant_docs]
                    actual_output = answer
                else:
                    # Use expected answer as actual output
                    actual_output = expected_answer
                    retrieval_contexts = [gt_context] if gt_context else []

                # Create DeepEval case
                case = {
                    "input": question,
                    "expected_output": expected_answer,
                    "actual_output": actual_output,
                    "context": [gt_context] if gt_context else [],
                    "retrieval_context": retrieval_contexts,
                    "metadata": {
                        "index": idx,
                        "num_retrieved_docs": len(retrieval_contexts),
                    }
                }
                deepeval_cases.append(case)

                # Create QA item if running RAG
                if run_rag and qa_items is not None:
                    # Generate evaluation metrics
                    context_str = "\n\n".join(retrieval_contexts)
                    groundedness_eval = self._evaluate_groundedness(
                        question, answer, context_str
                    )
                    question_relevancy_eval = self._evaluate_question_relevancy(
                        question)
                    faithfulness_eval = self._evaluate_faithfulness(
                        answer, context_str)

                    qa_item = QADataItem(
                        index=idx,
                        question=question,
                        answer=answer,
                        location_dependency_evaluator_target_answer=expected_answer or answer,
                        context=context_str,
                        groundedness_score=groundedness_eval["score"],
                        groundedness_eval=groundedness_eval["explanation"],
                        question_relevancy_score=question_relevancy_eval["score"],
                        question_relevancy_eval=question_relevancy_eval["explanation"],
                        faithfulness_score=faithfulness_eval["score"],
                        faithfulness_eval=faithfulness_eval["explanation"],
                    )
                    qa_items.append(qa_item)

            except Exception as e:
                logger.error(
                    "❌ Error processing question %d: %s\n   Question: %s",
                    idx,
                    e,
                    question[:100]
                )
                continue

        if show_progress:
            logger.info(
                "✅ DeepEval dataset built: %d cases created",
                len(deepeval_cases)
            )

        return deepeval_cases, qa_items

    def run_deepeval_evaluation(
        self,
        questions: List[str],
        expected_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete DeepEval evaluation pipeline.

        This method integrates with the setup.ipynb evaluation workflow
        and returns results compatible with DeepEvalScorer.

        Args:
            questions: List of questions to evaluate
            expected_answers: Optional list of expected answers
            ground_truth_contexts: Optional list of ground truth contexts
            save_results: Save results to file
            output_dir: Directory to save results (default: data/)

        Returns:
            Dictionary with evaluation results

        Example:
            ```python
            # Complete evaluation in one call
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)

            results = rag.run_deepeval_evaluation(
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
                save_results=True
            )
            ```
        """
        logger.info("🎯 Starting complete DeepEval evaluation...")

        # Build DeepEval dataset
        deepeval_cases, qa_items = self.build_deepeval_dataset(
            questions=questions,
            expected_answers=expected_answers,
            ground_truth_contexts=ground_truth_contexts,
            run_rag=True,
            show_progress=True,
        )

        # Calculate metrics summary
        if qa_items:
            stats = self.get_qa_statistics(qa_items)
        else:
            stats = {}

        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_cases": len(deepeval_cases),
            "collection_name": self.config.collection_name,
            "retrieval_mode": self.config.retrieval_mode.value,
            "llm_model": self.config.llm_model.value,
            "embedding_model": self.config.embedding_model.value,
            "deepeval_cases": deepeval_cases,
            "qa_statistics": stats,
        }

        # Save results if requested
        if save_results:
            output_dir = Path(output_dir) if output_dir else Path("data")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save DeepEval cases
            deepeval_path = output_dir / "deepeval_cases.json"
            with open(deepeval_path, "w", encoding="utf-8") as f:
                json.dump(deepeval_cases, f, indent=2, ensure_ascii=False)
                f.write('\n')
            logger.info("💾 DeepEval cases saved to: %s", deepeval_path)

            # Save QA items if available
            if qa_items:
                qa_path = output_dir / "evaluation_qa_dataset.json"
                self.save_qa_dataset(qa_items, qa_path)

            # Save results summary
            results_path = output_dir / "evaluation_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v for k, v in results.items() if k != "deepeval_cases"},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
                f.write('\n')
            logger.info("💾 Results summary saved to: %s", results_path)

        logger.info("✅ DeepEval evaluation complete!")
        return results

    def compare_with_baseline(
        self,
        baseline_results_path: Union[str, Path],
        current_results: Optional[Dict[str, Any]] = None,
        questions: Optional[List[str]] = None,
        expected_answers: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare current RAG performance with baseline results.

        Args:
            baseline_results_path: Path to baseline evaluation results
            current_results: Optional pre-computed current results
            questions: Questions to evaluate (if current_results not provided)
            expected_answers: Expected answers (if current_results not provided)
            ground_truth_contexts: Ground truth contexts (if current_results not provided)

        Returns:
            Dictionary with comparison metrics

        Example:
            ```python
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)

            comparison = rag.compare_with_baseline(
                baseline_results_path="data/baseline_results.json",
                questions=sample_queries,
                expected_answers=expected_responses,
                ground_truth_contexts=ground_truth_contexts,
            )
            ```
        """
        logger.info("📊 Comparing with baseline results...")

        # Load baseline
        baseline_path = Path(baseline_results_path)
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"❌ Baseline file not found: {baseline_path}")

        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        # Get current results if not provided
        if current_results is None:
            if not questions:
                raise ValueError(
                    "❌ Must provide either current_results or questions")
            current_results = self.run_deepeval_evaluation(
                questions=questions,
                expected_answers=expected_answers,
                ground_truth_contexts=ground_truth_contexts,
                save_results=False,
            )

        # Compare statistics
        baseline_stats = baseline.get("qa_statistics", {})
        current_stats = current_results.get("qa_statistics", {})

        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "baseline_model": baseline.get("llm_model", "unknown"),
            "current_model": current_results.get("llm_model", "unknown"),
            "num_cases": {
                "baseline": baseline.get("num_cases", 0),
                "current": current_results.get("num_cases", 0),
            },
            "score_comparison": {},
        }

        # Compare score averages
        if baseline_stats and current_stats:
            for score_type in ["groundedness", "question_relevancy", "faithfulness"]:
                baseline_avg = baseline_stats.get("score_statistics", {}).get(
                    score_type, {}
                ).get("average", 0)
                current_avg = current_stats.get("score_statistics", {}).get(
                    score_type, {}
                ).get("average", 0)

                improvement = current_avg - baseline_avg
                improvement_pct = (improvement / baseline_avg *
                                   100) if baseline_avg > 0 else 0

                comparison["score_comparison"][score_type] = {
                    "baseline": baseline_avg,
                    "current": current_avg,
                    "improvement": round(improvement, 2),
                    "improvement_percentage": round(improvement_pct, 1),
                }

        logger.info("✅ Comparison complete")
        return comparison

    def export_for_setup_notebook(
        self,
        output_dir: Union[str, Path] = "data",
    ) -> Dict[str, Path]:
        """
        Export all necessary files for setup.ipynb evaluation workflow.

        This convenience method exports everything needed for the notebook:
        - Deployment configuration
        - Collection information
        - Ready-to-use evaluation templates

        Args:
            output_dir: Directory to save exports

        Returns:
            Dictionary mapping export type to file path

        Example:
            ```python
            rag = RAGPrototype.from_env()
            rag.deploy_embeddings(documents)

            exports = rag.export_for_setup_notebook(output_dir="data/exports")
            print(f"Config: {exports['config']}")
            print(f"Info: {exports['collection_info']}")
            ```
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exports = {}

        # Export configuration
        config_path = output_dir / f"{self.config.collection_name}_config.json"
        self.config.save_to_file(config_path, include_computed=True)
        exports["config"] = config_path

        # Export collection info
        info_path = output_dir / f"{self.config.collection_name}_info.json"
        info = self.get_collection_info()
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            f.write('\n')
        exports["collection_info"] = info_path

        # Export deployment report
        if self.deployment_metadata:
            report_path = output_dir / \
                f"{self.config.collection_name}_deployment.json"
            self.export_deployment_report(report_path)
            exports["deployment_report"] = report_path

        logger.info(
            "📦 Exported %d files for setup.ipynb to: %s",
            len(exports),
            output_dir
        )

        return exports


__all__ = [
    "RAGConfig",
    "RAGPrototype",
    "DeploymentMetadata",
    "RetrievalResult",
    "QADataItem",
    "EmbeddingModel",
    "LLMModel",
    "ResponseModeType",
    "RetrievalMode",
    "ParseMode",
]
